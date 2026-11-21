"""Tests du cache disque des inférences Timestral (chantier #14).

Le vrai modèle n'est JAMAIS chargé : ``torch`` et ``timestral`` sont remplacés
par des modules factices dans ``sys.modules`` le temps de la construction de
``MistralTimestral``, et le ``_modele`` injecté est un moteur factice qui compte
ses appels ``forecast``. Chaque test écrit dans un dossier cache temporaire.
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.inference_timestral import DOSSIER_CACHE, MistralTimestral


CONTEXTE = np.arange(12, dtype=np.float32).reshape(2, 6)
HORIZON = 4


class _ResultatFactice:
    def __init__(self, quantiles, niveaux):
        self.quantiles = quantiles
        self.quantile_levels = niveaux


class _ModeleFactice:
    """Moteur factice : compte les appels, renvoie des quantiles déterministes."""

    def __init__(self):
        self.appels = 0
        self.leve_erreur = False
        self.renvoie_nan = False

    def forecast(self, contexte, prediction_length, past_covariates=None, known_covariates=None):
        self.appels += 1
        if self.leve_erreur:
            raise RuntimeError("échec simulé du modèle")
        base = np.asarray(contexte, dtype=np.float32).sum(axis=-1)[:, None, None]
        base = base + np.arange(1, int(prediction_length) + 1)[None, None, :]
        quantiles = np.stack([base - 2.0, base, base + 2.0], axis=-1).astype(np.float32)
        if self.renvoie_nan:
            quantiles[..., 0] = np.nan
        return _ResultatFactice(quantiles, np.array([0.1, 0.5, 0.9], dtype=np.float32))


class CacheTimestralTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dossier = Path(self._tmp.name) / "cache_timestral"
        self.modele_factice = _ModeleFactice()

        test = self
        faux_torch = types.ModuleType("torch")
        faux_torch.set_num_threads = lambda *args, **kwargs: None
        faux_timestral = types.ModuleType("timestral")

        class _TimestralForecasterFactice:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                # résolu à l'appel : changer test.modele_factice change le moteur injecté
                return test.modele_factice

        faux_timestral.TimestralForecaster = _TimestralForecasterFactice
        self._faux_modules = {"torch": faux_torch, "timestral": faux_timestral}

    def _construire(self, **kwargs):
        """Construit un MistralTimestral SANS charger le vrai modèle."""
        kwargs.setdefault("dossier_cache", self.dossier)
        with patch.dict(sys.modules, self._faux_modules):
            return MistralTimestral(**kwargs)

    def test_deux_appels_identiques_un_seul_calcul(self):
        moteur = self._construire()
        q1, n1 = moteur.prevoir(CONTEXTE, HORIZON)
        q2, n2 = moteur.prevoir(CONTEXTE.copy(), HORIZON)   # mêmes octets float32
        q3, _ = moteur.prevoir(CONTEXTE.tolist(), HORIZON)   # liste -> mêmes float32
        self.assertEqual(self.modele_factice.appels, 1)
        for q in (q2, q3):
            np.testing.assert_array_equal(q, q1)
        np.testing.assert_array_equal(n2, n1)

    def test_entrees_differentes_plusieurs_calculs(self):
        moteur = self._construire()
        moteur.prevoir(CONTEXTE, HORIZON)                                        # 1
        moteur.prevoir(CONTEXTE + 1, HORIZON)                                    # 2 : contexte
        moteur.prevoir(CONTEXTE, HORIZON + 1)                                    # 3 : horizon
        moteur.prevoir(CONTEXTE, HORIZON, bornes=(0, 20))                        # 4 : bornes
        moteur.prevoir(CONTEXTE, HORIZON, passe=np.ones((2, 6), np.float32))     # 5 : passe
        moteur.prevoir(CONTEXTE, HORIZON, calendrier=np.zeros((2, 6, 2), np.float32))  # 6 : calendrier
        self.assertEqual(self.modele_factice.appels, 6)

    def test_bornes_dans_la_cle_et_valeur_clippee_en_cache(self):
        moteur = self._construire()
        q_sans, _ = moteur.prevoir(CONTEXTE, HORIZON)             # bornes (0, None)
        q_clip, _ = moteur.prevoir(CONTEXTE, HORIZON, bornes=(0, 20))
        self.assertEqual(self.modele_factice.appels, 2)
        self.assertTrue((q_sans > 20).any())
        np.testing.assert_array_equal(q_clip, np.clip(q_sans, 0, 20))
        # rejoué : servi par le cache, mêmes valeurs clippées
        q_rejeu, _ = moteur.prevoir(CONTEXTE, HORIZON, bornes=(0, 20))
        self.assertEqual(self.modele_factice.appels, 2)
        np.testing.assert_array_equal(q_rejeu, q_clip)

    def test_cache_desactive_par_parametre(self):
        moteur = self._construire(cache=False)
        self.assertFalse(moteur._cache_actif)
        moteur.prevoir(CONTEXTE, HORIZON)
        moteur.prevoir(CONTEXTE, HORIZON)
        self.assertEqual(self.modele_factice.appels, 2)
        self.assertFalse(self.dossier.exists())  # rien d'écrit sur le disque

    def test_cache_desactive_par_env(self):
        with patch.dict(os.environ, {"TIMESTRAL_CACHE": "0"}):
            moteur = self._construire()
        self.assertFalse(moteur._cache_actif)
        moteur.prevoir(CONTEXTE, HORIZON)
        moteur.prevoir(CONTEXTE, HORIZON)
        self.assertEqual(self.modele_factice.appels, 2)

    def test_parametre_explicite_gagne_sur_env(self):
        with patch.dict(os.environ, {"TIMESTRAL_CACHE": "0"}):
            moteur = self._construire(cache=True)
        self.assertTrue(moteur._cache_actif)
        moteur.prevoir(CONTEXTE, HORIZON)
        moteur.prevoir(CONTEXTE, HORIZON)
        self.assertEqual(self.modele_factice.appels, 1)

    def test_persistance_disque_nouvelle_instance(self):
        premier = self._construire()
        q1, n1 = premier.prevoir(CONTEXTE, HORIZON)
        self.assertEqual(self.modele_factice.appels, 1)
        self.assertEqual(len(list(self.dossier.glob("*.npz"))), 1)

        self.modele_factice = _ModeleFactice()  # nouveau moteur, compteur à zéro
        second = self._construire()
        q2, n2 = second.prevoir(CONTEXTE, HORIZON)
        self.assertEqual(self.modele_factice.appels, 0)  # lu sur le disque
        np.testing.assert_array_equal(q2, q1)
        np.testing.assert_array_equal(n2, n1)

    def test_erreur_du_modele_jamais_mise_en_cache(self):
        moteur = self._construire()
        self.modele_factice.leve_erreur = True
        with self.assertRaises(RuntimeError):
            moteur.prevoir(CONTEXTE, HORIZON)
        self.assertFalse(self.dossier.exists())

        self.modele_factice.leve_erreur = False
        moteur.prevoir(CONTEXTE, HORIZON)  # recalculé malgré l'échec précédent
        self.assertEqual(self.modele_factice.appels, 2)
        moteur.prevoir(CONTEXTE, HORIZON)  # puis servi par le cache
        self.assertEqual(self.modele_factice.appels, 2)

    def test_quantiles_non_finis_jamais_mis_en_cache(self):
        moteur = self._construire()
        self.modele_factice.renvoie_nan = True
        with self.assertRaises(ValueError):
            moteur.prevoir(CONTEXTE, HORIZON)
        self.assertFalse(self.dossier.exists())

        self.modele_factice.renvoie_nan = False
        q, _ = moteur.prevoir(CONTEXTE, HORIZON)
        self.assertTrue(np.isfinite(q).all())
        moteur.prevoir(CONTEXTE, HORIZON)
        self.assertEqual(self.modele_factice.appels, 2)

    def test_fichier_cache_corrompu_recalcule(self):
        moteur = self._construire()
        moteur.prevoir(CONTEXTE, HORIZON)
        self.assertEqual(self.modele_factice.appels, 1)
        for f in self.dossier.glob("*.npz"):
            f.write_bytes(b"ceci n'est pas un npz")
        q, _ = moteur.prevoir(CONTEXTE, HORIZON)
        self.assertEqual(self.modele_factice.appels, 2)
        self.assertTrue(np.isfinite(q).all())

    def test_quantiles_methode_utilise_le_cache(self):
        moteur = self._construire()
        calendrier = np.zeros((2, 6 + HORIZON, 2), np.float32)
        a = moteur.quantiles(CONTEXTE, HORIZON)
        b = moteur.quantiles(CONTEXTE, HORIZON, calendrier=calendrier)
        c = moteur.quantiles(CONTEXTE, HORIZON, calendrier=calendrier)
        self.assertEqual(self.modele_factice.appels, 2)  # a, puis b, c en cache
        self.assertEqual(a.shape, (2, 1, HORIZON, 3))
        np.testing.assert_array_equal(b, c)

    def test_dossier_cache_par_defaut(self):
        with patch.dict(sys.modules, self._faux_modules):
            moteur = MistralTimestral()
        self.assertTrue(moteur._cache_actif)
        self.assertEqual(moteur._dossier_cache, DOSSIER_CACHE)
        self.assertEqual(
            DOSSIER_CACHE,
            Path(__file__).resolve().parents[1] / "data" / "cache" / "timestral",
        )


if __name__ == "__main__":
    unittest.main()
