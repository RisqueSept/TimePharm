"""Tests des alertes actionnables et de l'indicateur de fraîcheur (chantiers #9 et #10).

- `alertes_actionnables` : fonction pure, filtre exactement les lignes dont le score
  atteint le seuil a priori (src/seuils.py), avec le type d'alerte déduit de la
  colonne `degradation` ;
- `metadonnees_fraicheur` : dates max des relevés ANSM et des volumes Medic'AM,
  révision du modèle lue dans le manifeste si présent ;
- anti-fuite minimal : les alertes à une origine ne dépendent pas des relevés
  publiés après cette origine (même logique que tests/test_fuites_pharma.py).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.modele_pharma import alertes_actionnables, backtest, metadonnees_fraicheur
from src.seuils import SEUIL_ALERTE_RISQUE
from test_fuites_pharma import T, ira_modifiee, mini_ira, mini_stocks, mini_volumes, stocks_modifies


def mini_predictions():
    """Petit jeu de prévisions hors échantillon, scores autour du seuil."""
    return pd.DataFrame({
        "origine": pd.to_datetime(["2025-01-10", "2025-01-10", "2025-01-17", "2025-01-17"]),
        "produit": ["amox250", "azithro", "amox250", "clarithro50"],
        "horizon": [14, 14, 28, 28],
        "timestral": [0.92, 0.41, SEUIL_ALERTE_RISQUE, 0.73],
        "degradation": [np.nan, 0.88, np.nan, 0.61],
        "grossistes": [2, 0, 1, 2],
        "observé": [1, 0, 0, 1],
    })


class AlertesActionnablesTest(unittest.TestCase):
    def test_filtre_exactement_les_scores_au_dessus_du_seuil(self):
        df = mini_predictions()
        copie = df.copy(deep=True)
        alertes = alertes_actionnables(df)
        pd.testing.assert_frame_equal(df, copie)  # fonction pure : entrée non modifiée
        attendu = df.loc[df.timestral >= SEUIL_ALERTE_RISQUE]
        self.assertGreater(len(attendu), 0)
        self.assertEqual(len(alertes), len(attendu))
        self.assertEqual(
            sorted(zip(alertes.produit, alertes.horizon)),
            sorted(zip(attendu.produit, attendu.horizon)),
        )
        self.assertTrue((alertes.timestral >= SEUIL_ALERTE_RISQUE).all())
        # azithro (0,41 < seuil) est exclu, la ligne à exactement le seuil est incluse.
        self.assertNotIn("azithro", set(alertes.produit))
        self.assertIn(("amox250", 28), set(zip(alertes.produit, alertes.horizon)))

    def test_tri_par_score_decroissant_et_seuil_affiche(self):
        alertes = alertes_actionnables(mini_predictions())
        self.assertEqual(alertes.timestral.tolist(), sorted(alertes.timestral, reverse=True))
        self.assertTrue((alertes.seuil == SEUIL_ALERTE_RISQUE).all())

    def test_type_alerte_depuis_la_colonne_degradation(self):
        alertes = alertes_actionnables(mini_predictions())
        types = {f"{p}|{h}": t for p, h, t in zip(alertes.produit, alertes.horizon, alertes.type_alerte)}
        self.assertEqual(types["clarithro50|28"], "nouvelle dégradation")  # degradation non nulle
        self.assertEqual(types["amox250|14"], "tension connue")           # degradation nulle
        self.assertEqual(types["amox250|28"], "tension connue")

    def test_seuil_parametre_modifie_le_filtre(self):
        df = mini_predictions()
        self.assertEqual(len(alertes_actionnables(df, seuil=0.95)), 0)
        self.assertEqual(len(alertes_actionnables(df, seuil=0.0)), len(df))
        self.assertTrue((alertes_actionnables(df, seuil=0.9).seuil == 0.9).all())


class MetadonneesFraicheurTest(unittest.TestCase):
    def test_dates_max_et_revision_du_manifeste(self):
        stocks, volumes = mini_stocks(), mini_volumes(24)
        manifest = {"run": "pharma", "date_utc": "2026-11-21T10:00:00+00:00",
                    "revision_modele": "abcdef1234567890"}
        meta = metadonnees_fraicheur(stocks, volumes, manifest)
        self.assertEqual(pd.Timestamp(meta["dernier_releve_ansm"]), stocks.observation.max())
        self.assertEqual(pd.Timestamp(meta["dernier_releve_disponible"]), stocks.disponible.max())
        self.assertEqual(pd.Timestamp(meta["dernier_mois_medicam"]), volumes.mois.max())
        self.assertEqual(meta["revision_modele"], "abcdef1234567890")
        self.assertEqual(meta["date_generation"], "2026-11-21T10:00:00+00:00")
        json.dumps(meta)  # le bandeau doit être JSON-sérialisable

    def test_manifest_absent_revision_none_et_date_generation_presente(self):
        meta = metadonnees_fraicheur(mini_stocks(), mini_volumes(24), None)
        self.assertIsNone(meta["revision_modele"])
        self.assertTrue(meta["date_generation"])
        self.assertIn("T", meta["date_generation"])  # horodatage ISO

    def test_manifest_lu_depuis_le_fichier_si_present(self):
        stocks, volumes = mini_stocks(), mini_volumes(24)
        with tempfile.TemporaryDirectory() as tmp:
            chemin = Path(tmp) / "manifest.json"
            chemin.write_text(json.dumps({
                "date_utc": "2026-01-02T03:04:05+00:00", "revision_modele": "deadbeefcafe",
            }), encoding="utf-8")
            meta = metadonnees_fraicheur(stocks, volumes, chemin)
            self.assertEqual(meta["revision_modele"], "deadbeefcafe")
            self.assertEqual(meta["date_generation"], "2026-01-02T03:04:05+00:00")
            absent = Path(tmp) / "manifest_absent.json"
            meta_absent = metadonnees_fraicheur(stocks, volumes, absent)
            self.assertIsNone(meta_absent["revision_modele"])


class AlertesFuiteTest(unittest.TestCase):
    """Les alertes à une origine <= T ne changent pas si les relevés publiés après T changent."""

    def test_alertes_a_origine_anterieure_inchangees(self):
        stocks, ira = mini_stocks(), mini_ira()
        bt = backtest(stocks, ira)
        bt_mod = backtest(stocks_modifies(stocks), ira_modifiee(ira))
        # `contexte` joue le rôle du score de risque sur le backtest brut.
        a1 = alertes_actionnables(bt, seuil=0.0, colonne_score="contexte")
        a2 = alertes_actionnables(bt_mod, seuil=0.0, colonne_score="contexte")
        commun = a1.origine <= T
        self.assertGreater(commun.sum(), 0)
        # Colonnes cibles : inconnues au départ par définition, exclues de la comparaison.
        cibles = ["cible", "cible_disponible", "delai_reel", "observé", "source_cible"]
        colonnes = [c for c in a1.columns if c not in cibles]
        pd.testing.assert_frame_equal(
            a1.loc[commun, colonnes].reset_index(drop=True),
            a2.loc[a2.origine <= T, colonnes].reset_index(drop=True),
        )

    def test_memes_alertes_au_seuil_reel_a_origine_anterieure(self):
        stocks, ira = mini_stocks(), mini_ira()
        bt = backtest(stocks, ira)
        bt_mod = backtest(stocks_modifies(stocks), ira_modifiee(ira))
        cles = ["origine", "produit", "horizon", "contexte", "type_alerte"]
        d1 = alertes_actionnables(bt, colonne_score="contexte")
        d2 = alertes_actionnables(bt_mod, colonne_score="contexte")
        avant = d1.loc[d1.origine <= T, cles].reset_index(drop=True)
        apres = d2.loc[d2.origine <= T, cles].reset_index(drop=True)
        pd.testing.assert_frame_equal(avant, apres)


if __name__ == "__main__":
    unittest.main()
