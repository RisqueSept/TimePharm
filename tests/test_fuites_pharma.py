"""Tests anti-fuite temporelle et contrôles de chronologie pour le volet pharma.

Principe : modifier toutes les données publiées après une date T ne doit changer
aucune prévision faite à une origine <= T. Si ces tests cassent, une information
du futur fuite dans les entrées du modèle.
"""

import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.modele_pharma import backtest, points_evaluation, previsions_timestral
from src.modele_volumes_pharma import DELAI_MOIS, estimer, prevoir_volumes, previsions_timestral as previsions_volumes


# Assez tard pour que des origines de la seconde saison aient un apprentissage
# (saison antérieure, 8 exemples de chaque état) tout en restant avant la fin des données.
T = pd.Timestamp("2025-01-15")
PRODUITS_TEST = ["amox250", "azithro"]


def mini_stocks():
    """Relevés hebdomadaires synthétiques sur deux saisons, deux produits."""
    lignes = []
    for p_idx, produit in enumerate(PRODUITS_TEST):
        for debut, fin in [(pd.Timestamp("2023-10-02"), pd.Timestamp("2024-04-29")),
                           (pd.Timestamp("2024-10-07"), pd.Timestamp("2025-04-28"))]:
            for obs in pd.date_range(debut, fin, freq="7D"):
                annee = obs.year - (obs.month < 7)
                semaine = obs.isocalendar().week
                grossistes = 0 if (semaine + p_idx) % 4 == 0 else 1
                officines = np.nan if semaine % 7 == 3 else 1 + (semaine + p_idx) % 2
                lignes.append({
                    "observation": obs, "produit": produit,
                    "disponible": obs + pd.Timedelta(days=7),
                    "grossistes": grossistes, "industrie": 1 + (semaine + p_idx) % 2,
                    "officines": officines,
                    "saison": f"{annee}–{str(annee + 1)[-2:]}",
                    "source": "bulletin_test.pdf",
                })
    return pd.DataFrame(lignes)


def mini_ira():
    semaines = pd.date_range("2023-09-25", "2025-05-12", freq="7D")
    inc = 100 + 80 * np.sin(np.arange(len(semaines)) * 0.7)
    ira = pd.DataFrame({
        "semaine": semaines, "inc100": inc,
        "disponible": semaines + pd.Timedelta(days=10),
    })
    ira["ira_log"] = np.log1p(ira.inc100)
    ira["ira_evolution"] = ira.ira_log.diff()
    return ira


def stocks_modifies(stocks):
    """Inverse toutes les valeurs publiées après T (disponible > T)."""
    mod = stocks.copy()
    masque = mod.disponible > T
    mod.loc[masque, "grossistes"] = 1 - mod.loc[masque, "grossistes"]
    mod.loc[masque, ["industrie", "officines"]] = 2
    return mod


def ira_modifiee(ira):
    mod = ira.copy()
    masque = mod.disponible > T
    mod.loc[masque, "inc100"] = 99999.0
    mod.loc[masque, "ira_log"] = np.log1p(99999.0)
    mod.loc[masque, "ira_evolution"] = 0.0
    return mod


class MoteurFictif:
    """Prévisions qui dépendent des entrées reçues : une fuite change le score."""

    def prevoir(self, cibles, horizon, *, passe=None, calendrier=None, bornes=(0, None)):
        n = cibles.shape[0]
        niveaux = np.array([0.1, 0.3, 0.7, 0.9])
        base = float(np.nanmean(cibles))
        if passe is not None:
            base += float(np.nanmean(passe))
        vals = np.clip(np.array([0.2, 0.4, 0.6, 0.8]) + 0.01 * base, 0.0, 1.0)
        return np.broadcast_to(vals, (n, 1, horizon, 4)).astype(np.float32).copy(), niveaux


class MoteurQuantiles:
    """Capture les historiques passés à Timestral et renvoie des quantiles constants."""

    def __init__(self):
        self.captures = []

    def quantiles(self, valeurs, pas, *, calendrier=None):
        self.captures.append(np.array(valeurs, dtype=float))
        return np.full((valeurs.shape[0], 1, pas, 3), 1000.0)

    def prevoir(self, cibles, horizon, *, passe=None, calendrier=None, bornes=(0, None)):
        return MoteurFictif().prevoir(cibles, horizon, passe=passe, calendrier=calendrier)


class FuitesTensionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stocks = mini_stocks()
        cls.ira = mini_ira()

    def test_points_evaluation_ignore_les_releves_publies_apres_l_origine(self):
        avant = points_evaluation(self.stocks, self.ira, 14)
        apres = points_evaluation(stocks_modifies(self.stocks), self.ira, 14)
        commun = (avant.origine <= T) & (avant.cible_disponible <= T)
        self.assertGreater(commun.sum(), 0)
        pd.testing.assert_frame_equal(
            avant.loc[commun].reset_index(drop=True),
            apres.loc[commun].reset_index(drop=True),
        )

    def test_points_evaluation_ignore_les_incidences_publiees_apres_l_origine(self):
        avant = points_evaluation(self.stocks, self.ira, 14)
        apres = points_evaluation(self.stocks, ira_modifiee(self.ira), 14)
        commun = avant.origine <= T
        self.assertGreater(commun.sum(), 0)
        pd.testing.assert_frame_equal(
            avant.loc[commun].reset_index(drop=True),
            apres.loc[commun].reset_index(drop=True),
        )

    def test_backtest_apprentissage_strictement_anterieur_et_sans_fuite(self):
        bt = backtest(self.stocks, self.ira)
        self.assertGreater(len(bt), 0)
        self.assertTrue((bt.derniere_cible_connue < bt.origine).all(),
                        "une cible publiée à l'origine ou après entre dans l'apprentissage")
        bt_mod = backtest(stocks_modifies(self.stocks), ira_modifiee(self.ira))
        commun = bt.origine <= T
        self.assertGreater(commun.sum(), 0)
        # La cible (observé, dates de cible) est par définition inconnue au départ :
        # elle peut être publiée après T sans que ce soit une fuite. On exclut ces
        # colonnes et on exige l'identique sur toutes les entrées du modèle.
        cibles = ["cible", "cible_disponible", "delai_reel", "observé", "source_cible"]
        pd.testing.assert_frame_equal(
            bt.loc[commun].drop(columns=cibles).reset_index(drop=True),
            bt_mod.loc[commun].drop(columns=cibles).reset_index(drop=True),
        )

    def test_previsions_timestral_ignore_les_donnees_publiees_apres_l_origine(self):
        bt = backtest(self.stocks, self.ira)
        p1 = previsions_timestral(self.stocks, self.ira, bt, MoteurFictif())
        p2 = previsions_timestral(stocks_modifies(self.stocks), ira_modifiee(self.ira), bt, MoteurFictif())
        commun = p1.origine <= T
        self.assertGreater(commun.sum(), 0)
        self.assertTrue(np.allclose(p1.loc[commun, "timestral"], p2.loc[commun, "timestral"]),
                        "les scores à des origines <= T changent si les données publiées après T fuient")


class ChronologieTensionsTest(unittest.TestCase):
    def test_cible_dans_la_fenetre_autour_de_l_horizon(self):
        stocks, ira = mini_stocks(), mini_ira()
        for horizon in (14, 28):
            points = points_evaluation(stocks, ira, horizon)
            self.assertGreater(len(points), 0)
            ecart = (points.cible - points.origine).dt.days
            self.assertTrue(((ecart >= horizon - 3) & (ecart <= horizon + 3)).all())
            self.assertTrue((points.delai_reel == ecart).all())


def mini_volumes(n_mois=24, produits=("a", "b"), base_ordinale=False):
    mois = pd.period_range("2024-01", periods=n_mois, freq="M")
    lignes = []
    for p in produits:
        for m in mois:
            boites = m.ordinal if base_ordinale else 1000 + m.ordinal % 97
            lignes.append({"produit": p, "mois": m.start_time, "boites": boites})
    return pd.DataFrame(lignes)


class FuitesVolumesTest(unittest.TestCase):
    def test_estimer_ignore_les_mois_apres_le_cutoff(self):
        serie = pd.Series({m: 1000 + i for i, m in
                           enumerate(pd.period_range("2023-01", periods=36, freq="M"))})
        origine = pd.Period("2025-11", freq="M")
        reference = estimer(serie, origine, 1)
        self.assertIsNotNone(reference)
        modifiee = serie.copy()
        modifiee.loc[origine - DELAI_MOIS + 1:] *= 1000
        self.assertEqual(reference, estimer(modifiee, origine, 1))

    def test_previsions_timestral_ne_contient_que_les_mois_connus(self):
        volumes = mini_volumes(24, base_ordinale=True)
        lignes = pd.DataFrame({
            "origine": [pd.Timestamp("2025-11-15"), pd.Timestamp("2025-11-15")],
            "mois": [pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01")],
            "produit": ["a", "b"], "observe": [0, 0],
        })
        moteur = MoteurQuantiles()
        previsions_volumes(volumes, lignes, moteur)
        cutoff = pd.Period("2025-11", freq="M") - DELAI_MOIS
        self.assertEqual(len(moteur.captures), 1)
        self.assertLessEqual(int(np.nanmax(moteur.captures[0])), cutoff.ordinal,
                             "l'historique transmis au modèle dépasse le mois de coupure")

    def test_prevoir_volumes_ignore_les_mois_apres_cutoff(self):
        volumes = mini_volumes(24)
        depart = pd.Timestamp("2026-01-01")
        test1, fut1 = prevoir_volumes(volumes, depart, MoteurQuantiles())
        modifie = volumes.copy()
        modifie.loc[modifie.mois >= pd.Timestamp("2025-12-01"), "boites"] = 999999
        test2, fut2 = prevoir_volumes(modifie, depart, MoteurQuantiles())
        # `observe` est la cible réelle du mois modifié : exclu de la comparaison.
        pd.testing.assert_frame_equal(
            test1.drop(columns="observe").reset_index(drop=True),
            test2.drop(columns="observe").reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(fut1.reset_index(drop=True), fut2.reset_index(drop=True))

    def test_prevoir_volumes_structure_horizons_et_colonnes(self):
        test, fut = prevoir_volumes(mini_volumes(30), pd.Timestamp("2026-01-01"), MoteurQuantiles())
        self.assertEqual(set(test.horizon.unique()), {1, 2, 3})
        self.assertTrue({"saisonnier", "ajuste", "timestral", "timestral_q10", "timestral_q90",
                         "observe", "produit", "mois"}.issubset(test.columns))
        self.assertGreater(len(fut), 0)
        self.assertNotIn("observe", fut.columns)


if __name__ == "__main__":
    unittest.main()
