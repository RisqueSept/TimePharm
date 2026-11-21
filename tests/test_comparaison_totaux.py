"""Tests de la comparaison totaux (chantier #8 de docs/ROADMAP.md).

Données synthétiques : on contrôle exactement les agrégats (somme des prévisions
individuelles vs total observé) et les métriques attendues (MAE, WAPE %, biais
signé %), plus un test d'intégration de `main()` sur un dossier temporaire et
des bornes sur les vrais backtests commités (lecture seule).
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import comparaison_totaux
from src.comparaison_totaux import (
    COLONNES_RAPPORT,
    DOMAINE_MENSUEL,
    DOMAINE_PHARMA,
    STRATEGIE_SOMME,
    aggreger_total,
    comparaison_mensuel,
    comparaison_pharma,
    metriques_total,
    rapport,
)

RACINE = Path(__file__).resolve().parents[1]


def df_boites(lignes):
    """Backtest pharma synthétique : (origine, horizon, produit, observe, timestral)."""
    return pd.DataFrame(lignes, columns=["origine", "horizon", "produit", "observe", "timestral"])


def df_mensuel(lignes):
    """Backtest mensuel synthétique : (origine, horizon_mois, reg, lits_reel, pred_Timestral)."""
    return pd.DataFrame(lignes, columns=["origine", "horizon_mois", "reg", "lits_reel", "pred_Timestral"])


class AggregerTotalTest(unittest.TestCase):
    def test_somme_par_groupe(self):
        df = df_boites([
            ("2022-05-01", 1, "a", 100, 110),
            ("2022-05-01", 1, "b", 300, 290),
            ("2022-05-01", 2, "a", 50, 60),
            ("2022-05-01", 2, "b", 50, 40),
            ("2022-06-01", 1, "a", 200, 180),
            ("2022-06-01", 1, "b", 200, 220),
        ])
        agg = aggreger_total(df, ["origine", "horizon"], "timestral", "observe")
        self.assertEqual(list(agg.columns), ["origine", "horizon", "total_predit", "total_observe"])
        self.assertEqual(len(agg), 3)
        lignes = {(r.origine, r.horizon): (r.total_predit, r.total_observe)
                  for r in agg.itertuples(index=False)}
        self.assertEqual(lignes[("2022-05-01", 1)], (400, 400))
        self.assertEqual(lignes[("2022-05-01", 2)], (100, 100))
        self.assertEqual(lignes[("2022-06-01", 1)], (400, 400))

    def test_lignes_incompletes_ignorees(self):
        # un total partiel (prévision manquante pour une forme) n'est pas un total
        df = df_boites([
            ("2022-05-01", 1, "a", 100, 110),
            ("2022-05-01", 1, "b", 300, np.nan),
            ("2022-06-01", 1, "a", 200, 180),
            ("2022-06-01", 1, "b", 200, 220),
        ])
        agg = aggreger_total(df, ["origine", "horizon"], "timestral", "observe")
        lignes = {(r.origine, r.horizon): (r.total_predit, r.total_observe)
                  for r in agg.itertuples(index=False)}
        self.assertNotIn(("2022-05-01", 1), lignes)
        self.assertEqual(lignes[("2022-06-01", 1)], (400, 400))

    def test_observe_manquant_aussi_ignore(self):
        df = df_boites([
            ("2022-05-01", 1, "a", np.nan, 110),
            ("2022-05-01", 1, "b", 300, 290),
        ])
        agg = aggreger_total(df, ["origine", "horizon"], "timestral", "observe")
        self.assertEqual(len(agg), 0)

    def test_colonnes_absentes_leve_une_erreur(self):
        df = df_boites([("2022-05-01", 1, "a", 100, 110)])
        with self.assertRaises(ValueError):
            aggreger_total(df, ["origine", "horizon"], "timestral", "observe_absente")
        with self.assertRaises(ValueError):
            aggreger_total(df, ["origine", "horizon_absent"], "timestral", "observe")


class MetriquesTotalTest(unittest.TestCase):
    def test_predictions_parfaites(self):
        m = metriques_total([100, 200, 300], [100, 200, 300])
        self.assertEqual(m["n"], 3)
        self.assertEqual(m["mae"], 0.0)
        self.assertEqual(m["wape"], 0.0)
        self.assertEqual(m["biais"], 0.0)

    def test_valeurs_connues(self):
        # erreurs +10 et -10 sur des observés 100 et 100
        m = metriques_total([110, 90], [100, 100])
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["mae"], 10.0)
        self.assertAlmostEqual(m["wape"], 10.0)   # 100 * 20 / 200
        self.assertAlmostEqual(m["biais"], 0.0)   # 100 * 0 / 200

    def test_biais_signe(self):
        m = metriques_total([120, 100], [100, 100])
        self.assertAlmostEqual(m["biais"], 10.0)  # surestimation de 10 %
        self.assertAlmostEqual(m["wape"], 10.0)
        self.assertAlmostEqual(m["mae"], 10.0)
        m = metriques_total([80, 100], [100, 100])
        self.assertAlmostEqual(m["biais"], -10.0)  # sous-estimation
        self.assertAlmostEqual(m["wape"], 10.0)

    def test_ponderation_par_les_gros_volumes(self):
        # WAPE et biais sont pondérés par l'observé : un gros mois pèse davantage
        m = metriques_total([220, 90], [200, 100])
        self.assertAlmostEqual(m["mae"], 15.0)
        self.assertAlmostEqual(m["wape"], 100.0 * (20 + 10) / 300)
        self.assertAlmostEqual(m["biais"], 100.0 * (20 - 10) / 300)

    def test_somme_observee_nulle(self):
        m = metriques_total([5, 5], [0, 0])
        self.assertAlmostEqual(m["mae"], 5.0)
        self.assertTrue(np.isnan(m["wape"]))
        self.assertTrue(np.isnan(m["biais"]))

    def test_taille_invalide(self):
        with self.assertRaises(ValueError):
            metriques_total([100], [100, 200])
        with self.assertRaises(ValueError):
            metriques_total([], [])


class ComparaisonPharmaTest(unittest.TestCase):
    def setUp(self):
        # h=1 : totaux (440 vs 400) et (380 vs 400) -> erreurs +40, -20
        # h=2 : total (110 vs 100) -> erreur +10
        self.boites = df_boites([
            ("2022-05-01", 1, "a", 100, 110),
            ("2022-05-01", 1, "b", 300, 330),
            ("2022-06-01", 1, "a", 200, 180),
            ("2022-06-01", 1, "b", 200, 200),
            ("2022-05-01", 2, "a", 50, 50),
            ("2022-05-01", 2, "b", 50, 60),
        ])

    def test_metriques_par_horizon(self):
        res = comparaison_pharma(self.boites)
        self.assertEqual(len(res), 2)
        self.assertTrue((res.domaine == DOMAINE_PHARMA).all())
        self.assertTrue((res.stratégie == STRATEGIE_SOMME).all())
        self.assertEqual(list(res.horizon), [1, 2])
        h1 = res.loc[res.horizon == 1].iloc[0]
        self.assertEqual(h1.n, 2)
        self.assertAlmostEqual(h1.mae, 30.0)          # (40 + 20) / 2
        self.assertAlmostEqual(h1.wape, 100.0 * 60 / 800)   # 7.5 %
        self.assertAlmostEqual(h1.biais, 100.0 * 20 / 800)  # 2.5 %
        h2 = res.loc[res.horizon == 2].iloc[0]
        self.assertEqual(h2.n, 1)
        self.assertAlmostEqual(h2.mae, 10.0)
        self.assertAlmostEqual(h2.wape, 10.0)
        self.assertAlmostEqual(h2.biais, 10.0)

    def test_colonne_prevision_personnalisee(self):
        df = self.boites.rename(columns={"timestral": "ajuste"})
        res = comparaison_pharma(df, prevision="ajuste")
        self.assertEqual(len(res), 2)


class ComparaisonMensuelTest(unittest.TestCase):
    def test_somme_regions_vs_total_france(self):
        # h=1 : régions (12 vs 10) et (28 vs 30) -> total 40 vs 40, erreur 0
        # h=3 : régions (30 vs 20) et (20 vs 20) -> total 50 vs 40, erreur +10
        mensuel = df_mensuel([
            ("2022-02-01", 1, 1, 10, 12),
            ("2022-02-01", 1, 2, 30, 28),
            ("2022-02-01", 3, 1, 20, 30),
            ("2022-02-01", 3, 2, 20, 20),
        ])
        res = comparaison_mensuel(mensuel)
        self.assertEqual(len(res), 2)
        self.assertTrue((res.domaine == DOMAINE_MENSUEL).all())
        self.assertTrue((res.stratégie == STRATEGIE_SOMME).all())
        self.assertEqual(list(res.horizon), [1, 3])  # horizon_mois renommé en horizon
        h1 = res.loc[res.horizon == 1].iloc[0]
        self.assertEqual(h1.n, 1)
        self.assertAlmostEqual(h1.mae, 0.0)
        self.assertAlmostEqual(h1.wape, 0.0)
        self.assertAlmostEqual(h1.biais, 0.0)
        h3 = res.loc[res.horizon == 3].iloc[0]
        self.assertAlmostEqual(h3.mae, 10.0)
        self.assertAlmostEqual(h3.wape, 25.0)   # 100 * 10 / 40
        self.assertAlmostEqual(h3.biais, 25.0)


class RapportTest(unittest.TestCase):
    def test_structure_et_ordre(self):
        boites = df_boites([
            ("2022-05-01", 1, "a", 100, 110),
            ("2022-05-01", 1, "b", 300, 330),
            ("2022-06-01", 1, "a", 200, 180),
            ("2022-06-01", 1, "b", 200, 200),
            ("2022-05-01", 2, "a", 50, 50),
            ("2022-05-01", 2, "b", 50, 60),
        ])
        mensuel = df_mensuel([
            ("2022-02-01", 1, 1, 10, 12),
            ("2022-02-01", 1, 2, 30, 28),
            ("2022-02-01", 3, 1, 20, 30),
            ("2022-02-01", 3, 2, 20, 20),
        ])
        table = rapport(boites, mensuel)
        self.assertEqual(list(table.columns), COLONNES_RAPPORT)
        self.assertEqual(len(table), 4)  # 2 horizons pharma + 2 mensuel
        # pharma d'abord, horizons croissants dans chaque domaine
        self.assertEqual(list(table.domaine), [DOMAINE_PHARMA, DOMAINE_PHARMA,
                                               DOMAINE_MENSUEL, DOMAINE_MENSUEL])
        self.assertEqual(list(table.horizon), [1, 2, 1, 3])
        self.assertTrue((table.stratégie == STRATEGIE_SOMME).all())
        self.assertTrue(np.isfinite(table.mae).all())
        self.assertTrue(np.isfinite(table.wape).all())
        self.assertTrue(np.isfinite(table.biais).all())


class MainIntegrationTest(unittest.TestCase):
    def test_main_ecrit_le_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            racine = Path(tmp)
            sorties = racine / "outputs"
            sorties.mkdir(parents=True)
            df_boites([
                ("2022-05-01", 1, "a", 100, 110),
                ("2022-05-01", 1, "b", 300, 330),
                ("2022-06-01", 1, "a", 200, 180),
                ("2022-06-01", 1, "b", 200, 200),
                ("2022-05-01", 2, "a", 50, 50),
                ("2022-05-01", 2, "b", 50, 60),
            ]).to_csv(sorties / "backtest_boites_pharma.csv", index=False)
            df_mensuel([
                ("2022-02-01", 1, 1, 10, 12),
                ("2022-02-01", 1, 2, 30, 28),
                ("2022-02-01", 3, 1, 20, 30),
                ("2022-02-01", 3, 2, 20, 20),
            ]).to_csv(sorties / "backtest_mensuel.csv", index=False)

            table = comparaison_totaux.main(racine=racine)

            destination = sorties / "comparaison_totaux.csv"
            self.assertTrue(destination.exists())
            lu = pd.read_csv(destination)
            self.assertEqual(list(lu.columns), COLONNES_RAPPORT)
            self.assertEqual(len(lu), 4)
            self.assertEqual(set(lu.domaine), {DOMAINE_PHARMA, DOMAINE_MENSUEL})
            # la table retournée est celle écrite
            pd.testing.assert_frame_equal(table, lu)
            # valeurs attendues (cf. ComparaisonPharmaTest / ComparaisonMensuelTest)
            h1 = lu.loc[(lu.domaine == DOMAINE_PHARMA) & (lu.horizon == 1)].iloc[0]
            self.assertAlmostEqual(h1.mae, 30.0)
            self.assertAlmostEqual(h1.wape, 7.5)
            self.assertAlmostEqual(h1.biais, 2.5)
            h3 = lu.loc[(lu.domaine == DOMAINE_MENSUEL) & (lu.horizon == 3)].iloc[0]
            self.assertAlmostEqual(h3.mae, 10.0)
            self.assertAlmostEqual(h3.wape, 25.0)
            self.assertAlmostEqual(h3.biais, 25.0)


@unittest.skipUnless((RACINE / "outputs" / "backtest_boites_pharma.csv").exists()
                     and (RACINE / "outputs" / "backtest_mensuel.csv").exists(),
                     "backtests absents")
class BacktestsReelsTest(unittest.TestCase):
    """Bornes de salubrité sur les vrais backtests commités (lecture seule)."""

    def test_rapport_reel_structure(self):
        boites = pd.read_csv(RACINE / "outputs" / "backtest_boites_pharma.csv")
        mensuel = pd.read_csv(RACINE / "outputs" / "backtest_mensuel.csv")
        table = rapport(boites, mensuel)
        self.assertEqual(list(table.columns), COLONNES_RAPPORT)
        self.assertEqual(set(table.domaine), {DOMAINE_PHARMA, DOMAINE_MENSUEL})
        # 3 horizons pharma (1-3 mois) + 4 horizons mensuels (1, 3, 6, 12)
        self.assertEqual(len(table), 7)
        self.assertTrue((table.wape >= 0).all())
        self.assertTrue(np.isfinite(table.mae).all())
        self.assertTrue(np.isfinite(table.wape).all())
        self.assertTrue(np.isfinite(table.biais).all())

    def test_pharma_horizon_1_conforme_a_la_reference_publiee(self):
        # docs/PASSATION.md : 51 mois à 1 mois, total des six formes,
        # MAE = 114 336 boîtes et WAPE = 13,29 %
        boites = pd.read_csv(RACINE / "outputs" / "backtest_boites_pharma.csv")
        mensuel = pd.read_csv(RACINE / "outputs" / "backtest_mensuel.csv")
        table = rapport(boites, mensuel)
        h1 = table.loc[(table.domaine == DOMAINE_PHARMA) & (table.horizon == 1)].iloc[0]
        self.assertAlmostEqual(h1.mae, 114336.0, delta=500.0)
        self.assertAlmostEqual(h1.wape, 13.29, delta=0.05)


if __name__ == "__main__":
    unittest.main()
