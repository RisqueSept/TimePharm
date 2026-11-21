"""Tests de la calibration mesurée (chantier #4 de docs/ROADMAP.md).

Données synthétiques : on contrôle exactement les métriques attendues
(couverture = 1 quand tout est dans la bande, Brier = 0 pour des prédictions
parfaites, PR-AUC connu, biais signé, etc.), plus un test d'intégration de
`main()` sur un dossier temporaire et des bornes sur les vrais backtests.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import calibration
from src.calibration import (
    biais_par_forme,
    couverture_quantiles,
    metriques_binaires,
    metriques_degradations,
    table_tensions,
    table_volumes,
)
from src.seuils import COUVERTURE_VISEE, SEUIL_ALERTE_RISQUE

RACINE = Path(__file__).resolve().parents[1]


def df_boites(lignes):
    """DataFrame synthétique de backtest des volumes (une ligne = un cas)."""
    return pd.DataFrame(lignes, columns=[
        "produit", "horizon", "observe", "timestral_q10", "timestral", "timestral_q90",
    ])


def df_tensions(lignes):
    """DataFrame synthétique de backtest des tensions."""
    return pd.DataFrame(lignes, columns=[
        "produit", "horizon", "observé", "timestral", "degradation",
    ])


def ligne_resume(res, niveau, produit="tous", horizon="tous"):
    masque = (
        (res["resume"].niveau == niveau)
        & (res["resume"].produit == produit)
        & (res["resume"].horizon == horizon)
    )
    return res["resume"].loc[masque].iloc[0]


class CouvertureQuantilesTest(unittest.TestCase):
    def test_couverture_1_quand_tout_dans_la_bande(self):
        df = df_boites([
            ("a", 1, 15, 10, 12, 20),
            ("a", 2, 25, 20, 22, 30),
            ("b", 1, 5, 0, 3, 10),
            ("b", 2, 50, 40, 45, 60),
        ])
        res = couverture_quantiles(df)
        g = res.loc[res.niveau == "global"].iloc[0]
        self.assertEqual(g.n, 4)
        self.assertEqual(g.couverture, 1.0)

    def test_couverture_0_quand_tout_hors_bande(self):
        df = df_boites([
            ("a", 1, 5, 10, 12, 20),   # sous q10
            ("b", 1, 25, 10, 12, 20),  # au-dessus de q90
        ])
        res = couverture_quantiles(df)
        self.assertEqual(res.loc[res.niveau == "global"].iloc[0].couverture, 0.0)

    def test_couverture_partielle_par_produit_et_horizon(self):
        df = df_boites([
            ("a", 1, 15, 10, 12, 20),  # dans la bande
            ("a", 1, 99, 10, 12, 20),  # hors bande
            ("b", 1, 15, 10, 12, 20),  # dans la bande
            ("a", 2, 15, 10, 12, 20),  # dans la bande
        ])
        res = couverture_quantiles(df)
        g = res.loc[res.niveau == "global"].iloc[0]
        self.assertAlmostEqual(g.couverture, 0.75)
        pa = res.loc[(res.niveau == "produit") & (res.produit == "a")].iloc[0]
        self.assertAlmostEqual(pa.couverture, 2 / 3)
        pb = res.loc[(res.niveau == "produit") & (res.produit == "b")].iloc[0]
        self.assertAlmostEqual(pb.couverture, 1.0)
        h1 = res.loc[(res.niveau == "horizon") & (res.horizon == 1)].iloc[0]
        self.assertAlmostEqual(h1.couverture, 2 / 3)
        h2 = res.loc[(res.niveau == "horizon") & (res.horizon == 2)].iloc[0]
        self.assertAlmostEqual(h2.couverture, 1.0)
        croise = res.loc[(res.niveau == "horizon_produit") & (res.produit == "a") & (res.horizon == 1)]
        self.assertEqual(len(croise), 1)
        self.assertAlmostEqual(croise.iloc[0].couverture, 0.5)

    def test_bornes_bande_inclusives(self):
        df = df_boites([("a", 1, 10, 10, 11, 20), ("a", 1, 20, 10, 11, 20)])
        res = couverture_quantiles(df)
        self.assertEqual(res.loc[res.niveau == "global"].iloc[0].couverture, 1.0)

    def test_lignes_incompletes_ignorees(self):
        df = df_boites([("a", 1, 15, 10, 12, 20), ("b", 1, np.nan, 0, 3, 10)])
        res = couverture_quantiles(df)
        g = res.loc[res.niveau == "global"].iloc[0]
        self.assertEqual(g.n, 1)
        self.assertEqual(g.couverture, 1.0)

    def test_niveaux_attendus(self):
        df = df_boites([("a", 1, 15, 10, 12, 20), ("b", 2, 25, 20, 22, 30)])
        res = couverture_quantiles(df)
        self.assertEqual(set(res.niveau), {"global", "produit", "horizon", "horizon_produit"})
        self.assertEqual(len(res), 1 + 2 + 2 + 2)


class MetriquesBinairesTest(unittest.TestCase):
    def test_brier_0_et_pr_auc_1_pour_predictions_parfaites(self):
        df = df_tensions([
            ("a", 14, 1, 1.0, np.nan),
            ("a", 14, 0, 0.0, np.nan),
            ("b", 28, 1, 1.0, np.nan),
            ("b", 28, 0, 0.0, np.nan),
        ])
        res = metriques_binaires(df)
        g = ligne_resume(res, "global")
        self.assertAlmostEqual(g.brier, 0.0)
        self.assertAlmostEqual(g.pr_auc, 1.0)
        self.assertEqual(g.n_detections, 2)
        self.assertEqual(g.n_fausses_alertes, 0)
        self.assertEqual(g.taux_detection, 1.0)
        self.assertEqual(g.taux_fausses_alertes, 0.0)

    def test_brier_valeur_connue(self):
        df = df_tensions([
            ("a", 14, 1, 0.5, np.nan),
            ("a", 14, 0, 0.5, np.nan),
            ("a", 14, 1, 0.5, np.nan),
            ("a", 14, 0, 0.5, np.nan),
        ])
        g = ligne_resume(metriques_binaires(df), "global")
        self.assertAlmostEqual(g.brier, 0.25)

    def test_pr_auc_valeur_connue(self):
        # scores décroissants, étiquettes [1, 0, 1, 0] -> AP = 1/2 + (1/2)*(2/3) = 5/6
        df = df_tensions([
            ("a", 14, 1, 0.9, np.nan),
            ("a", 14, 0, 0.8, np.nan),
            ("a", 14, 1, 0.3, np.nan),
            ("a", 14, 0, 0.2, np.nan),
        ])
        g = ligne_resume(metriques_binaires(df), "global")
        self.assertAlmostEqual(g.pr_auc, 5 / 6)

    def test_pr_auc_egal_a_sklearn_average_precision(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=200).astype(float)
        # scores arrondis pour forcer des égalités
        s = np.round(rng.random(200), decimals=1)
        df = df_tensions([("a", 14, int(v), float(p), np.nan) for v, p in zip(y, s)])
        g = ligne_resume(metriques_binaires(df), "global")
        self.assertAlmostEqual(g.pr_auc, average_precision_score(y, s), places=12)

    def test_pr_auc_nan_sans_positif(self):
        df = df_tensions([("a", 14, 0, 0.9, np.nan), ("a", 14, 0, 0.1, np.nan)])
        g = ligne_resume(metriques_binaires(df), "global")
        self.assertTrue(np.isnan(g.pr_auc))
        self.assertTrue(np.isnan(g.taux_detection))

    def test_detection_et_fausses_alertes_au_seuil_a_priori(self):
        df = df_tensions([
            ("a", 14, 1, 0.9, np.nan),  # alerte, détection
            ("a", 14, 0, 0.7, np.nan),  # alerte, fausse alerte
            ("a", 14, 1, 0.4, np.nan),  # pas d'alerte, manquée
            ("a", 14, 0, 0.1, np.nan),  # pas d'alerte
        ])
        res = metriques_binaires(df)
        g = ligne_resume(res, "global")
        self.assertEqual(g.seuil, SEUIL_ALERTE_RISQUE)
        self.assertEqual(g.n_alertes, 2)
        self.assertEqual(g.n_detections, 1)
        self.assertEqual(g.n_fausses_alertes, 1)
        self.assertAlmostEqual(g.taux_detection, 0.5)
        self.assertAlmostEqual(g.taux_fausses_alertes, 0.5)

    def test_score_egal_au_seuil_declenche_l_alerte(self):
        # convention de src/pharma.js : alerte si score >= seuil
        df = df_tensions([("a", 14, 1, SEUIL_ALERTE_RISQUE, np.nan)])
        g = ligne_resume(metriques_binaires(df), "global")
        self.assertEqual(g.n_alertes, 1)
        self.assertEqual(g.n_detections, 1)

    def test_seuil_personnalise(self):
        df = df_tensions([
            ("a", 14, 1, 0.9, np.nan),
            ("a", 14, 0, 0.7, np.nan),
        ])
        g = ligne_resume(metriques_binaires(df, seuil=0.8), "global")
        self.assertEqual(g.n_alertes, 1)
        self.assertEqual(g.n_fausses_alertes, 0)

    def test_courbe_fiabilite_10_bins_par_perimetre(self):
        df = df_tensions([
            ("a", 14, 1, 0.95, np.nan),
            ("a", 14, 0, 0.05, np.nan),
            ("b", 28, 1, 0.55, np.nan),
            ("b", 28, 0, 0.45, np.nan),
        ])
        res = metriques_binaires(df)
        fiab = res["fiabilite"]
        self.assertEqual(set(zip(fiab.niveau, fiab.produit, fiab.horizon)),
                         set(zip(res["resume"].niveau, res["resume"].produit, res["resume"].horizon)))
        for r in res["resume"].itertuples(index=False):
            masque = (
                (fiab.niveau == r.niveau) & (fiab.produit == r.produit) & (fiab.horizon == r.horizon)
            )
            bins = fiab.loc[masque]
            self.assertEqual(len(bins), 10)
            self.assertEqual(list(bins.bin), list(range(10)))
            self.assertAlmostEqual(bins.bin_min.iloc[0], 0.0)
            self.assertAlmostEqual(bins.bin_max.iloc[-1], 1.0)
            self.assertEqual(bins.n.sum(), r.n)
        # prédiction parfaite : le bin 9 ne contient que des positifs
        haut = fiab.loc[(fiab.niveau == "global") & (fiab.bin == 9)].iloc[0]
        self.assertEqual(haut.n, 1)
        self.assertAlmostEqual(haut.proba_moyenne, 0.95)
        self.assertAlmostEqual(haut.frequence_observee, 1.0)
        bas = fiab.loc[(fiab.niveau == "global") & (fiab.bin == 0)].iloc[0]
        self.assertAlmostEqual(bas.frequence_observee, 0.0)

    def test_metriques_par_horizon_et_par_produit(self):
        df = df_tensions([
            ("a", 14, 1, 0.9, np.nan),
            ("a", 14, 0, 0.2, np.nan),
            ("a", 28, 0, 0.6, np.nan),
            ("b", 14, 1, 0.8, np.nan),
            ("b", 28, 0, 0.3, np.nan),
        ])
        res = metriques_binaires(df)
        self.assertEqual(len(res["resume"]), 1 + 2 + 2 + 4)
        h14 = ligne_resume(res, "horizon", horizon=14)
        self.assertEqual(h14.n, 3)
        self.assertEqual(h14.n_positifs, 2)
        pa = ligne_resume(res, "produit", produit="a")
        self.assertEqual(pa.n, 3)
        self.assertAlmostEqual(pa.brier, np.mean([(0.9 - 1) ** 2, (0.2 - 0) ** 2, (0.6 - 0) ** 2]))
        croise = ligne_resume(res, "horizon_produit", produit="b", horizon=28)
        self.assertEqual(croise.n, 1)

    def test_lignes_sans_score_ignorees(self):
        df = df_tensions([
            ("a", 14, 1, 0.9, np.nan),
            ("a", 14, np.nan, 0.1, np.nan),
        ])
        g = ligne_resume(metriques_binaires(df), "global")
        self.assertEqual(g.n, 1)

    def test_colonne_score_absente_leve_une_erreur(self):
        df = pd.DataFrame({"produit": ["a"], "horizon": [14], "observé": [1]})
        with self.assertRaises(KeyError):
            metriques_binaires(df, score="timestral")


class MetriquesDegradationsTest(unittest.TestCase):
    def test_se_restreint_aux_lignes_degradation_non_nulles(self):
        df = df_tensions([
            ("a", 14, 1, 0.9, 1.0),   # nouvelle dégradation, bien notée
            ("a", 14, 0, 0.1, 0.0),   # pas de dégradation, bien notée
            ("b", 28, 0, 0.4, np.nan), # hors volet (tension déjà connue)
        ])
        res = metriques_degradations(df)
        g = ligne_resume(res, "global")
        self.assertEqual(g.n, 2)  # la ligne sans score degradation est exclue
        self.assertAlmostEqual(g.brier, 0.0)
        self.assertAlmostEqual(g.pr_auc, 1.0)
        self.assertEqual(g.n_detections, 1)
        self.assertEqual(g.n_fausses_alertes, 0)

    def test_colonne_degradation_absente_leve_une_erreur(self):
        df = df_tensions([("a", 14, 1, 0.9, np.nan)]).drop(columns="degradation")
        with self.assertRaises(KeyError):
            metriques_degradations(df)


class BiaisParFormeTest(unittest.TestCase):
    def test_biais_signe_positif_de_10_pourcent(self):
        df = df_boites([
            ("a", 1, 100, 10, 110, 200),
            ("a", 1, 300, 10, 330, 400),
            ("b", 2, 50, 10, 55, 100),
        ])
        res = biais_par_forme(df)
        self.assertEqual(len(res), 2)
        for r in res.itertuples(index=False):
            self.assertAlmostEqual(r.biais_pct, 10.0)
            self.assertAlmostEqual(r.erreur_signee, 0.10 * r.somme_observe)
        a1 = res.loc[(res.produit == "a") & (res.horizon == 1)].iloc[0]
        self.assertEqual(a1.n, 2)
        self.assertAlmostEqual(a1.somme_observe, 400.0)
        self.assertAlmostEqual(a1.somme_prevision, 440.0)

    def test_biais_signe_negatif(self):
        df = df_boites([("a", 1, 200, 10, 180, 300)])
        r = biais_par_forme(df).iloc[0]
        self.assertAlmostEqual(r.biais_pct, -10.0)

    def test_biais_nul_quand_prevision_egale_observe(self):
        df = df_boites([("a", 1, 100, 10, 100, 200), ("b", 3, 42, 10, 42, 100)])
        res = biais_par_forme(df)
        self.assertTrue((res.biais_pct == 0.0).all())

    def test_biais_nan_quand_somme_observee_nulle(self):
        df = df_boites([("a", 1, 0, 0, 5, 10)])
        r = biais_par_forme(df).iloc[0]
        self.assertTrue(np.isnan(r.biais_pct))

    def test_par_produit_et_horizon(self):
        df = df_boites([
            ("a", 1, 100, 10, 120, 200),
            ("a", 2, 100, 10, 100, 200),
            ("b", 1, 100, 10, 80, 200),
        ])
        res = biais_par_forme(df)
        self.assertEqual(set(zip(res.produit, res.horizon)), {("a", 1), ("a", 2), ("b", 1)})
        valeurs = {(r.produit, r.horizon): r.biais_pct for r in res.itertuples(index=False)}
        self.assertAlmostEqual(valeurs[("a", 1)], 20.0)
        self.assertAlmostEqual(valeurs[("a", 2)], 0.0)
        self.assertAlmostEqual(valeurs[("b", 1)], -20.0)


class TablesCsvTest(unittest.TestCase):
    def test_table_volumes_structure(self):
        boites = df_boites([
            ("a", 1, 15, 10, 12, 20),
            ("b", 2, 25, 20, 27, 30),
        ])
        table = table_volumes(boites)
        self.assertEqual(list(table.columns), calibration.COLONNES_VOLUMES)
        self.assertIn("couverture", set(table.metrique))
        self.assertIn("biais_pct", set(table.metrique))
        visee = table.loc[table.metrique == "couverture_visee"].iloc[0]
        self.assertAlmostEqual(visee.valeur, COUVERTURE_VISEE)

    def test_table_tensions_structure(self):
        df = df_tensions([
            ("a", 14, 1, 0.9, 0.8),
            ("a", 14, 0, 0.2, 0.1),
            ("b", 28, 0, 0.4, np.nan),
        ])
        res = metriques_binaires(df)
        table = table_tensions(res, sous_ensemble="tous")
        self.assertEqual(list(table.columns), calibration.COLONNES_TENSIONS)
        self.assertIn("brier", set(table.metrique))
        self.assertIn("pr_auc", set(table.metrique))
        self.assertIn("fiabilite", set(table.metrique))
        fiab = table.loc[table.metrique == "fiabilite"]
        self.assertEqual(len(fiab), 10 * len(res["resume"]))  # 10 bins par périmètre
        self.assertTrue(fiab.bin.notna().all())
        resume = table.loc[table.metrique == "brier"]
        g = resume.loc[resume.niveau == "global"].iloc[0]
        self.assertAlmostEqual(g.valeur, np.mean([(0.9 - 1) ** 2, (0.2 - 0) ** 2, (0.4 - 0) ** 2]))


class MainIntegrationTest(unittest.TestCase):
    def test_main_ecrit_les_deux_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            racine = Path(tmp)
            sorties = racine / "outputs"
            sorties.mkdir(parents=True)
            df_boites([
                ("a", 1, 10, 5, 11, 15),
                ("a", 2, 20, 15, 22, 25),
                ("b", 1, 30, 25, 30, 35),
            ]).to_csv(sorties / "backtest_boites_pharma.csv", index=False)
            df_tensions([
                ("a", 14, 1, 0.9, 0.7),
                ("a", 28, 0, 0.2, np.nan),
                ("b", 14, 0, 0.4, 0.1),
                ("b", 28, 1, 0.8, 0.6),
            ]).to_csv(sorties / "backtest_pharma.csv", index=False)

            volumes, tensions = calibration.main(racine=racine)

            f_volumes = sorties / "calibration_volumes.csv"
            f_tensions = sorties / "calibration_tensions.csv"
            self.assertTrue(f_volumes.exists())
            self.assertTrue(f_tensions.exists())
            lu_volumes = pd.read_csv(f_volumes)
            lu_tensions = pd.read_csv(f_tensions)
            self.assertIn("couverture", set(lu_volumes.metrique))
            self.assertIn("biais_pct", set(lu_volumes.metrique))
            self.assertIn("brier", set(lu_tensions.metrique))
            self.assertIn("fiabilite", set(lu_tensions.metrique))
            self.assertEqual(set(lu_tensions.sous_ensemble), {"tous", "degradations"})
            # le volet degradations ne garde que les lignes avec un score degradation
            degr = lu_tensions.loc[lu_tensions.sous_ensemble == "degradations"]
            brier_degr = degr.loc[(degr.metrique == "brier") & (degr.niveau == "global")].iloc[0]
            self.assertEqual(brier_degr.n, 3)
            self.assertAlmostEqual(brier_degr.valeur, np.mean([(0.7 - 1) ** 2, (0.1 - 0) ** 2, (0.6 - 1) ** 2]))
            # les tables retournées sont celles écrites (mêmes métriques)
            self.assertEqual(list(volumes.metrique), list(lu_volumes.metrique))
            self.assertEqual(len(tensions), len(lu_tensions))
            brier_tous = lu_tensions.loc[
                (lu_tensions.sous_ensemble == "tous")
                & (lu_tensions.metrique == "brier")
                & (lu_tensions.niveau == "global")
            ].iloc[0]
            attendu = np.mean([(0.9 - 1) ** 2, (0.2 - 0) ** 2, (0.4 - 0) ** 2, (0.8 - 1) ** 2])
            self.assertAlmostEqual(brier_tous.valeur, attendu)
            self.assertAlmostEqual(
                volumes.loc[volumes.metrique == "couverture_visee", "valeur"].iloc[0],
                COUVERTURE_VISEE,
            )


@unittest.skipUnless((RACINE / "outputs" / "backtest_boites_pharma.csv").exists(),
                     "backtests pharma absents")
class BacktestsReelsTest(unittest.TestCase):
    """Bornes de salubrité sur les vrais backtests commités (lecture seule)."""

    def test_couverture_backtest_boites_reel(self):
        df = pd.read_csv(RACINE / "outputs" / "backtest_boites_pharma.csv")
        res = couverture_quantiles(df)
        g = res.loc[res.niveau == "global"].iloc[0]
        self.assertEqual(g.n, len(df))
        self.assertGreaterEqual(g.couverture, 0.0)
        self.assertLessEqual(g.couverture, 1.0)
        # un périmètre par produit et par horizon
        self.assertEqual(len(res.loc[res.niveau == "produit"]), df.produit.nunique())
        self.assertEqual(len(res.loc[res.niveau == "horizon"]), df.horizon.nunique())

    def test_metriques_backtest_tensions_reel(self):
        df = pd.read_csv(RACINE / "outputs" / "backtest_pharma.csv")
        res = metriques_binaires(df)
        g = ligne_resume(res, "global")
        self.assertEqual(g.n, len(df.dropna(subset=["timestral", "observé"])))
        self.assertEqual(g.seuil, SEUIL_ALERTE_RISQUE)
        self.assertGreaterEqual(g.brier, 0.0)
        self.assertGreaterEqual(g.pr_auc, 0.0)
        self.assertLessEqual(g.pr_auc, 1.0)
        self.assertGreaterEqual(g.taux_detection, 0.0)
        self.assertLessEqual(g.taux_detection, 1.0)
        # 10 bins de fiabilité par périmètre du resume
        self.assertEqual(len(res["fiabilite"]), 10 * len(res["resume"]))
        degr = metriques_degradations(df)
        self.assertEqual(ligne_resume(degr, "global").n, int(df.degradation.notna().sum()))

    def test_biais_backtest_boites_reel(self):
        df = pd.read_csv(RACINE / "outputs" / "backtest_boites_pharma.csv")
        res = biais_par_forme(df)
        self.assertEqual(len(res), df.groupby(["produit", "horizon"]).ngroups)
        self.assertTrue(np.isfinite(res.biais_pct).all())


if __name__ == "__main__":
    unittest.main()
