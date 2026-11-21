"""Tests de l'incertitude sur la période de vérification réservée (chantier #6)."""

import unittest

import numpy as np
import pandas as pd

from src.recherche_delai_pharma import POIDS, poids_du_melange
from src.recherche_volumes_pharma import DEBUT_VERIFICATION, FIN_DEV, scores
from src.stats_verif import block_bootstrap_gain, diebold_mariano, scores_avec_ic


class BlockBootstrapGainTest(unittest.TestCase):
    def test_reproductible_au_seed_pres(self):
        rng = np.random.default_rng(42)
        a = np.abs(rng.normal(100.0, 20.0, 24))
        b = a + np.abs(rng.normal(10.0, 5.0, 24))
        r1 = block_bootstrap_gain(a, b, seed=7)
        r2 = block_bootstrap_gain(a, b, seed=7)
        self.assertEqual(r1, r2)
        r3 = block_bootstrap_gain(a, b, seed=8)
        self.assertNotEqual((r1["ic95_bas"], r1["ic95_haut"]), (r3["ic95_bas"], r3["ic95_haut"]))

    def test_gain_positif_et_ic_excluant_zero_quand_a_est_nettement_meilleur(self):
        a = np.full(24, 100.0)
        b = np.full(24, 130.0)
        r = block_bootstrap_gain(a, b, seed=0)
        self.assertAlmostEqual(r["gain"], 30.0)
        self.assertGreater(r["ic95_bas"], 0.0)
        self.assertLessEqual(r["ic95_bas"], r["gain"])
        self.assertLessEqual(r["gain"], r["ic95_haut"])
        self.assertEqual(r["proba_gain_positif"], 1.0)
        self.assertEqual(r["n_bootstrap"], 2000)
        self.assertEqual(r["taille_bloc"], 3)

    def test_denominateur_donne_un_gain_de_wape(self):
        # gain = (sum|e_b| - sum|e_a|) / sum(y) = (1200 - 600) / 12000 = 0,05
        y = np.full(12, 1000.0)
        r = block_bootstrap_gain(np.full(12, 50.0), np.full(12, 100.0), denominateur=y, seed=1)
        self.assertAlmostEqual(r["gain"], 0.05)

    def test_taille_bloc_tronquee_sur_petite_serie(self):
        r = block_bootstrap_gain([1.0, 2.0], [3.0, 4.0], taille_bloc=10, n_bootstrap=50, seed=1)
        self.assertEqual(r["taille_bloc"], 2)
        self.assertAlmostEqual(r["gain"], 2.0)
        self.assertEqual(r["ic95_bas"], r["ic95_haut"])

    def test_entrees_invalides(self):
        with self.assertRaises(ValueError):
            block_bootstrap_gain([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            block_bootstrap_gain([], [])
        with self.assertRaises(ValueError):
            block_bootstrap_gain([1.0, np.nan], [1.0, 2.0])
        with self.assertRaises(ValueError):
            block_bootstrap_gain([1.0, 2.0], [1.0, 2.0], taille_bloc=0)
        with self.assertRaises(ValueError):
            block_bootstrap_gain([1.0, 2.0], [1.0, 2.0], n_bootstrap=0)
        with self.assertRaises(ValueError):
            block_bootstrap_gain([1.0, 2.0], [1.0, 2.0], denominateur=[1.0, -1.0])


class DieboldMarianoTest(unittest.TestCase):
    def test_significatif_sur_series_tres_differentes(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 10.0, 60)
        b = rng.normal(0.0, 10.0, 60) + 50.0
        r = diebold_mariano(a, b, horizon=1)
        self.assertGreater(r["dm_stat"], 0.0)
        self.assertLess(r["p_value"], 0.05)
        self.assertTrue(r["significatif_5pct"])
        self.assertEqual(r["n"], 60)
        self.assertEqual(r["horizon"], 1)

    def test_non_significatif_sur_series_identiques(self):
        e = np.array([1.0, -2.0, 3.0, -4.0] * 10)
        r = diebold_mariano(e, e.copy(), horizon=1)
        self.assertEqual(r["dm_stat"], 0.0)
        self.assertEqual(r["p_value"], 1.0)
        self.assertFalse(r["significatif_5pct"])
        self.assertEqual(r["gain_moyen"], 0.0)

    def test_non_significatif_sur_series_proches(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0.0, 10.0, 30)
        b = a + rng.normal(0.0, 0.5, 30)
        r = diebold_mariano(a, b)
        self.assertGreater(r["p_value"], 0.05)
        self.assertFalse(r["significatif_5pct"])

    def test_horizon_superieur_tronque_la_variance_hac(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0.0, 5.0, 40)
        b = rng.normal(0.0, 5.0, 40) + 8.0
        r = diebold_mariano(a, b, horizon=3)
        self.assertEqual(r["horizon"], 3)
        self.assertLess(r["p_value"], 0.05)
        with self.assertRaises(ValueError):
            diebold_mariano(a, b, horizon=0)


class ScoresAvecIcTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        n = 24
        observe = 1_000_000.0 + rng.normal(0.0, 50_000.0, n).round()
        self.df = pd.DataFrame({
            "observe": observe,
            "bon": observe + rng.normal(0.0, 20_000.0, n).round(),
            "moyen": observe + rng.normal(0.0, 60_000.0, n).round(),
            "mauvais": observe + 150_000.0,
        })
        self.modeles = ["bon", "moyen", "mauvais"]

    def test_applique_scores_puis_ajoute_les_ic(self):
        base = scores(self.df, self.modeles)
        r = scores_avec_ic(self.df, self.modeles, bloc=3)
        self.assertEqual(r.modele.tolist(), base.modele.tolist())
        pd.testing.assert_frame_equal(r[base.columns], base)
        for colonne in ["est_meilleur", "gain_mae_boites", "ic95_mae_bas", "ic95_mae_haut",
                        "gain_wape_pct", "ic95_wape_bas", "ic95_wape_haut",
                        "proba_gain_positif", "dm_stat", "dm_p_value"]:
            self.assertIn(colonne, r.columns)
        self.assertEqual(r.modele.iloc[0], "bon")
        self.assertTrue(r.est_meilleur.iloc[0])

    def test_ligne_du_meilleur_a_des_gains_nuls(self):
        r = scores_avec_ic(self.df, self.modeles)
        ligne = r.loc[r.est_meilleur].iloc[0]
        self.assertEqual(ligne.gain_mae_boites, 0.0)
        self.assertEqual((ligne.ic95_mae_bas, ligne.ic95_mae_haut), (0.0, 0.0))
        self.assertEqual(ligne.gain_wape_pct, 0.0)
        self.assertEqual((ligne.ic95_wape_bas, ligne.ic95_wape_haut), (0.0, 0.0))
        self.assertEqual(ligne.dm_p_value, 1.0)

    def test_ic_du_gain_du_meilleur_vers_chaque_reference(self):
        r = scores_avec_ic(self.df, self.modeles, bloc=3, seed=0)
        for _, ligne in r.loc[~r.est_meilleur].iterrows():
            self.assertGreater(ligne.gain_mae_boites, 0.0)
            self.assertLessEqual(ligne.ic95_mae_bas, ligne.gain_mae_boites)
            self.assertLessEqual(ligne.gain_mae_boites, ligne.ic95_mae_haut)
            self.assertGreater(ligne.gain_wape_pct, 0.0)
            self.assertLessEqual(ligne.ic95_wape_bas, ligne.gain_wape_pct)
            self.assertLessEqual(ligne.gain_wape_pct, ligne.ic95_wape_haut)
        mauvais = r.loc[r.modele.eq("mauvais")].iloc[0]
        self.assertGreater(mauvais.ic95_mae_bas, 0.0)  # gain robuste
        self.assertLess(mauvais.dm_p_value, 0.05)
        self.assertEqual(mauvais.proba_gain_positif, 1.0)

    def test_gain_wape_coherent_avec_les_wape(self):
        r = scores_avec_ic(self.df, self.modeles)
        mauvais = r.loc[r.modele.eq("mauvais")].iloc[0]
        attendu = mauvais.wape_pct - r.loc[r.est_meilleur].iloc[0].wape_pct
        self.assertAlmostEqual(mauvais.gain_wape_pct, attendu, places=6)
    def test_meilleur_impose_et_cible_renommee(self):
        df = self.df.rename(columns={"observe": "volume"})
        r = scores_avec_ic(df, self.modeles, cible="volume", meilleur="moyen")
        self.assertEqual(r.loc[r.est_meilleur, "modele"].iloc[0], "moyen")
        bon = r.loc[r.modele.eq("bon")].iloc[0]
        self.assertLess(bon.gain_mae_boites, 0.0)  # "moyen" est moins bon que "bon"
        with self.assertRaises(ValueError):
            scores_avec_ic(df, self.modeles, cible="volume", meilleur="inconnu")
        with self.assertRaises(ValueError):
            scores_avec_ic(df, ["bon", "absent"], cible="volume")
        with self.assertRaises(ValueError):
            scores_avec_ic(df, [], cible="volume")

    def test_reproductible(self):
        r1 = scores_avec_ic(self.df, self.modeles, seed=4)
        r2 = scores_avec_ic(self.df, self.modeles, seed=4)
        pd.testing.assert_frame_equal(r1, r2)


class PoidsDuMelangeTest(unittest.TestCase):
    def test_le_choix_de_poids_ignore_la_verification(self):
        # Sur le développement seul, poids = 1.0 est optimal ; si la ligne de
        # vérification (>= DEBUT_VERIFICATION) comptait, le choix basculerait.
        donnees = pd.DataFrame({
            "observe": [100.0, 0.0],
            "timestral_60_mois": [100.0, 1000.0],
            "saisonnier_ajuste": [0.0, 0.0],
        }, index=pd.PeriodIndex([FIN_DEV, DEBUT_VERIFICATION], freq="M"))
        predictions = (donnees["timestral_60_mois"].to_numpy()[:, None] * POIDS
                       + donnees["saisonnier_ajuste"].to_numpy()[:, None] * (1 - POIDS))
        erreurs_toutes = np.abs(predictions - donnees.observe.to_numpy()[:, None]).mean(axis=0)
        self.assertEqual(float(POIDS[erreurs_toutes.argmin()]), 0.0)  # bascule si la vérification compte
        self.assertEqual(poids_du_melange(donnees), 1.0)  # elle est exclue du réglage

    def test_aucune_donnee_de_developpement_refuse(self):
        donnees = pd.DataFrame({
            "observe": [1.0],
            "timestral_60_mois": [1.0],
            "saisonnier_ajuste": [1.0],
        }, index=pd.PeriodIndex([DEBUT_VERIFICATION], freq="M"))
        with self.assertRaises(ValueError):
            poids_du_melange(donnees)


if __name__ == "__main__":
    unittest.main()
