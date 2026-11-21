"""Tests de l'indicateur de besoin relatif par région (chantier #12)."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

import besoin_regional as br

PATHOLOGIES = ["prev_diabete", "prev_cancer", "prev_cardio", "prev_psy", "prev_resp", "prev_neuro"]


def _previsions_synth():
    rows = []
    for horizon, mois in [(1, "2026-12-01"), (2, "2027-01-01")]:
        for produit, val in [("amox250", 100.0), ("amox500", 200.0), ("azithro", 400.0)]:
            rows.append({"origine": "2026-11-01", "mois": mois, "horizon": horizon,
                         "produit": produit, "timestral": val, "ajuste": val * 1.1})
    return pd.DataFrame(rows)


def _prevalence_synth():
    prev_2024 = {"prev_diabete": 10.0, "prev_cancer": 5.0, "prev_cardio": 8.0,
                 "prev_psy": 4.0, "prev_resp": 6.0, "prev_neuro": 2.0}
    prev_2030 = {
        11: [11.0, 5.5, 9.0, 4.4, 6.6, 2.2],
        53: [9.0, 4.5, 7.0, 3.6, 5.4, 1.8],
        94: [12.0, 6.0, 10.0, 4.8, 7.2, 2.4],
    }
    noms = {11: "Île-de-France", 53: "Bretagne", 94: "Corse"}
    rows = []
    for reg, valeurs in prev_2030.items():
        for patho, v in zip(PATHOLOGIES, valeurs):
            rows.append({"reg": reg, "region_nom": noms[reg], "patho": patho,
                         "prev_2024": prev_2024[patho], "prev_2030": v})
    return pd.DataFrame(rows)


def _population_synth():
    return pd.DataFrame({
        "reg": [11, 53, 94],
        "region_nom": ["Île-de-France", "Bretagne", "Corse"],
        "annee": [2026, 2026, 2026],
        "population": [1_000_000.0, 2_000_000.0, 500_000.0],
        "part_75plus": [8.0, 12.0, 14.0],
    })


class Normaliser0100Test(unittest.TestCase):
    def test_bornes_et_extremes(self):
        self.assertEqual(br.normaliser_0_100([10, 20, 30]), [0.0, 50.0, 100.0])
        self.assertEqual(br.normaliser_0_100([30, 10, 20]), [100.0, 0.0, 50.0])

    def test_toujours_dans_0_100(self):
        valeurs = [-5.0, 0.0, 3.7, 1000.0, 42.0]
        res = br.normaliser_0_100(valeurs)
        self.assertTrue(all(0.0 <= v <= 100.0 for v in res))
        self.assertEqual(min(res), 0.0)
        self.assertEqual(max(res), 100.0)

    def test_serie_constante_et_vide(self):
        self.assertEqual(br.normaliser_0_100([5.0, 5.0, 5.0]), [0.0, 0.0, 0.0])
        self.assertEqual(br.normaliser_0_100([7.0]), [0.0])
        self.assertEqual(br.normaliser_0_100([]), [])


class AttribuerRangsTest(unittest.TestCase):
    def test_rang_1_pour_le_plus_grand_indice(self):
        self.assertEqual(br.attribuer_rangs([10.0, 30.0, 20.0], ["a", "b", "c"]), [3, 1, 2])

    def test_rangs_toujours_uniques(self):
        import random
        random.seed(12)
        for _ in range(20):
            n = random.randint(1, 30)
            indices = [random.choice([1.0, 2.0, 3.0]) for _ in range(n)]
            cles = [f"r{i}" for i in range(n)]
            rangs = br.attribuer_rangs(indices, cles)
            self.assertEqual(sorted(rangs), list(range(1, n + 1)))

    def test_egalite_departagee_par_cle(self):
        rangs = br.attribuer_rangs([5.0, 5.0], ["b", "a"])
        self.assertEqual(sorted(rangs), [1, 2])
        self.assertEqual(rangs, [2, 1])  # « a » avant « b » -> rang 1 pour « a »


class PoidsPrevalencesTest(unittest.TestCase):
    def test_somme_1_et_proportions(self):
        prev = _prevalence_synth()
        poids = br.poids_prevalences(prev)
        self.assertAlmostEqual(sum(poids.values()), 1.0)
        # prev_2024 moyennes : diabète 10, cancer 5 -> poids diabète = 10/35
        self.assertAlmostEqual(poids["prev_diabete"], 10.0 / 35.0)
        self.assertAlmostEqual(poids["prev_cancer"], 5.0 / 35.0)

    def test_prevalences_nulles_rejetees(self):
        prev = _prevalence_synth()
        prev["prev_2024"] = 0.0
        with self.assertRaises(ValueError):
            br.poids_prevalences(prev)


class CalculerBesoinRegionalTest(unittest.TestCase):
    def setUp(self):
        self.df = br.calculer_besoin_regional(_previsions_synth(), _prevalence_synth(), _population_synth())

    def test_colonnes_et_taille(self):
        attendues = {"origine", "mois_cible", "horizon", "reg", "region_nom", "population",
                     "part_75plus", "pop_75plus", "boites_prevues_total",
                     "boites_amox250", "boites_amox500", "boites_azithro",
                     "prevalence_ponderee_2030", "indice_besoin_0_100", "rang"}
        self.assertTrue(attendues.issubset(set(self.df.columns)))
        self.assertEqual(len(self.df), 3)
        self.assertEqual(self.df.mois_cible.iloc[0], "2026-12-01")
        self.assertEqual(self.df.horizon.iloc[0], 1)

    def test_indice_dans_0_100_et_rangs_uniques(self):
        self.assertTrue(self.df.indice_besoin_0_100.between(0, 100).all())
        self.assertEqual(min(self.df.indice_besoin_0_100), 0.0)
        self.assertEqual(max(self.df.indice_besoin_0_100), 100.0)
        self.assertEqual(sorted(self.df.rang.tolist()), [1, 2, 3])
        # rang 1 = indice le plus élevé
        ligne_max = self.df.loc[self.df.indice_besoin_0_100.idxmax()]
        self.assertEqual(int(ligne_max.rang), 1)

    def test_conservation_des_boites_par_forme(self):
        for produit, national in [("amox250", 100.0), ("amox500", 200.0), ("azithro", 400.0)]:
            self.assertAlmostEqual(self.df[f"boites_{produit}"].sum(), national, places=6)
        self.assertAlmostEqual(
            self.df.boites_prevues_total.sum(), 700.0, places=6)
        for _, row in self.df.iterrows():
            self.assertAlmostEqual(
                row.boites_prevues_total,
                row.boites_amox250 + row.boites_amox500 + row.boites_azithro, places=6)

    def test_prevalence_ponderee_encadree(self):
        prev = _prevalence_synth()
        for _, row in self.df.iterrows():
            v = prev[prev.reg == row.reg].prev_2030
            self.assertGreaterEqual(row.prevalence_ponderee_2030, v.min() - 1e-9)
            self.assertLessEqual(row.prevalence_ponderee_2030, v.max() + 1e-9)

    def test_repartition_au_prorata_des_75plus(self):
        # Corse (14 % de 75+) a plus de boîtes par habitant qu'Île-de-France (8 %)
        par_hab = self.df.set_index("region_nom").apply(
            lambda r: r.boites_prevues_total / r.population, axis=1)
        self.assertGreater(par_hab["Corse"], par_hab["Île-de-France"])
        self.assertGreater(par_hab["Bretagne"], par_hab["Île-de-France"])

    def test_erreurs_de_donnees(self):
        with self.assertRaises(ValueError):
            br.calculer_besoin_regional(_previsions_synth(), _prevalence_synth(),
                                        _population_synth(), horizon=99)
        with self.assertRaises(ValueError):
            br.calculer_besoin_regional(_previsions_synth(), _prevalence_synth(),
                                        _population_synth(), colonne="inconnue")
        pop_sans_corse = _population_synth()[lambda d: d.reg != 94]
        with self.assertRaises(ValueError):
            br.calculer_besoin_regional(_previsions_synth(), _prevalence_synth(), pop_sans_corse)


@unittest.skipUnless((RACINE / "outputs" / "previsions_boites_pharma.csv").exists()
                     and (RACINE / "outputs" / "prevalence_projections_2030.csv").exists()
                     and br.POP_ZIP.exists(), "données du dépôt absentes")
class DonneesReellesTest(unittest.TestCase):
    def test_population_regions_complete(self):
        pop = br.charger_population_regions()
        self.assertEqual(len(pop), 18)
        self.assertFalse(pop.population.isna().any() or pop.part_75plus.isna().any())
        self.assertEqual(pop.annee.nunique(), 1)
        self.assertEqual(sorted(pop.reg.tolist()), sorted(br.REGIONS))

    def test_besoin_regional_reel(self):
        previsions = pd.read_csv(RACINE / "outputs" / "previsions_boites_pharma.csv")
        prevalence = pd.read_csv(RACINE / "outputs" / "prevalence_projections_2030.csv")
        df = br.calculer_besoin_regional(previsions, prevalence, br.charger_population_regions())
        self.assertEqual(len(df), 18)
        self.assertTrue(df.indice_besoin_0_100.between(0, 100).all())
        self.assertEqual(sorted(df.rang.tolist()), list(range(1, 19)))
        nat = previsions[previsions.horizon == 1].set_index("produit")["timestral"]
        for produit in nat.index:
            self.assertAlmostEqual(df[f"boites_{produit}"].sum(), nat[produit], delta=1.0)

    def test_main_ecrit_le_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = br.main(Path(tmp) / "besoin_regional.csv")
            lu = pd.read_csv(dest)
            self.assertEqual(len(lu), 18)
            self.assertTrue(lu.indice_besoin_0_100.between(0, 100).all())
            self.assertEqual(sorted(lu.rang.tolist()), list(range(1, 19)))
            self.assertIn("boites_amox250", lu.columns)
            self.assertIn("prevalence_ponderee_2030", lu.columns)


if __name__ == "__main__":
    unittest.main()
