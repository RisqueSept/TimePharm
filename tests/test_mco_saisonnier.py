"""Tests des scénarios MCO à distribution saisonnière (chantier #11).

Vérifie que le profil mensuel par région est valide (parts sommant à 1),
que le pic mensuel est toujours >= la moyenne annuelle du scénario, que les
lits équivalents moyens restent cohérents avec scenarios_capacite_mco_regions.csv
et scenarios_capacite_mco.csv, et que les fonctions existantes ne sont pas
modifiées (non-régression).
"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import modele_lits

BASE = Path(__file__).resolve().parents[1]
COLONNES = ["annee_reference", "reg", "territoire", "lits_sae_mco_au_31_decembre",
            "journees_mco_reference", "variation_activite", "taux_occupation_cible", "mois",
            "jours_du_mois", "part_saisonnier", "journees_mco_mois",
            "lits_equivalents_pic", "lits_equivalents_moyen", "source_profil"]
N_REGIONS = 19
N_HYPOTHESES = len(modele_lits.HYPOTHESES_MCO)


def _ecrire_mco_synthetique(dossier):
    """mco_regional_reference.csv synthétique : reg 11 = 100 j/jour, reg 94 = 20 j/jour."""
    df = pd.DataFrame({
        "annee": [2025, 2025],
        "reg": [11, 94],
        "lit_mco": [1000.0, 500.0],
        "jou_mco": [36500.0, 7300.0],
        "jli_mco": [100000.0, 20000.0],
    })
    df.to_csv(Path(dossier) / "mco_regional_reference.csv", index=False)
    return df


class ProfilSaisonnierTest(unittest.TestCase):
    def test_parts_somment_a_1_avec_12_mois_par_region(self):
        prof = modele_lits.charger_profil_saisonnier_mco(2025, regions=[11, 94, 5])
        self.assertEqual(sorted(prof.reg.unique().tolist()), [5, 11, 94])
        g = prof.groupby("reg")
        self.assertTrue(g.mois.nunique().eq(12).all())
        self.assertTrue(np.allclose(g.part.sum(), 1.0))
        self.assertTrue((prof.part > 0).all())

    def test_region_sans_serie_mensuelle_recoit_le_profil_plat_documente(self):
        # la région 5 (Saint-Pierre-et-Miquelon) n'est pas dans la série mensuelle SPF
        prof = modele_lits.charger_profil_saisonnier_mco(2025, regions=[5])
        self.assertEqual(prof.source_profil.unique().tolist(), ["profil_plat"])
        jours = modele_lits._jours_par_mois(2025)
        for _, row in prof.iterrows():
            self.assertAlmostEqual(row.part, jours[row.mois] / 365)

    def test_profil_plat_si_aucune_donnee_mensuelle(self):
        vide = pd.DataFrame(columns=["reg", "mois", "niveau"])
        with patch.object(modele_lits, "_niveau_mensuel_hosp_regional", return_value=vide):
            prof = modele_lits.charger_profil_saisonnier_mco(2025, regions=[11])
        self.assertEqual(prof.source_profil.unique().tolist(), ["profil_plat"])
        self.assertTrue(np.allclose(prof.groupby("reg").part.sum(), 1.0))

    def test_profil_depuis_niveaux_synthetiques(self):
        # niveau 10 en janvier, 0 ailleurs -> toute l'année est en janvier
        niveau = pd.DataFrame({"reg": [11, 11], "mois": [1, 2], "niveau": [10.0, 0.0]})
        with patch.object(modele_lits, "_niveau_mensuel_hosp_regional", return_value=niveau):
            prof = modele_lits.charger_profil_saisonnier_mco(2025, regions=[11])
        p = prof.set_index("mois").part
        self.assertEqual(prof.source_profil.iloc[0], "spf_covid_mensuel")
        self.assertAlmostEqual(p[1], 1.0)
        self.assertAlmostEqual(p.drop(1).sum(), 0.0)


class ScenariosSaisonnierTest(unittest.TestCase):
    """Scénarios générés sur les vraies données de référence (data/processed)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        with patch.object(modele_lits, "OUT", cls._tmp.name):
            cls.res = modele_lits.scenarios_mco_saisonnier()
        cls.regions = cls.res["regions"]
        cls.national = cls.res["national"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_fichiers_csv_et_colonnes(self):
        for nom in ("scenarios_capacite_mco_saisonnier_regions.csv",
                    "scenarios_capacite_mco_saisonnier_national.csv"):
            chemin = Path(self._tmp.name) / nom
            self.assertTrue(chemin.exists(), nom)
            self.assertEqual(pd.read_csv(chemin).columns.tolist(), COLONNES)

    def test_structure_regions_scenarios_mois(self):
        r = self.regions
        self.assertEqual(len(r), N_REGIONS * N_HYPOTHESES * 12)
        self.assertEqual(r.reg.nunique(), N_REGIONS)
        self.assertTrue(r.groupby(["reg", "variation_activite", "taux_occupation_cible"])
                        .mois.nunique().eq(12).all())
        self.assertEqual(len(self.national), N_HYPOTHESES * 12)
        self.assertEqual(self.national.reg.unique().tolist(), [0])
        self.assertEqual(self.national.territoire.unique().tolist(), ["France entière"])

    def test_pic_mensuel_superieur_ou_egal_a_la_moyenne(self):
        for df, cle in ((self.regions, ["reg", "variation_activite", "taux_occupation_cible"]),
                        (self.national, ["variation_activite", "taux_occupation_cible"])):
            g = df.groupby(cle).agg(pic=("lits_equivalents_pic", "max"),
                                    moyen=("lits_equivalents_moyen", "first"))
            self.assertTrue((g.pic >= g.moyen).all())

    def test_somme_des_journees_mensuelles_proche_de_la_reference(self):
        s = (self.regions.groupby(["reg", "variation_activite", "taux_occupation_cible"])
             .agg(sj=("journees_mco_mois", "sum"), ref=("journees_mco_reference", "first")))
        # journees_mco_mois est arrondi à 0,1 dans le CSV : tolérance < 12 * 0,05
        self.assertTrue(np.allclose(s.sj, s.ref, atol=1.0))

    def test_coherence_moyenne_avec_scenarios_capacite_mco_regions(self):
        old = pd.read_csv(BASE / "outputs" / "scenarios_capacite_mco_regions.csv")
        m = (self.regions.groupby(["reg", "variation_activite", "taux_occupation_cible"])
             .lits_equivalents_moyen.first().reset_index())
        j = m.merge(old, on=["reg", "variation_activite", "taux_occupation_cible"])
        self.assertEqual(len(j), N_REGIONS * N_HYPOTHESES)
        self.assertTrue((j.lits_equivalents_moyen == j.lits_ouverts_equivalents).all())

    def test_coherence_moyenne_nationale_avec_scenarios_capacite_mco(self):
        old = pd.read_csv(BASE / "outputs" / "scenarios_capacite_mco.csv")
        # le CSV national répète les mêmes hypothèses pour chaque année projetée
        self.assertTrue(old.groupby(["variation_activite", "taux_occupation_cible"])
                        .lits_ouverts_equivalents.nunique().eq(1).all())
        old = old.drop_duplicates(subset=["variation_activite", "taux_occupation_cible"])
        mn = (self.national.groupby(["variation_activite", "taux_occupation_cible"])
              .lits_equivalents_moyen.first().reset_index())
        j = mn.merge(old, on=["variation_activite", "taux_occupation_cible"])
        self.assertEqual(len(j), N_HYPOTHESES)
        self.assertTrue((j.lits_equivalents_moyen == j.lits_ouverts_equivalents).all())

    def test_national_egale_somme_des_regions(self):
        sn = (self.regions.groupby(["variation_activite", "taux_occupation_cible", "mois"], as_index=False)
              .journees_mco_mois.sum()
              .merge(self.national, on=["variation_activite", "taux_occupation_cible", "mois"],
                     suffixes=("_reg", "_nat")))
        self.assertTrue(np.allclose(sn.journees_mco_mois_reg, sn.journees_mco_mois_nat, rtol=1e-9))
        annuel = pd.read_csv(BASE / "data" / "processed" / "lits_sae_annuel.csv")
        total_ref = float(annuel.iloc[-1].jou_mco)
        self.assertEqual(float(self.national.journees_mco_reference.iloc[0]), total_ref)

    def test_outputs_committes_present_et_coherents(self):
        for nom in ("scenarios_capacite_mco_saisonnier_regions.csv",
                    "scenarios_capacite_mco_saisonnier_national.csv"):
            self.assertTrue((BASE / "outputs" / nom).exists(), nom)
        r = pd.read_csv(BASE / "outputs" / "scenarios_capacite_mco_saisonnier_regions.csv")
        self.assertEqual(r.columns.tolist(), COLONNES)
        g = r.groupby(["reg", "variation_activite", "taux_occupation_cible"]).agg(
            pic=("lits_equivalents_pic", "max"), moyen=("lits_equivalents_moyen", "first"))
        self.assertTrue((g.pic >= g.moyen).all())


class CalculSynthetiqueTest(unittest.TestCase):
    def test_calcul_exact_avec_profil_connu(self):
        with tempfile.TemporaryDirectory() as tmp:
            _ecrire_mco_synthetique(tmp)
            profil = pd.DataFrame(
                [{"reg": reg, "mois": m, "part": 0.5 if m == 1 else 0.5 / 11,
                  "source_profil": "test"} for reg in (11, 94) for m in range(1, 13)])
            with patch.object(modele_lits, "PROCESSED", tmp), patch.object(modele_lits, "OUT", tmp):
                r = modele_lits.scenarios_mco_saisonnier_regionaux(profil=profil)
                n = modele_lits.scenarios_mco_saisonnier_national(r)
        ligne = r[(r.reg == 11) & (r.variation_activite == 0.0)
                  & (r.taux_occupation_cible == 0.8) & (r.mois == 1)].iloc[0]
        self.assertAlmostEqual(ligne.journees_mco_mois, 18250.0, places=1)
        self.assertEqual(ligne.lits_equivalents_pic, int(np.ceil(18250.0 / 31 / 0.8)))
        self.assertEqual(ligne.lits_equivalents_moyen, int(np.ceil(36500.0 / 365 / 0.8)))
        self.assertGreaterEqual(ligne.lits_equivalents_pic, ligne.lits_equivalents_moyen)
        # février : part 0,5/11 -> charge plus faible qu'en janvier
        fev = r[(r.reg == 11) & (r.variation_activite == 0.0)
                & (r.taux_occupation_cible == 0.8) & (r.mois == 2)].iloc[0]
        self.assertLess(fev.lits_equivalents_pic, ligne.lits_equivalents_pic)
        # national = somme des deux régions synthétiques
        self.assertAlmostEqual(float(n.journees_mco_reference.iloc[0]), 36500.0 + 7300.0)
        self.assertEqual(len(n), N_HYPOTHESES * 12)


class NonRegressionTest(unittest.TestCase):
    def test_scenarios_mco_regionaux_inchanges(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(modele_lits, "OUT", tmp):
            df = modele_lits.scenarios_mco_regionaux()
            self.assertTrue((Path(tmp) / "scenarios_capacite_mco_regions.csv").exists())
        self.assertEqual(df.columns.tolist(), [
            "annee_reference", "reg", "lits_sae_mco_au_31_decembre", "journees_mco_reference",
            "territoire", "taux_occupation_observe_ref", "variation_activite",
            "taux_occupation_cible", "lits_ouverts_equivalents", "ecart_equivalents_moins_capacite"])
        self.assertEqual(len(df), N_REGIONS * N_HYPOTHESES)
        self.assertTrue(np.issubdtype(df.lits_ouverts_equivalents.dtype, np.integer))

    def test_entrainer_et_prevoir_produit_aussi_les_scenarios_saisonniers(self):
        hist = pd.DataFrame({
            "annee": list(range(2019, 2026)),
            "lits_total": [400000 - 1000 * i for i in range(7)],
            "lits_mco": [180000.0] * 7, "lits_psy": [50000.0] * 7,
            "lits_ssr": [100000.0] * 7, "lits_usld": [30000.0] * 7,
            "jou_mco": [51000000.0] * 7, "jli_mco": [65000000.0] * 7,
            "population": [67_000_000 + 100_000 * i for i in range(7)],
            "pop_75plus": [6_000_000.0] * 7,
        })
        hist["lits_pour_1000"] = 1000 * hist.lits_total / hist.population
        prev = pd.DataFrame({"annee": [2026, 2027], "lits_pred": [390000, 388000],
                             "lits_mco": [180000, 179000]})
        moteur = Mock()
        moteur.quantiles.return_value = np.zeros((1, 1, 10, 3))
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(modele_lits, "OUT", tmp), \
                patch.object(modele_lits, "scenarios_mco_saisonnier") as mock_saisonnier:
            modele_lits.entrainer_et_prevoir(hist, moteur)
        mock_saisonnier.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
