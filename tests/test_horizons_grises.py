"""Tests des horizons hors validation grisés (chantier #13).

Logique de classification pure (`est_hors_validation` et friends, définies dans
`src/plots_mensuel.py` et réutilisées par `src/plots_interactifs.py`), légende
« hors validation » de la note de projection, passage du champ `hors_validation`
à travers `generer_html` (rétrocompatible), et classification sur les vraies
sorties du dépôt.
"""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

import plots_interactifs
import plots_mensuel
from plots_mensuel import (DELAI_VALIDATION, est_hors_validation, limite_validation,
                          marquer_hors_validation, note_hors_validation)
from src.dashboard import generer_html


class DelaiValidationTest(unittest.TestCase):
    def test_horizons_du_backtest(self):
        self.assertEqual(DELAI_VALIDATION, {"an": 3, "mois": 12})


class EstHorsValidationTest(unittest.TestCase):
    def test_annuel_3_ans_apres_la_derniere_observation(self):
        dernier = 2025
        self.assertFalse(est_hors_validation(2025, dernier, "an"))
        self.assertFalse(est_hors_validation(2028, dernier, "an"))   # 3 ans : encore validé
        self.assertTrue(est_hors_validation(2029, dernier, "an"))    # 4 ans : hors validation
        self.assertTrue(est_hors_validation(2035, dernier, "an"))

    def test_annuel_accepte_les_dates_iso_et_un_delai_explicite(self):
        self.assertTrue(est_hors_validation("2029-06-01", 2025, "an"))
        self.assertFalse(est_hors_validation("2028-12-31", "2025-01-01", "an"))
        self.assertTrue(est_hors_validation(2027, 2025, "an", delai_max=1))
        self.assertFalse(est_hors_validation(2026, 2025, "an", delai_max=1))

    def test_mensuel_12_mois_apres_la_derniere_observation(self):
        dernier = "2023-06-01"
        self.assertFalse(est_hors_validation("2023-07-01", dernier, "mois"))
        self.assertFalse(est_hors_validation("2024-06-01", dernier, "mois"))  # 12 mois : validé
        self.assertTrue(est_hors_validation("2024-07-01", dernier, "mois"))   # 13 mois : hors
        self.assertTrue(est_hors_validation("2025-06-01", dernier, "mois"))   # 24 mois : hors

    def test_unite_inconnue_rejetee(self):
        with self.assertRaises(ValueError):
            est_hors_validation(2030, 2025, "semaine")
        with self.assertRaises(ValueError):
            limite_validation(2025, "semaine")
        with self.assertRaises(ValueError):
            note_hors_validation(2025, "semaine")


class LimiteValidationTest(unittest.TestCase):
    def test_premiere_date_hors_validation(self):
        self.assertEqual(limite_validation(2025, "an"), 2029)
        self.assertEqual(limite_validation("2023-06-01", "mois"), "2024-07")
        self.assertEqual(limite_validation(2025, "an", delai_max=1), 2027)


class MarquerHorsValidationTest(unittest.TestCase):
    def test_champ_ajoute_sans_perdre_les_cles(self):
        records = [{"date": 2027, "lits_pred": 10}, {"date": 2029, "lits_pred": 12}]
        marques = marquer_hors_validation(records, 2025, "an")
        self.assertEqual([m["hors_validation"] for m in marques], [0, 1])
        self.assertEqual(marques[0]["lits_pred"], 10)
        self.assertEqual(marques[1]["date"], 2029)

    def test_mensuel(self):
        records = [{"date": "2024-06-01T00:00:00"}, {"date": "2024-07-01T00:00:00"}]
        marques = marquer_hors_validation(records, "2023-06-01T00:00:00", "mois")
        self.assertEqual([m["hors_validation"] for m in marques], [0, 1])


class NoteHorsValidationTest(unittest.TestCase):
    def test_note_annuelle(self):
        note = note_hors_validation(2025, "an")
        self.assertIn("hors validation", note)
        self.assertIn("2029", note)
        self.assertIn("3 ans", note)
        self.assertIn("2025", note)

    def test_note_mensuelle(self):
        note = note_hors_validation("2023-06-01", "mois")
        self.assertIn("hors validation", note)
        self.assertIn("2024-07", note)
        self.assertIn("12 mois", note)
        self.assertIn("2023-06", note)


class ReexportPlotsInteractifsTest(unittest.TestCase):
    def test_plots_interactifs_reutilise_la_logique_pure(self):
        self.assertIs(plots_interactifs.est_hors_validation, est_hors_validation)
        self.assertIs(plots_interactifs.limite_validation, limite_validation)
        self.assertIs(plots_interactifs.marquer_hors_validation, marquer_hors_validation)
        self.assertIs(plots_interactifs.note_hors_validation, note_hors_validation)
        self.assertEqual(plots_interactifs.DELAI_VALIDATION, DELAI_VALIDATION)
        self.assertTrue(plots_interactifs.est_hors_validation(2030, 2025, "an"))


class GenererHtmlRetrocompatibleTest(unittest.TestCase):
    def test_champ_hors_validation_et_note_passen_t_a_travers(self):
        donnees = {
            "territoire_defaut": "France", "territoires": [], "unite_horizon": "an",
            "horizons": [1], "modele_retenu": "Timestral", "modeles": {"Timestral": "Timestral"},
            "comparaisons": {}, "contexte": [],
            "projections": {"France": [
                {"date": 2026, "lits_pred": 360000, "est_prevision": 1, "hors_validation": 0},
                {"date": 2030, "lits_pred": 350000, "est_prevision": 1, "hors_validation": 1},
            ]},
        }
        note = "Note de projection. " + note_hors_validation(2025, "an")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "dash.html"
            generer_html(dest, "Titre", "Accroche", donnees, note, "", [])
            html = dest.read_text(encoding="utf-8")
            self.assertIn("hors validation", html)  # la note est dans le HTML
            m = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>',
                          html, re.DOTALL)
            self.assertIsNotNone(m)
            data = json.loads(m.group(1))
            self.assertEqual(data["projections"]["France"][1]["hors_validation"], 1)
            self.assertEqual(data["projections"]["France"][0]["hors_validation"], 0)


@unittest.skipUnless((RACINE / "outputs" / "prediction_lits_2026_2035.csv").exists()
                     and (RACINE / "outputs" / "prevision_mensuelle_regions.csv").exists()
                     and (RACINE / "data" / "processed" / "dataset_annuel.csv").exists(),
                     "sorties du dépôt absentes")
class DonneesReellesTest(unittest.TestCase):
    def test_lits_2029_2035_hors_validation(self):
        hist = pd.read_csv(RACINE / "data" / "processed" / "dataset_annuel.csv")
        prev = pd.read_csv(RACINE / "outputs" / "prediction_lits_2026_2035.csv")
        dernier = int(hist.annee.max())
        hors = sorted(int(a) for a in prev.annee if est_hors_validation(a, dernier, "an"))
        valides = sorted(int(a) for a in prev.annee if not est_hors_validation(a, dernier, "an"))
        self.assertEqual(valides, [2026, 2027, 2028])       # backtest 1-3 ans
        self.assertEqual(hors, list(range(2029, 2036)))      # au-delà de 3 ans après 2025

    def test_mensuel_au_dela_de_12_mois_hors_validation(self):
        full = pd.read_csv(RACINE / "outputs" / "prevision_mensuelle_regions.csv")
        dernier = full.loc[full.est_prevision == 0, "mois"].max()
        proj = full[full.est_prevision == 1]
        hors = sorted({m for m in proj.mois if est_hors_validation(m, dernier, "mois")})
        valides = sorted({m for m in proj.mois if not est_hors_validation(m, dernier, "mois")})
        self.assertEqual(hors[0], "2024-07-01")             # 13 mois après 2023-06
        self.assertEqual(len(hors), 12)                      # 2024-07 -> 2025-06
        self.assertEqual(valides[-1], "2024-06-01")          # 12 mois : encore validé


if __name__ == "__main__":
    unittest.main()
