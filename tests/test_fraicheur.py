"""Tests du bandeau de fraîcheur des dashboards et du délai réel branché au pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]

from src.dashboard import fraicheur_dashboard, generer_html
from src.delai_medicam import delais_pipeline
from src.manifest import lire_manifest


def _donnees_minimales():
    return {
        "territoire_defaut": "France", "territoires": [], "unite_horizon": "an",
        "horizons": [1], "modele_retenu": "Timestral", "modeles": {"Timestral": "Timestral"},
        "comparaisons": {}, "projections": {}, "contexte": [],
    }


class LireManifestTest(unittest.TestCase):
    def test_absent_puis_present_puis_invalide(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(lire_manifest(tmp))
            (Path(tmp) / "outputs").mkdir()
            (Path(tmp) / "outputs" / "manifest.json").write_text("{invalid", encoding="utf-8")
            self.assertIsNone(lire_manifest(tmp))
            (Path(tmp) / "outputs" / "manifest.json").write_text(
                json.dumps({"run": "x", "date_utc": "2026-11-21T20:00:00+00:00",
                            "revision_modele": "abcdef123456"}), encoding="utf-8")
            m = lire_manifest(tmp)
            self.assertEqual(m["run"], "x")


class FraicheurDashboardTest(unittest.TestCase):
    def test_paires_depuis_manifest_et_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "outputs").mkdir()
            (Path(tmp) / "outputs" / "manifest.json").write_text(json.dumps({
                "date_utc": "2026-11-21T20:00:00+00:00", "revision_modele": "0123456789abcdef"
            }), encoding="utf-8")
            f = fraicheur_dashboard(tmp, **{"Dernière année SAE": 2025})
            self.assertEqual(f["Tableau généré le"], "2026-11-21T20:00:00+00:00")
            self.assertEqual(f["Révision du modèle"], "0123456789ab")
            self.assertEqual(f["Dernière année SAE"], 2025)

    def test_generer_html_avec_et_sans_bandeau(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "dash.html"
            generer_html(dest, "Titre", "Accroche", _donnees_minimales(), "Note", "", [],
                         fraicheur={"Dernier mois": "2026-08"})
            html = dest.read_text(encoding="utf-8")
            self.assertIn('class="fraicheur"', html)
            self.assertIn("2026-08", html)
            dest2 = Path(tmp) / "dash2.html"
            generer_html(dest2, "Titre", "Accroche", _donnees_minimales(), "Note", "", [])
            self.assertNotIn('class="fraicheur"', dest2.read_text(encoding="utf-8"))


class DelaiPipelineTest(unittest.TestCase):
    def test_vrai_calendrier_branché(self):
        delais = delais_pipeline(RACINE)
        self.assertIsNotNone(delais, "le CSV des archives existe dans le dépôt")
        self.assertTrue(callable(delais))
        # Origine novembre 2026 (datée au 21) -> dernier mois documenté : août 2026.
        self.assertEqual(delais(pd.Period("2026-11", freq="M")), pd.Period("2026-08", freq="M"))

    def test_sans_archives_renvoie_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(delais_pipeline(tmp))


if __name__ == "__main__":
    unittest.main()
