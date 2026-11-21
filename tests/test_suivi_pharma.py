import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.suivi_pharma import evaluer_previsions, figer_previsions


class SuiviPharmaTest(unittest.TestCase):
    def test_premiere_emission_et_observations_disponibles(self):
        manifest = {
            "date_utc": "2026-11-21T23:30:00Z", "modele": "Timestral", "revision_modele": "test",
            "revision_code": "test", "entrees": {},
        }
        tensions = pd.DataFrame([{
            "cible": pd.Timestamp("2026-12-04"), "produit": "amox250", "horizon": 14,
            "grossistes": 2, "timestral": 0.75,
        }])
        boites = pd.DataFrame([{
            "mois": pd.Timestamp("2026-12-01"), "produit": "amox250", "horizon": 1,
            "timestral": 2500, "ajuste": 2300,
        }])
        stocks = pd.DataFrame([{
            "observation": pd.Timestamp("2026-12-05"), "disponible": pd.Timestamp("2026-12-11"),
            "produit": "amox250", "grossistes": 0,
        }])
        volumes = boites[["mois", "produit"]].assign(boites=2400)
        with tempfile.TemporaryDirectory() as tmp:
            dossier = Path(tmp)
            archive = figer_previsions(dossier, tensions, boites, manifest)
            self.assertEqual(archive.name, "2026-11-22.json")
            figer_previsions(dossier, tensions.assign(timestral=0.10), boites, manifest)
            avant = evaluer_previsions(dossier, stocks, volumes.iloc[:0], pd.Timestamp("2026-12-10"))
            self.assertIsNone(avant["tensions"][0]["observe"])
            self.assertIsNone(avant["boites"][0]["observe"])
            apres = evaluer_previsions(dossier, stocks, volumes, pd.Timestamp("2027-02-01"))
            self.assertEqual(apres["tensions"][0]["observe"], 1)
            self.assertEqual(apres["tensions"][0]["timestral"], 0.75)
            self.assertEqual(apres["boites"][0]["observe"], 2400)
