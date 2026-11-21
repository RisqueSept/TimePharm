"""Tests du JSON sidecar accompagnant chaque sortie."""

import json
import tempfile
import unittest
from pathlib import Path

from src.manifest import ecrire_sidecar, hasher_fichier


class SidecarTest(unittest.TestCase):
    def test_sidecar_contient_hash_revisions_et_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            racine = Path(tmp)
            (racine / "outputs").mkdir()
            sortie = racine / "outputs" / "rapport.csv"
            sortie.write_text("a,b\n1,2\n", encoding="utf-8")
            (racine / "outputs" / "manifest.json").write_text(json.dumps({
                "run": "pharma", "revision_code": "abc123",
            }), encoding="utf-8")
            s = ecrire_sidecar(sortie, racine)
            self.assertEqual(s["fichier"], "rapport.csv")
            self.assertEqual(s["sha256"], hasher_fichier(sortie))
            self.assertEqual(s["run"], "pharma")
            self.assertEqual(s["revision_code"], "abc123")
            self.assertIn("revision_modele", s)
            self.assertIn("date_utc", s)
            lu = json.loads((racine / "outputs" / "rapport.csv.json").read_text(encoding="utf-8"))
            self.assertEqual(lu["sha256"], s["sha256"])

    def test_hash_change_si_sortie_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            racine = Path(tmp)
            sortie = racine / "f.txt"
            sortie.write_text("v1", encoding="utf-8")
            h1 = ecrire_sidecar(sortie, racine)["sha256"]
            sortie.write_text("v2", encoding="utf-8")
            h2 = ecrire_sidecar(sortie, racine)["sha256"]
            self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
