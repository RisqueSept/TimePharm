"""Tests du manifeste d'exécution (hash des entrées, révisions, paramètres)."""

import json
import tempfile
import unittest
from pathlib import Path

from src.inference_timestral import REVISION
from src.manifest import entrees_dossier, ecrire_manifest, hasher_fichier


class ManifesteTest(unittest.TestCase):
    def test_hash_stable_puis_change_si_contenu_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.csv"
            f.write_text("a,b\n1,2\n", encoding="utf-8")
            h1 = hasher_fichier(f)
            self.assertEqual(h1, hasher_fichier(f))
            f.write_text("a,b\n1,3\n", encoding="utf-8")
            self.assertNotEqual(h1, hasher_fichier(f))

    def test_entrees_dossier_liste_les_fichiers_tries(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "b.csv").write_text("x")
            (Path(tmp) / "sous").mkdir()
            (Path(tmp) / "sous" / "a.csv").write_text("y")
            liste = entrees_dossier(tmp)
            self.assertEqual(liste, sorted(liste))
            self.assertEqual(len(liste), 2)
            self.assertEqual(entrees_dossier(Path(tmp) / "inexistant"), [])

    def test_manifeste_enregistre_hash_revisions_parametres_et_historique(self):
        with tempfile.TemporaryDirectory() as tmp:
            racine = Path(tmp)
            data = racine / "data" / "raw"
            data.mkdir(parents=True)
            (data / "x.csv").write_text("1,2\n", encoding="utf-8")
            m = ecrire_manifest(racine, "test", [data / "x.csv"], {"delai_mois": 2})
            self.assertEqual(m["run"], "test")
            self.assertEqual(m["revision_modele"], REVISION)
            self.assertEqual(m["entrees"], {"data/raw/x.csv": hasher_fichier(data / "x.csv")})
            self.assertEqual(m["parametres"], {"delai_mois": 2})
            self.assertIn("date_utc", m)
            lu = json.loads((racine / "outputs" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(lu["run"], "test")
            histo = list((racine / "outputs" / "manifests").glob("test_*.json"))
            self.assertEqual(len(histo), 1)

    def test_deux_runs_ecrivent_deux_manifests_historises(self):
        with tempfile.TemporaryDirectory() as tmp:
            racine = Path(tmp)
            (racine / "d.csv").write_text("1\n", encoding="utf-8")
            ecrire_manifest(racine, "run", [racine / "d.csv"], {})
            (racine / "d.csv").write_text("2\n", encoding="utf-8")
            ecrire_manifest(racine, "run", [racine / "d.csv"], {})
            histo = list((racine / "outputs" / "manifests").glob("run_*.json"))
            self.assertEqual(len(histo), 2)
            hashes = {json.loads(h.read_text(encoding="utf-8"))["entrees"]["d.csv"] for h in histo}
            self.assertEqual(len(hashes), 2, "deux contenus différents doivent donner deux hash différents")


if __name__ == "__main__":
    unittest.main()
