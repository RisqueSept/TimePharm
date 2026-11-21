"""Tests du dispatch de la CLI `python -m timestral`."""

import unittest
from unittest.mock import patch

from timestral.__main__ import SCRIPTS, construire_parser, main


class ParseTest(unittest.TestCase):
    def test_run_cible_et_args(self):
        a = construire_parser().parse_args(["run", "lits", "--skip-fetch"])
        self.assertEqual((a.commande, a.cible, a.args), ("run", "lits", ["--skip-fetch"]))

    def test_commandes(self):
        for argv, cmd in [(["dashboard", "mensuel"], "dashboard"),
                          (["test"], "test"),
                          (["recherche", "delai", "--date-depart", "2026-11-21"], "recherche"),
                          (["publier"], "publier")]:
            self.assertEqual(construire_parser().parse_args(argv).commande, cmd)

    def test_cible_invalide_rejetée(self):
        with self.assertRaises(SystemExit):
            construire_parser().parse_args(["run", "inconnu"])


class DispatchTest(unittest.TestCase):
    def test_run_appelle_le_bon_script(self):
        with patch("timestral.__main__._run_script") as m:
            main(["run", "pharma"])
        m.assert_called_once_with("main_pharma.py", [])

    def test_recherche_appelle_le_bon_main(self):
        with patch("timestral.__main__._run_script") as m:
            main(["recherche", "volumes"])
        m.assert_called_once_with("main_recherche_pharma.py", [])

    def test_scripts_couvrent_les_trois_pipelines(self):
        self.assertEqual(sorted(SCRIPTS), ["lits", "mensuel", "pharma"])


if __name__ == "__main__":
    unittest.main()
