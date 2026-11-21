"""Ancrage des seuils a priori : constantes présentes et documentées dans PASSATION."""

import unittest
from pathlib import Path

from src import seuils


class SeuilsTest(unittest.TestCase):
    def test_seuil_alerte_dans_0_1(self):
        self.assertGreaterEqual(seuils.SEUIL_ALERTE_RISQUE, 0.0)
        self.assertLessEqual(seuils.SEUIL_ALERTE_RISQUE, 1.0)

    def test_metriques_et_couverture_definies(self):
        self.assertIn("pr_auc", seuils.METRIQUES_TENSIONS)
        self.assertIn("biais", seuils.METRIQUES_VOLUMES)
        self.assertAlmostEqual(seuils.COUVERTURE_VISEE, 0.80)

    def test_passation_documente_les_seuils_a_priori(self):
        doc = (Path(__file__).resolve().parents[1] / "docs" / "PASSATION.md").read_text(encoding="utf-8")
        self.assertIn("a priori", doc)
        self.assertIn("SEUIL_ALERTE_RISQUE", doc)


if __name__ == "__main__":
    unittest.main()
