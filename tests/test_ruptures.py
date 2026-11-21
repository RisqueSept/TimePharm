"""Tests de la détection de ruptures de séries."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ruptures import detecter_ruptures, rapport_ruptures


class RupturesTest(unittest.TestCase):
    def test_serie_reguliere_sans_rupture(self):
        index = pd.date_range("2020-01-01", periods=60, freq="7D")
        serie = pd.Series(100 + np.sin(np.arange(60) * 0.5), index=index)
        r = detecter_ruptures(serie, fenetre=12, seuil=3.0)
        self.assertFalse(r.any())

    def test_saut_brutal_detecte_a_la_bonne_date(self):
        index = pd.date_range("2020-01-01", periods=60, freq="7D")
        valeurs = np.full(60, 100.0)
        valeurs[40:] = 300.0
        serie = pd.Series(valeurs, index=index)
        r = detecter_ruptures(serie, fenetre=12, seuil=3.0)
        self.assertTrue(r.iloc[40])
        self.assertFalse(r.iloc[:40].any())

    def test_serie_trop_courte_ne_leve_rien(self):
        serie = pd.Series([1.0, 2.0, 1000.0], index=pd.date_range("2020-01-01", periods=3))
        self.assertFalse(detecter_ruptures(serie, min_points=24).any())

    def test_rapport_sur_vrai_jeu_de_donnees(self):
        racine = Path(__file__).resolve().parents[1]
        rapport = rapport_ruptures(racine / "data" / "raw", racine / "data" / "research")
        self.assertIn("serie", rapport.columns)
        self.assertGreater(len(rapport), 0, "au moins une série suivie doit exister")
        self.assertTrue(rapport.serie.isin(
            ["sentinelles_ira", "medicam_total", "ansm_part_insuffisante", "odisse_pediatrie"]
        ).all())


if __name__ == "__main__":
    unittest.main()
