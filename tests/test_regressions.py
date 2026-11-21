"""Tests des agrégations et de la chronologie des projections."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import extract_lits
import modele_lits
import modele_mensuel
import table_mensuelle


class CapacitesTest(unittest.TestCase):
    def test_psy_et_readaptation_ne_cumulent_pas_details_et_totaux(self):
        psy = pd.DataFrame({"DIS": ["GEN", "INF", "TOT"], "CAP_HTP": [7, 3, 10]})
        smr = pd.DataFrame({"GDE": ["SSR_A", "SSR_A50", "SSR_TOT"], "LIT": [10, 4, 10]})
        self.assertEqual(extract_lits._lits_pour_bordereau(psy, "PSY"), 10)
        self.assertEqual(extract_lits._lits_pour_bordereau(smr, "SSR"), 10)
        self.assertEqual(extract_lits._lits_pour_bordereau(smr, "SMR"), 10)
        with self.assertRaises(ValueError):
            extract_lits._lits_pour_bordereau(psy.iloc[:2], "PSY")

    def test_population_projetee_independante_du_bloc_de_test(self):
        hist = pd.DataFrame({
            "annee": range(2013, 2023),
            "population": range(60_000, 70_000, 1_000),
            "lits_total": range(500, 400, -10),
        })
        hist["lits_pour_1000"] = 1000 * hist.lits_total / hist.population
        bt, _ = modele_lits._backtest(hist)
        hist.loc[hist.annee.gt(2018), "population"] *= 1000
        hist["lits_pour_1000"] = 1000 * hist.lits_total / hist.population
        autre, _ = modele_lits._backtest(hist)
        pd.testing.assert_series_equal(
            bt.loc[bt.origine.eq(2018), "pred_B_ratio_x_pop"],
            autre.loc[autre.origine.eq(2018), "pred_B_ratio_x_pop"],
        )


class MensuelTest(unittest.TestCase):
    def test_prevalence_ameli_utilise_le_code_agrege(self):
        brut = pd.DataFrame({
            "region": [11, 11], "annee": [2024, 2024],
            "top": ["CAN_CAT_CAT", "CAN_BPU_ACT"], "prev": [4.356, 0.140],
            "Ntop": [534140, 17230], "Npop": [12261580, 12261580],
        })
        with tempfile.TemporaryDirectory() as tmp, patch.object(table_mensuelle, "RAW", tmp):
            brut.to_csv(Path(tmp) / "ameli_pathologies_regions.csv", sep=";", index=False)
            self.assertEqual(table_mensuelle.charger_prevalences().prev_cancer.item(), 4.356)

    def test_selection_spf_et_geographie(self):
        raw = pd.DataFrame({
            "reg": [44] * 3, "jour": ["2023-06-01"] * 3, "PourAvec": [0, 1, 2],
            "tx_prev_hosp": [15, 10, 5], "tx_indic_7J_hosp": [3, 2, 1],
        })
        with tempfile.TemporaryDirectory() as tmp, patch.object(table_mensuelle, "RAW", tmp):
            raw.to_csv(Path(tmp) / "covid_hosp_txad_reg.csv", sep=";", index=False)
            self.assertEqual(table_mensuelle.charger_covid().tx_prev_hosp.tolist(), [15])
        self.assertEqual(table_mensuelle.REGIONS[44], "Grand Est")
        self.assertEqual(table_mensuelle.REGIONS[52], "Pays de la Loire")

    def test_une_prevision_par_region_et_mois_et_somme_france(self):
        dates = pd.date_range("2022-07-01", periods=12, freq="MS", name="mois")
        hist = pd.DataFrame({44: range(12), 52: range(10, 22), 75: range(20, 32)}, index=dates)
        hist.columns.name = "reg"
        moteur = Mock()
        moteur.quantiles.return_value = np.broadcast_to(
            np.array([11, 21, 31])[:, None, None, None], (3, 1, modele_mensuel.HORIZON, 3)
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(modele_mensuel, "OUT", tmp):
            regions, france = modele_mensuel.exporter_previsions(hist, moteur)
        moteur.quantiles.assert_called_once()
        self.assertEqual(regions.loc[regions.est_prevision.eq(1), "modele"].unique().tolist(), ["Timestral"])
        prev = regions.loc[regions.est_prevision.eq(1)]
        self.assertFalse(prev.duplicated(["reg", "mois"]).any())
        self.assertEqual(len(prev), 3 * modele_mensuel.HORIZON)
        self.assertTrue(prev.groupby("mois").reg.nunique().eq(3).all())
        self.assertTrue(france.loc[france.est_prevision.eq(1), "lits_pred"].eq(63).all())


if __name__ == "__main__":
    unittest.main()
