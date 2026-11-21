"""Vrai calendrier de délai Medic'AM : coupure fondée sur les éditions documentées.

- anti-fuite : les dates d'édition postérieures à l'origine ne changent pas la coupure ;
- le vrai `data/research/pharma/medicam_dates_archives.csv` donne une coupure
  conforme aux éditions documentées ;
- sans `delais`, `prevoir_volumes` reste identique au délai fixe `DELAI_MOIS`.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.delai_medicam import cutoff_pour, delais_reel, editions_archives
from src.modele_volumes_pharma import DELAI_MOIS, estimer, prevoir_volumes

RACINE = Path(__file__).resolve().parents[1]
ARCHIVES = RACINE / "data" / "research" / "pharma" / "medicam_dates_archives.csv"


class MoteurQuantiles:
    """Quantiles constants ; capture les historiques transmis au moteur."""

    def __init__(self):
        self.captures = []

    def quantiles(self, valeurs, pas, *, calendrier=None):
        self.captures.append(np.array(valeurs, dtype=float))
        return np.full((valeurs.shape[0], 1, max(pas, 0), 3), 1000.0)


def mini_volumes(n_mois=30, produits=("a", "b")):
    mois = pd.period_range("2024-01", periods=n_mois, freq="M")
    lignes = []
    for p in produits:
        for m in mois:
            lignes.append({"produit": p, "mois": m.start_time, "boites": 1000 + m.ordinal % 97})
    return pd.DataFrame(lignes)


def editions_synthetiques():
    """Éditions semestrielles fictives, façon Medic'AM, de 2024-01 à 2026-12."""
    tranches = [
        ("2024-01", "2024-06", "2024-10-01"),
        ("2024-07", "2024-12", "2025-03-31"),
        ("2025-01", "2025-06", "2025-12-04"),
        ("2025-07", "2025-12", "2026-04-20"),
        ("2026-01", "2026-06", "2026-09-16"),
        ("2026-07", "2026-12", "2026-11-06"),
    ]
    entrees = {}
    for debut, fin, edition in tranches:
        for mois in pd.period_range(debut, fin, freq="M"):
            entrees[mois] = pd.Timestamp(edition)
    return pd.Series(entrees)


class AntiFuiteDelaiTest(unittest.TestCase):
    def test_editions_posterieures_a_l_origine_ne_changent_pas_la_coupure(self):
        editions = editions_synthetiques()
        origine = pd.Timestamp("2025-03-15")
        reference = cutoff_pour(origine, editions)
        # L'archive 2024-07..2024-12 est publiée le 2025-03-31, après l'origine.
        self.assertEqual(reference, pd.Period("2024-06", freq="M"))
        modifiees = editions.copy()
        modifiees[editions >= origine] = pd.Timestamp("2099-01-01")
        self.assertEqual(cutoff_pour(origine, modifiees), reference)

    def test_prevoir_volumes_ne_fuit_pas_par_les_editions_du_futur(self):
        editions = editions_synthetiques()
        volumes = mini_volumes(30)  # 2024-01 .. 2026-06, origines jusqu'à 2026-05
        depart = pd.Timestamp("2026-03-01")
        test1, fut1 = prevoir_volumes(volumes, depart, MoteurQuantiles(),
                                      delais=delais_reel(editions, jour_origine=21))
        # Décale toutes les éditions publiées après la dernière origine (2026-05-21).
        modifiees = editions.copy()
        modifiees[editions > pd.Timestamp("2026-05-21")] = pd.Timestamp("2099-01-01")
        test2, fut2 = prevoir_volumes(volumes, depart, MoteurQuantiles(),
                                      delais=delais_reel(modifiees, jour_origine=21))
        pd.testing.assert_frame_equal(test1, test2)
        pd.testing.assert_frame_equal(fut1, fut2)


class CalendrierReelArchivesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.editions = editions_archives(ARCHIVES)

    def test_archives_couvrent_les_mois_attendus(self):
        # 30 archives semestrielles de 2012-01 à 2026-08, soit 176 mois.
        self.assertEqual(len(self.editions), 176)
        self.assertEqual(self.editions.index[0], pd.Period("2012-01", freq="M"))
        self.assertEqual(self.editions.index[-1], pd.Period("2026-08", freq="M"))

    def test_coupure_conforme_aux_editions_documentees(self):
        editions = self.editions
        # Archive 2024-07-a-12 publiée le 2025-03-31 : non disponible le 2025-03-15.
        self.assertEqual(cutoff_pour(pd.Timestamp("2025-03-15"), editions),
                         pd.Period("2024-06", freq="M"))
        # Archive 2026-07-a-08 publiée le 2026-11-06 : disponible le 2026-11-21.
        self.assertEqual(cutoff_pour(pd.Timestamp("2026-11-21"), editions),
                         pd.Period("2026-08", freq="M"))
        # Avant la première édition documentée (2019-10-01) : repli DELAI_MOIS.
        self.assertEqual(cutoff_pour(pd.Timestamp("2019-05-15"), editions),
                         pd.Period("2019-03", freq="M"))

    def test_coupure_respecte_les_editions_pour_toute_origine(self):
        editions = self.editions
        for origine in pd.date_range("2020-01-01", "2026-12-31", freq="MS"):
            for jour in (1, 21):
                date_origine = origine + pd.Timedelta(days=jour - 1)
                coupure = cutoff_pour(date_origine, editions)
                disponibles = (editions < date_origine).to_numpy()
                n = int(np.cumprod(disponibles).sum())
                if n == 0:
                    self.assertEqual(coupure, date_origine.to_period(freq="M") - DELAI_MOIS,
                                     msg=str(date_origine))
                else:
                    self.assertEqual(coupure, editions.index[n - 1], msg=str(date_origine))
                    self.assertLess(editions.iloc[n - 1], date_origine)
                    if n < len(editions):
                        self.assertGreaterEqual(editions.iloc[n], date_origine)

    def test_prevoir_volumes_utilise_le_vrai_calendrier(self):
        editions = self.editions
        volumes = mini_volumes(30)  # 2024-01 .. 2026-06
        depart = pd.Timestamp("2026-03-01")
        test, fut = prevoir_volumes(volumes, depart, MoteurQuantiles(),
                                    delais=delais_reel(editions, jour_origine=21))
        # Coupure du départ au 2026-03-21 : l'archive 2025-07-a-12 (2026-04-20) n'est
        # pas publiée, celle de 2025-01-a-06 (2025-12-04) l'est -> 2025-06.
        self.assertEqual(cutoff_pour(pd.Timestamp("2026-03-21"), editions),
                         pd.Period("2025-06", freq="M"))
        self.assertEqual({pd.Period(ts, freq="M") for ts in fut.dernier_mois_utilise},
                         {pd.Period("2025-06", freq="M")})
        # Chaque origine du backtest respecte les éditions documentées à son 21.
        dernier_volumes = pd.Period("2026-06", freq="M")
        self.assertGreater(len(test), 0)
        for _, ligne in test.iterrows():
            origine = pd.Period(ligne.origine, freq="M")
            attendu = min(cutoff_pour(origine.start_time + pd.Timedelta(days=20), editions),
                          dernier_volumes)
            self.assertEqual(pd.Period(ligne.dernier_mois_utilise, freq="M"), attendu,
                             msg=str(origine))


class NonRegressionDelaiFixeTest(unittest.TestCase):
    def test_sans_delais_comportement_identique_au_delai_fixe(self):
        volumes = mini_volumes(30)
        depart = pd.Timestamp("2026-01-01")
        test1, fut1 = prevoir_volumes(volumes, depart, MoteurQuantiles())
        origines = pd.period_range("2023-10", "2026-06", freq="M")
        # Délai fixe explicite, en fonction puis en dict (clés Period et str) :
        # même résultat que sans `delais`.
        variantes = [
            lambda mois: mois - DELAI_MOIS,
            {m: m - DELAI_MOIS for m in origines},
            {str(m): m - DELAI_MOIS for m in origines},
        ]
        for delais in variantes:
            test2, fut2 = prevoir_volumes(volumes, depart, MoteurQuantiles(), delais=delais)
            pd.testing.assert_frame_equal(test1, test2)
            pd.testing.assert_frame_equal(fut1, fut2)

    def test_estimer_sans_delais_coupure_fixe(self):
        serie = pd.Series({m: 1000 + i for i, m in
                           enumerate(pd.period_range("2023-01", periods=36, freq="M"))})
        origine = pd.Period("2025-11", freq="M")
        reference = estimer(serie, origine, 1)
        self.assertIsNotNone(reference)
        self.assertEqual(reference, estimer(serie, origine, 1, delais=lambda m: m - DELAI_MOIS))
        attendu = serie.loc[:origine - DELAI_MOIS].index[-1].to_timestamp()
        self.assertEqual(reference["dernier_mois_utilise"], attendu)


if __name__ == "__main__":
    unittest.main()
