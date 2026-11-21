"""Contrôles temporels du benchmark de volumes."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.modele_volumes_pharma import DELAI_MOIS
from src.recherche_archives_pharma import dates_avec_prescripteurs, mois_documentes
from src.recherche_delai_pharma import poids_du_melange
from src.recherche_signaux_pharma import signal_avant_origine
from src.recherche_volumes_pharma import (
    COLONNES_RIDGE,
    DEBUT_VERIFICATION,
    FIN_DEV,
    calculer_features,
    prevision_ridge,
)


class RecherchePharmaTest(unittest.TestCase):
    def test_choix_du_modele_avant_la_premiere_prevision_reservee(self):
        self.assertEqual(FIN_DEV, DEBUT_VERIFICATION - 1 - DELAI_MOIS)

    def test_volumes_non_publies_exclus_de_la_regression(self):
        dates = pd.period_range("2015-01", periods=60, freq="M")
        stocks = pd.DataFrame({"produit": 100 * dates.month + np.arange(len(dates))}, index=dates)
        jours = pd.Series(21, index=dates)
        sans_signal = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        cible = dates[-1]
        originaux = calculer_features(stocks, jours, sans_signal, sans_signal)
        modifie = stocks.copy()
        modifie.loc[cible - DELAI_MOIS:] *= 10
        apres = calculer_features(modifie, jours, sans_signal, sans_signal)

        self.assertNotEqual(originaux.at[cible, "observe"], apres.at[cible, "observe"])
        self.assertEqual(
            prevision_ridge(originaux, cible, COLONNES_RIDGE),
            prevision_ridge(apres, cible, COLONNES_RIDGE),
        )

    def test_poids_du_melange_ignore_la_verification(self):
        donnees = pd.DataFrame({
            "observe": [100, 0],
            "timestral_60_mois": [100, 1000],
            "saisonnier_ajuste": [0, 0],
        }, index=[FIN_DEV, DEBUT_VERIFICATION])
        self.assertEqual(poids_du_melange(donnees), 1.0)

    def test_signal_odisse_evalue_uniquement_les_semaines_deja_diffusees(self):
        origine = pd.Timestamp("2024-11-01")
        dates = pd.date_range("2023-09-25", "2024-11-11", freq="W-MON")
        source = pd.DataFrame({"urgences": 10.0}, index=dates)
        reference = signal_avant_origine(source, origine)
        source.loc[origine:, "urgences"] = 1000
        pd.testing.assert_series_equal(signal_avant_origine(source, origine), reference)
        self.assertEqual(reference.item(), 0.0)

    def test_archives_ne_franchit_pas_un_mois_non_documente(self):
        mois = pd.period_range("2024-01", periods=3, freq="M")
        volumes = pd.DataFrame({"a": [1, 2, 3]}, index=mois)
        editions = pd.Series(pd.to_datetime(["2024-04-01", "2024-06-10", "2024-05-25"]), index=mois)
        connus = mois_documentes(volumes, editions, pd.Timestamp("2024-06-10"))
        self.assertEqual(connus.index.tolist(), [mois[0]])

    def test_date_anticipee_exige_la_concordance_du_mois_entier(self):
        mois = pd.period_range("2024-01", periods=2, freq="M")
        volumes = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=mois)
        editions = pd.Series(pd.Timestamp("2024-06-10"), index=mois)
        prescripteurs = pd.DataFrame({
            "mois": ["2024-01", "2024-01", "2024-02", "2024-02"],
            "produit": ["a", "b", "a", "b"], "boites": [1, 3, 2, 10],
            "edition": "2024-04-02",
        })
        with TemporaryDirectory() as dossier:
            racine = Path(dossier)
            donnee = racine / "data" / "research" / "pharma"
            donnee.mkdir(parents=True)
            prescripteurs.to_csv(donnee / "medicam_prescripteurs_2023_2026.csv", index=False)
            dates = dates_avec_prescripteurs(racine, volumes, editions)
        self.assertEqual(dates.iloc[0], pd.Timestamp("2024-04-02"))
        self.assertEqual(dates.iloc[1], pd.Timestamp("2024-06-10"))
