"""Mesure rétrospective de l'intérêt éventuel de volumes plus récents."""

import numpy as np
import pandas as pd

from .inference_timestral import MistralTimestral
from .modele_volumes_pharma import DELAI_MOIS, estimer
from .recherche_volumes_pharma import (
    DEBUT_VERIFICATION,
    FIN_DEV,
    charger,
    quantiles_timestral,
    scores,
    timestral,
)


TIMESTRAL = "timestral_60_mois"
SAISONNIER = "saisonnier_ajuste"
POIDS = np.linspace(0, 1, 5)


def poids_du_melange(developpement):
    """Choisit le poids par minimisation de l'erreur absolue sur le développement."""
    donnees = developpement.loc[developpement.index <= FIN_DEV]
    predictions = (donnees[TIMESTRAL].to_numpy()[:, None] * POIDS
                   + donnees[SAISONNIER].to_numpy()[:, None] * (1 - POIDS))
    erreurs = np.abs(predictions - donnees.observe.to_numpy()[:, None]).mean(axis=0)
    return float(POIDS[erreurs.argmin()])


def prevision_prospective(table, jours, moteur, date_depart, poids):
    origine = pd.Period(date_depart, freq="M")
    cible = origine + 1
    if cible <= table.index.max():
        raise ValueError("La cible prospective figure déjà dans les observations Medic’AM.")
    quantiles = quantiles_timestral(moteur, table, jours, cible, fenetre=60, mode="formes",
                                    calendrier_ouvre=False)
    medianes = np.rint(quantiles[:, 0, -1, 1]).astype(int)
    ajustees = [estimer(table[produit], origine, 1)["ajuste"] for produit in table.columns]
    par_forme = pd.DataFrame({
        "produit": table.columns, "timestral_60_mois": medianes, "saisonnier_ajuste": ajustees,
        "melange": np.rint(poids * medianes + (1 - poids) * np.asarray(ajustees)).astype(int),
    })
    valeurs = ["timestral_60_mois", "saisonnier_ajuste", "melange"]
    total = pd.DataFrame([dict(produit="total", **par_forme[valeurs].sum().to_dict())])
    return pd.concat([par_forme, total], ignore_index=True).assign(
        date_depart=date_depart, mois=str(cible), poids_timestral=poids,
        dernier_mois_utilise=str(table.loc[:origine - DELAI_MOIS].index.max()),
    )


def experimenter_delai(racine, date_depart=None):
    destination = racine / "outputs" / "recherche_pharma"
    developpement = pd.read_csv(destination / "developpement.csv", index_col="mois")
    developpement.index = pd.PeriodIndex(developpement.index, freq="M")
    poids = poids_du_melange(developpement)
    verification = pd.read_csv(destination / "verification.csv", index_col="mois")
    verification.index = pd.PeriodIndex(verification.index, freq="M")
    if verification.index.min() != DEBUT_VERIFICATION:
        raise ValueError("Les dates du benchmark de vérification ont changé.")
    verification["melange"] = np.rint(verification[TIMESTRAL] * poids
                                      + verification[SAISONNIER] * (1 - poids))
    comparables = [TIMESTRAL, SAISONNIER, "melange"]
    table, jours, _, _ = charger(racine)
    if not table.sum(axis=1).reindex(verification.index).eq(verification.observe).all():
        raise ValueError("Les volumes Medic’AM et le rapport précédent diffèrent.")
    moteur = MistralTimestral()
    for delai in range(DELAI_MOIS - 1, -1, -1):
        modele = f"hypothese_delai_{delai}"
        print(f"Test contrefactuel avec délai de {delai} mois", flush=True)
        verification[modele] = [
            timestral(moteur, table, jours, cible, fenetre=60, mode="formes",
                      calendrier_ouvre=False, delai_mois=delai)
            for cible in verification.index
        ]
        verification[f"dernier_mois_{modele}"] = (verification.index - 1 - delai).astype(str)
        comparables.append(modele)
    resultats = scores(verification, comparables)
    resultats["scenario"] = np.where(
        resultats.modele.str.startswith("hypothese"), "volumes_plus_recents_simules", "delai_courant"
    )
    resultats["poids_timestral_melange"] = poids
    verification.to_csv(destination / "verification_delai.csv", float_format="%.4f")
    resultats.to_csv(destination / "scores_delai.csv", index=False, float_format="%.4f")
    print(resultats.to_string(index=False))
    if date_depart is not None:
        fichier = destination / f"prevision_prospective_{date_depart.isoformat()}.csv"
        if fichier.exists():
            print(f"Prévision déjà figée : {fichier}")
            return
        futur = prevision_prospective(table, jours, moteur, date_depart, poids)
        futur.to_csv(fichier, index=False, mode="x")
        print(futur.to_string(index=False))
