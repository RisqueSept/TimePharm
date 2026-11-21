"""Rejeu conservateur des prévisions avec les dates des éditions Medic'AM archivées."""

import numpy as np
import pandas as pd

from .inference_timestral import MistralTimestral
from .modele_volumes_pharma import DELAI_MOIS, estimer
from .recherche_delai_pharma import poids_du_melange
from .recherche_signaux_pharma import lire_benchmark
from .recherche_volumes_pharma import DEBUT_VERIFICATION, charger, scores, timestral


def dates_documentees(racine):
    recherche = racine / "data" / "research" / "pharma"
    production = racine / "data" / "raw" / "pharma"
    dates = pd.read_csv(recherche / "medicam_dates_archives.csv",
                        parse_dates=["date_creation", "date_mise_a_jour"])
    dates["edition_documentee"] = dates[["date_creation", "date_mise_a_jour"]].max(axis=1)
    sources = pd.concat([
        pd.read_csv(fichier, usecols=["mois", "source"])
        for fichier in (recherche / "medicam_2012_2020.csv", production / "medicam_volumes.csv")
    ]).drop_duplicates()
    par_mois = sources.merge(dates[["source", "edition_documentee"]], on="source", validate="many_to_one")
    dates = par_mois.assign(mois=pd.PeriodIndex(par_mois.mois, freq="M"))
    if len(sources) != len(dates) or dates.edition_documentee.isna().any():
        raise ValueError("Date d’édition manquante dans les archives Medic’AM.")
    return dates.groupby("mois").edition_documentee.max().sort_index()


def mois_documentes(table, editions, date_origine):
    disponibles = editions.reindex(table.index).lt(date_origine).to_numpy()
    # S'arrête au premier mois dont l'édition actuelle n'est pas encore attestée.
    return table.iloc[:int(np.cumprod(disponibles).sum())]


def dates_avec_prescripteurs(racine, table, editions):
    fichier = racine / "data" / "research" / "pharma" / "medicam_prescripteurs_2023_2026.csv"
    autre = pd.read_csv(fichier, parse_dates=["edition"])
    autre["mois"] = pd.PeriodIndex(autre.mois, freq="M")
    valeurs = autre.pivot(index="mois", columns="produit", values="boites")
    # Utilise une date plus ancienne uniquement quand les six volumes correspondent exactement.
    concordants = valeurs.eq(table.reindex(index=valeurs.index)).all(axis=1)
    dates = autre.groupby("mois").edition.max().loc[concordants]
    return pd.concat([editions, dates], axis=1).min(axis=1).sort_index()


def predictions_aux_dates(table, jours, publications, verification, poids, moteur, jours_origine=(1, 21)):
    lignes = []
    for cible, ligne in verification.iterrows():
        origine = pd.Period(ligne.origine, freq="M")
        for jour in jours_origine:
            date_origine = origine.start_time + pd.Timedelta(days=jour - 1)
            historique = mois_documentes(table, publications, date_origine)
            dernier = historique.index[-1]
            # Garde le même plafond que le benchmark pour isoler l'effet du retard des archives.
            historique = historique.loc[:origine - DELAI_MOIS]
            prediction = timestral(moteur, historique, jours, cible, fenetre=60,
                                   mode="formes", calendrier_ouvre=False)
            estimations = [estimer(historique[forme], origine, 1) for forme in historique]
            saison = (sum(estimation["ajuste"] for estimation in estimations)
                      if all(estimation is not None for estimation in estimations) else np.nan)
            lignes.append({
                "mois": str(cible), "date_origine": date_origine.date(),
                "jour_origine": jour, "borne_archive": str(dernier),
                "dernier_mois_utilise": str(historique.index[-1]),
                "dernier_mois_simule": str(ligne.dernier_mois_utilise),
                "mois_manquants": ligne.dernier_mois_utilise.ordinal - historique.index[-1].ordinal,
                "observe": ligne.observe, "timestral_archives": prediction,
                "saisonnier_archives": saison,
                "melange_archives": np.rint(prediction * poids + saison * (1 - poids)),
                "melange_delai_fixe": round(ligne.timestral_60_mois * poids
                                            + ligne.saisonnier_ajuste * (1 - poids)),
            })
    return pd.DataFrame(lignes)


def poids_archives(predictions):
    donnees = predictions.dropna(subset=["saisonnier_archives"])
    poids = np.linspace(0, 1, 5)
    estimations = (donnees.timestral_archives.to_numpy()[:, None] * poids
                   + donnees.saisonnier_archives.to_numpy()[:, None] * (1 - poids))
    erreurs = np.abs(estimations - donnees.observe.to_numpy()[:, None]).mean(axis=0)
    return float(poids[erreurs.argmin()])


def evaluer_poids_archives(table, jours, editions, developpement, verification, poids, moteur):
    premiere_origine = (DEBUT_VERIFICATION - 1).start_time + pd.Timedelta(days=20)
    connus = mois_documentes(table, editions, premiere_origine)
    developpement = developpement.loc["2020-01":connus.index[-1]]
    anciennes = predictions_aux_dates(table, jours, editions, developpement, poids, moteur, (21,))
    poids_retenu = poids_archives(anciennes)
    test = verification.query("archives == 'atc_et_prescripteurs' and jour_origine == 21").copy()
    test["melange_adapte_archives"] = np.rint(test.timestral_archives * poids_retenu
                                              + test.saisonnier_archives * (1 - poids_retenu))
    modeles = ["melange_archives", "melange_adapte_archives", "timestral_archives"]
    return pd.concat([
        scores(anciennes.dropna(subset=["saisonnier_archives"]), ["melange_archives", "timestral_archives"])
        .assign(periode="developpement_documente"),
        scores(test.dropna(subset=modeles), modeles).assign(periode="verification_reutilisee"),
    ]).assign(poids_timestral_retenu=poids_retenu,
              dernier_mois_developpement=str(developpement.index[-1]))


def experimenter_archives(racine):
    destination = racine / "outputs" / "recherche_pharma"
    developpement = lire_benchmark(destination, "developpement.csv")
    verification = lire_benchmark(destination, "verification.csv")
    poids = poids_du_melange(developpement)
    table, jours, _, _ = charger(racine)
    atc = dates_documentees(racine)
    editions = {"atc": atc, "atc_et_prescripteurs": dates_avec_prescripteurs(racine, table, atc)}
    moteur = MistralTimestral()
    predictions = pd.concat([
        predictions_aux_dates(table, jours, dates, verification, poids, moteur).assign(archives=nom)
        for nom, dates in editions.items()
    ], ignore_index=True)
    modeles = ["melange_archives", "timestral_archives", "saisonnier_archives", "melange_delai_fixe"]
    resultats = pd.concat([
        scores(groupe.dropna(subset=modeles), modeles).assign(jour_origine=jour, archives=archive)
        for (archive, jour), groupe in predictions.groupby(["archives", "jour_origine"])
    ])
    predictions.to_csv(destination / "predictions_dates_archives.csv", index=False, float_format="%.4f")
    resultats.to_csv(destination / "scores_dates_archives.csv", index=False, float_format="%.4f")
    print(predictions.groupby(["archives", "jour_origine"]).mois_manquants.agg(["mean", "max"]))
    print(resultats.to_string(index=False))
    poids_testes = evaluer_poids_archives(
        table, jours, editions["atc_et_prescripteurs"], developpement, predictions, poids, moteur
    )
    poids_testes.to_csv(destination / "scores_poids_archives.csv", index=False, float_format="%.4f")
    print(poids_testes.to_string(index=False))
