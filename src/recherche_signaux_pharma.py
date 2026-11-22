"""Correction des prévisions mensuelles à partir des indicateurs pédiatriques Odissé."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .recherche_delai_pharma import SAISONNIER, TIMESTRAL, poids_du_melange
from .recherche_volumes_pharma import DEBUT_VERIFICATION, FIN_DEV, charger, scores


DUREE_DISPONIBILITE = pd.Timedelta(days=17)
DEBUT_COMPARAISON = pd.Period("2023-12", freq="M")
FENETRE = pd.Timedelta(days=28)
MIN_SEMAINES = 3


def signal_avant_origine(source, origine):
    """Compare les quatre dernières semaines disponibles à celles de l'année précédente."""
    courant = source.loc[
        source.index.to_series().between(origine - FENETRE, origine, inclusive="left")
    ]
    repere = origine - pd.Timedelta(weeks=52)
    precedent = source.loc[
        source.index.to_series().between(repere - FENETRE, repere, inclusive="left")
    ]
    recent = courant.mean().where(courant.count().ge(MIN_SEMAINES))
    anterieur = precedent.mean().where(precedent.count().ge(MIN_SEMAINES))
    return np.log1p(recent) - np.log1p(anterieur)


def lire_odisse(fichier):
    donnees = pd.read_csv(fichier, parse_dates=["semaine"])
    if donnees.duplicated(["indicateur", "semaine"]).any():
        raise ValueError("Une série nationale et une tranche d’âge par indicateur sont attendues.")
    donnees["disponible_estime"] = donnees.semaine + DUREE_DISPONIBILITE
    tableau = donnees.pivot(index="disponible_estime", columns="indicateur", values=["urgences", "sos"])
    tableau.columns = [f"{type_mesure}_{indicateur}" for type_mesure, indicateur in tableau.columns]
    return tableau


def corriger_melange(lignes, indices, colonnes):
    predictions = []
    for cible in indices:
        ligne = lignes.loc[[cible]]
        connu = ligne.dernier_mois_utilise.iloc[0]
        historique = lignes.loc[:connu].dropna(subset=colonnes)
        if len(historique) < 18 or ligne[colonnes].isna().any(axis=None):
            predictions.append(np.nan)
            continue
        ecarts = np.log(historique.observe / historique.melange)
        if colonnes:
            modele = make_pipeline(StandardScaler(), Ridge(alpha=24))
            modele.fit(historique[colonnes], ecarts)
            correction = modele.predict(ligne[colonnes])[0]
        else:
            correction = ecarts.mean()
        predictions.append(round(ligne.melange.iloc[0] * np.exp(correction)))
    return pd.Series(predictions, index=indices, dtype=float)


def lire_benchmark(destination, nom):
    lignes = pd.read_csv(destination / nom, index_col="mois")
    lignes.index = pd.PeriodIndex(lignes.index, freq="M")
    lignes["dernier_mois_utilise"] = pd.PeriodIndex(lignes.dernier_mois_utilise, freq="M")
    return lignes


def profil_saisonnier(table, cible, connu, annees, mois_recents):
    historique = table.loc[:connu].tail(12 * annees)
    profil = historique.groupby(historique.index.month).median()
    recent = historique.tail(mois_recents)
    niveau = np.median(recent.to_numpy() / profil.loc[recent.index.month].to_numpy(), axis=0)
    return float(np.sum(profil.loc[cible.month].to_numpy() * niveau))


def experimenter_profils(racine, lignes, destination):
    table, _, _, _ = charger(racine)
    predictions = lignes[["origine", "dernier_mois_utilise", "observe", "melange"]].copy()
    for annees in (3, 5):
        for mois_recents in (3, 6):
            nom = f"profil_{annees}ans_{mois_recents}mois"
            predictions[nom] = [
                profil_saisonnier(table, cible, connu, annees, mois_recents)
                for cible, connu in predictions.dernier_mois_utilise.items()
            ]
            predictions["avec_" + nom] = (predictions[nom] + predictions.melange) / 2
    stats_dev = scores(predictions.loc[:FIN_DEV], predictions.columns[3:])
    candidat = stats_dev.modele.iloc[0]
    periode_test = predictions.loc[DEBUT_VERIFICATION:]
    comparaison = list(dict.fromkeys(["melange", candidat]))
    stats_test = scores(periode_test, comparaison)
    resultats = pd.concat([
        stats_dev.assign(periode="developpement"),
        stats_test.assign(periode="verification_reutilisee"),
    ]).assign(candidat_retenu=candidat)
    resultats.to_csv(destination / "scores_profils_saisonniers.csv", index=False, float_format="%.4f")
    periode_test[["origine", "dernier_mois_utilise", "observe", *comparaison]].to_csv(
        destination / "predictions_profils_saisonniers.csv", float_format="%.4f"
    )
    print(resultats.to_string(index=False))


def experimenter_signaux(racine, jour=1):
    destination = racine / "outputs" / "recherche_pharma"
    developpement = lire_benchmark(destination, "developpement.csv")
    test = lire_benchmark(destination, "verification.csv")
    if developpement.index.max() != FIN_DEV or test.index.min() != DEBUT_VERIFICATION:
        raise ValueError("Régénérer le benchmark de référence avec le découpage temporel prévu.")
    signal = lire_odisse(racine / "data" / "research" / "pharma" / "odisse_respiratoire_pediatrie.csv")
    poids = poids_du_melange(developpement)
    lignes = pd.concat([developpement, test]).sort_index()
    lignes["melange"] = np.rint(lignes[TIMESTRAL] * poids + lignes[SAISONNIER] * (1 - poids))
    if jour == 1:
        experimenter_profils(racine, lignes, destination)
    dates = lignes.index[lignes.index >= DEBUT_COMPARAISON]
    features = pd.DataFrame(
        [signal_avant_origine(signal, origine.start_time + pd.Timedelta(days=jour - 1))
         for origine in pd.PeriodIndex(lignes.origine, freq="M")],
        index=lignes.index,
    )
    lignes = lignes.join(features)
    groupes = {
        "calibre_sans_signal": [],
        "calibre_urgences": features.filter(like="urgences_").columns.tolist(),
        "calibre_sos": features.filter(like="sos_").columns.tolist(),
    }
    for modele, colonnes in groupes.items():
        lignes[modele] = corriger_melange(lignes, dates, colonnes)
    modeles = ["melange", *groupes]
    comparable = lignes.loc[dates].dropna(subset=modeles)
    scores_dev = scores(comparable.loc[:FIN_DEV], modeles)
    nom_retenu = scores_dev.modele.iloc[0]
    verification = comparable.loc[DEBUT_VERIFICATION:]
    scores_test = scores(verification, list(dict.fromkeys(["melange", nom_retenu])))
    resultats = pd.concat([
        scores_dev.assign(periode="developpement"),
        scores_test.assign(periode="verification_reutilisee"),
    ]).assign(candidat_retenu=nom_retenu, jour_origine=jour)
    suffixe = "" if jour == 1 else f"_j{jour}"
    resultats.to_csv(destination / f"scores_signaux_odisse{suffixe}.csv",
                     index=False, float_format="%.4f")
    lignes["date_origine"] = pd.PeriodIndex(lignes.origine, freq="M").start_time + pd.Timedelta(days=jour - 1)
    lignes.loc[dates, ["date_origine", "dernier_mois_utilise", "observe", *modeles, *features.columns]].to_csv(
        destination / f"predictions_signaux_odisse{suffixe}.csv", float_format="%.4f"
    )
    print(resultats.to_string(index=False))
