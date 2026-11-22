"""Correction du modèle combiné par les écarts de calendrier."""

import numpy as np
import pandas as pd

from .recherche_delai_pharma import SAISONNIER, TIMESTRAL, poids_du_melange
from .recherche_signaux_pharma import corriger_melange, lire_benchmark
from .recherche_volumes_pharma import DEBUT_VERIFICATION, FIN_DEV, scores


def calendrier_mensuel(vacances, feries, dates):
    jours = pd.date_range(dates.min().start_time, dates.max().end_time.normalize(), freq="D")
    ouverts = (jours.dayofweek < 5) & ~jours.isin(feries)
    ecole = []
    for _, zone in vacances.groupby("zones"):
        absents = np.zeros(len(jours), dtype=bool)
        for debut, fin in zone[["start_date", "end_date"]].itertuples(index=False, name=None):
            absents |= (jours >= debut) & (jours < fin)
        ecole.append(ouverts & ~absents)
    return pd.DataFrame({
        "jours_ouvres": ouverts.astype(int), "jours_ecole": np.mean(ecole, axis=0),
    }, index=jours.to_period("M")).groupby(level=0).sum()


def experimenter_calendrier(racine):
    destination = racine / "outputs" / "recherche_pharma"
    developpement = lire_benchmark(destination, "developpement.csv")
    verification = lire_benchmark(destination, "verification.csv")
    historique = pd.concat([developpement, verification]).sort_index()
    historique = historique.loc["2018-01":].copy()
    poids = poids_du_melange(developpement)
    historique["melange"] = np.rint(historique[TIMESTRAL] * poids + historique[SAISONNIER] * (1 - poids))
    donnees = racine / "data" / "research" / "pharma"
    vacances = pd.read_csv(donnees / "vacances_scolaires_zones.csv",
                           parse_dates=["start_date", "end_date"])
    feries = pd.to_datetime(pd.read_csv(donnees / "jours_feries_metropole.csv").date)
    calendrier = calendrier_mensuel(vacances, feries, historique.index)
    historique = historique.join(calendrier)
    for champ in calendrier:
        # Écart à la valeur typique du même mois, calculée sur les années calendaires précédentes.
        historique[champ + "_ecart"] = (
            calendrier[champ] - calendrier[champ].groupby(calendrier.index.month).transform(
                lambda s: s.shift().expanding(min_periods=1).median()
            )
        )
    historique["ratio_jours"] = np.rint(historique.melange * historique.jours_ouvres /
                                        (historique.jours_ouvres - historique.jours_ouvres_ecart))
    variantes = {
        "biais_sans_calendrier": [],
        "biais_jours_ouvres": ["jours_ouvres_ecart"],
        "biais_avec_ecole": ["jours_ouvres_ecart", "jours_ecole_ecart"],
    }
    for nom, colonnes in variantes.items():
        historique[nom] = corriger_melange(historique, historique.index, colonnes)
    modeles = ["melange", "ratio_jours", *variantes]
    comparaison = historique.dropna(subset=modeles)
    developpement = scores(comparaison.loc[:FIN_DEV], modeles)
    candidat = developpement.modele.iloc[0]
    test = scores(comparaison.loc[DEBUT_VERIFICATION:], list(dict.fromkeys(["melange", candidat])))
    resultats = pd.concat([
        developpement.assign(periode="developpement"),
        test.assign(periode="verification_reutilisee"),
    ]).assign(candidat_retenu=candidat)
    resultats.to_csv(destination / "scores_calendrier.csv", index=False, float_format="%.4f")
    historique[["origine", "dernier_mois_utilise", "observe", *calendrier.columns, *modeles]].to_csv(
        destination / "predictions_calendrier.csv", float_format="%.4f"
    )
    print(resultats.to_string(index=False))
