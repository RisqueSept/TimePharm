"""Backtest chronologique sur les relevés de couverture de l'ANSM."""

from datetime import date

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .inference_timestral import probabilite_depassement


PRODUITS = {
    "amox250": "Amoxicilline 250 mg/5 mL",
    "amox500": "Amoxicilline 500 mg/5 mL",
    "amoxclav": "Amoxicilline / clavulanate 100/12,5 mg",
    "azithro": "Azithromycine 40 mg/mL",
    "clarithro25": "Clarithromycine 25 mg/mL",
    "clarithro50": "Clarithromycine 50 mg/mL",
}
HORIZONS = (14, 28)
BASE = ["cycle_sin", "cycle_cos", "grossistes", "age_releve", *PRODUITS]
CONTEXTE = [*BASE, "industrie", "officines", "autres_insuffisants", "autres_suivis",
            "ira_log", "ira_evolution"]
DEGRADATION = ["grossistes", "industrie", "officines", "variation_grossistes",
               "part_insuffisante_42j", "ira_log", "ira_evolution"]


def charger_donnees(dossier):
    stocks = pd.read_csv(dossier / "ansm_couverture.csv", parse_dates=["observation", "disponible"])
    if stocks.duplicated(["observation", "produit"]).any():
        raise ValueError("Plusieurs relevés pour une même date et un même produit.")
    if (stocks.disponible < stocks.observation).any() or not stocks.produit.isin(PRODUITS).all():
        raise ValueError("Date de disponibilité ou produit invalide dans les relevés.")
    stocks = stocks.sort_values(["observation", "produit"], ignore_index=True)
    annee = stocks.observation.dt.year - (stocks.observation.dt.month < 7)
    stocks["saison"] = annee.astype(str) + "–" + (annee + 1).astype(str).str[-2:]

    ira = pd.read_csv(dossier / "sentinelles_ira.csv", comment="#", usecols=["week", "inc100"])
    ira["semaine"] = pd.to_datetime([
        date.fromisocalendar(*divmod(int(week), 100), 1) for week in ira.week
    ])
    ira = ira.dropna(subset=["inc100"]).sort_values("semaine", ignore_index=True)
    ira["disponible"] = ira.semaine + pd.Timedelta(days=10)
    ira["ira_log"] = np.log1p(ira.inc100)
    ira["ira_evolution"] = (ira.ira_log - ira.ira_log.shift()).where(
        ira.semaine.diff().eq(pd.Timedelta(days=7))
    )
    volumes = pd.read_csv(dossier / "medicam_volumes.csv", parse_dates=["mois"])
    return stocks, ira, volumes


def points_evaluation(stocks, ira, horizon):
    """Associe les informations disponibles à un état de stock observé plus tard."""
    lignes = []
    minimum = pd.Timedelta(days=horizon - 3)
    maximum = pd.Timedelta(days=horizon + 3)
    for origine in sorted(stocks.disponible.unique()):
        publies = stocks.loc[stocks.disponible.le(origine)].sort_values("observation")
        connus = publies.drop_duplicates("produit", keep="last")
        connus = connus.loc[connus.observation.ge(origine - pd.Timedelta(days=16))]
        recents = publies.loc[publies.observation.ge(origine - pd.Timedelta(days=42))]
        semaine = ira.loc[ira.disponible.le(origine)].iloc[-1]
        for releve in connus.itertuples(index=False):
            age = (origine - releve.observation).days
            if releve.disponible != origine or pd.isna(releve.grossistes):
                continue
            candidats = stocks.loc[
                stocks.produit.eq(releve.produit) & stocks.saison.eq(releve.saison)
                & stocks.observation.between(origine + minimum, origine + maximum)
                & stocks.grossistes.notna()
            ]
            if candidats.empty:
                continue
            ecarts = (candidats.observation - origine - pd.Timedelta(days=horizon)).abs()
            cible = candidats.loc[ecarts.idxmin()]
            autres = connus.loc[connus.produit.ne(releve.produit), "grossistes"]
            historique = recents.loc[
                recents.produit.eq(releve.produit)
                & recents.observation.lt(releve.observation), "grossistes"
            ].dropna()
            angle = 2 * np.pi * origine.dayofyear / 365.25
            lignes.append({
                "origine": origine, "observation_initiale": releve.observation,
                "cible": cible.observation, "cible_disponible": cible.disponible,
                "saison": releve.saison, "produit": releve.produit,
                "horizon": horizon, "delai_reel": (cible.observation - origine).days,
                "observé": int(cible.grossistes == 0), "source_initiale": releve.source,
                "source_cible": cible.source, "cycle_sin": np.sin(angle),
                "cycle_cos": np.cos(angle), "age_releve": age,
                "grossistes": releve.grossistes, "industrie": releve.industrie,
                "officines": releve.officines, "autres_insuffisants": autres.eq(0).sum(),
                "variation_grossistes": releve.grossistes - historique.iloc[-1] if len(historique) else np.nan,
                "part_insuffisante_42j": historique.eq(0).mean(),
                "autres_suivis": autres.notna().sum(), "ira_semaine": semaine.semaine,
                "ira_disponible_estimee": semaine.disponible, "ira_incidence": semaine.inc100,
                "ira_log": semaine.ira_log, "ira_evolution": semaine.ira_evolution,
                **{produit: int(produit == releve.produit) for produit in PRODUITS},
            })
    return pd.DataFrame(lignes)


def probabilites(apprentissage, test, colonnes):
    modele = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=0.3, max_iter=1000),
    )
    modele.fit(apprentissage[colonnes], apprentissage["observé"])
    return modele.predict_proba(test[colonnes])[:, 1]


def backtest(stocks, ira):
    blocs = []
    for horizon in HORIZONS:
        points = points_evaluation(stocks, ira, horizon)
        for origine, test in points.groupby("origine", sort=True):
            apprentissage = points.loc[points.cible_disponible.lt(origine)]
            # L'évaluation commence avec une saison antérieure et les deux états observés.
            if apprentissage["observé"].value_counts().reindex([0, 1], fill_value=0).min() < 8:
                continue
            if apprentissage.saison.min() >= test.saison.iloc[0]:
                continue
            test = test.assign(
                reference=probabilites(apprentissage, test, BASE),
                contexte=probabilites(apprentissage, test, CONTEXTE),
                n_apprentissage=len(apprentissage),
                derniere_cible_connue=apprentissage.cible_disponible.max(),
            )
            historique_degradations = apprentissage.loc[apprentissage.grossistes.gt(0)]
            risque_degradation = test.grossistes.gt(0)
            test["degradation"] = np.nan
            if (risque_degradation.any()
                and historique_degradations["observé"].value_counts().reindex([0, 1], fill_value=0).min() >= 2):
                test.loc[risque_degradation, "degradation"] = probabilites(
                    historique_degradations, test.loc[risque_degradation], DEGRADATION
                )
            blocs.append(test)
    if not blocs:
        raise ValueError("Historique insuffisant pour un backtest chronologique.")
    return pd.concat(blocs, ignore_index=True)


def previsions_timestral(stocks, ira, backtest, moteur):
    """Prévoit l'indicateur de couverture insuffisante sur une grille hebdomadaire."""
    dates_releves = stocks.observation.dt.to_period("W-SUN")
    dates_ira = ira.semaine.dt.to_period("W-SUN")
    resultats = []
    for origine, lignes in backtest.groupby("origine", sort=True):
        publications = stocks.loc[stocks.disponible.le(origine)].assign(semaine=dates_releves)
        publications = publications.sort_values("observation").drop_duplicates(["semaine", "produit"], keep="last")
        dates = pd.period_range(dates_releves.min(), origine.to_period("W-SUN"), freq="W-SUN")
        grille = publications.set_index(["semaine", "produit"]).unstack("produit")
        produits = lignes.produit.unique()
        grille = grille.reindex(index=dates, columns=pd.MultiIndex.from_product(
            [["grossistes", "industrie", "officines"], produits]
        ))
        niveaux_stocks = grille["grossistes"].to_numpy().T
        cibles = np.where(np.isnan(niveaux_stocks), np.nan, niveaux_stocks == 0)
        incidence = (ira.loc[ira.disponible.le(origine)].assign(semaine=dates_ira)
                     .set_index("semaine").ira_log.reindex(dates).to_numpy())
        covariables = np.stack([
            grille["industrie"].to_numpy().T,
            grille["officines"].to_numpy().T,
            np.broadcast_to(incidence, cibles.shape),
        ], axis=1)
        horizon = lignes.cible.dt.to_period("W-SUN").astype("int64") - dates[-1].ordinal
        quantiles, probabilites_quantiles = moteur.prevoir(
            cibles, int(horizon.max()), passe=covariables, bornes=(0, 1)
        )
        indices = pd.Index(produits).get_indexer(lignes.produit)
        valeurs = [
            probabilite_depassement(prediction, probabilites_quantiles, 0.5)
            for prediction in quantiles[indices, 0, horizon - 1]
        ]
        resultats.append(lignes.assign(timestral=valeurs))
    return pd.concat(resultats).sort_index()
