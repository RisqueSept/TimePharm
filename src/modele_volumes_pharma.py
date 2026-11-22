"""Prévision mensuelle des boîtes remboursées par forme buvable."""

import pandas as pd
import numpy as np

from .inference_timestral import cycle


HORIZONS_VOLUMES = (1, 2, 3)
DELAI_MOIS = 2


def estimer(serie, origine, horizon):
    cible = origine + horizon
    reference = serie.get(cible - 12)
    connus = serie.loc[:origine - DELAI_MOIS]
    ratios = [
        valeur / serie[mois - 12]
        for mois, valeur in connus.tail(6).items()
        if mois - 12 in connus.index and serie[mois - 12] > 0
    ]
    if reference is None or len(ratios) < 3:
        return None
    facteur = pd.Series(ratios).median()
    return {
        "origine": origine.to_timestamp(),
        "mois": cible.to_timestamp(),
        "dernier_mois_utilise": connus.index[-1].to_timestamp(),
        "horizon": horizon,
        "saisonnier": int(reference),
        "ajuste": round(reference * facteur),
    }


def previsions_timestral(volumes, lignes, moteur):
    table = volumes.pivot(index="mois", columns="produit", values="boites")
    table.index = table.index.to_period("M")
    resultats = []
    for origine, groupe in lignes.groupby("origine", sort=True):
        cutoff = origine.to_period("M") - DELAI_MOIS
        connus = table.loc[:cutoff].dropna(how="all")
        dernier = connus.index[-1]
        cible = groupe.mois.dt.to_period("M")
        dates = pd.period_range(connus.index[0], cible.max(), freq="M")
        historique = connus.reindex(dates[dates <= dernier])
        calendrier = np.repeat(cycle(dates.month)[None], len(historique.columns), axis=0)
        quantiles = moteur.quantiles(historique.to_numpy().T, int((cible.max() - dernier).n),
                                     calendrier=calendrier)
        positions = historique.columns.get_indexer(groupe.produit)
        horizons = cible.astype("int64") - dernier.ordinal - 1
        previsions = np.rint(quantiles[positions, 0, horizons]).astype(int)
        groupe = groupe.copy()
        groupe[["timestral_q10", "timestral", "timestral_q90"]] = previsions
        resultats.append(groupe)
    return pd.concat(resultats).sort_index()


def prevoir_volumes(volumes, date_depart, moteur):
    if volumes.duplicated(["produit", "mois"]).any() or not volumes.boites.ge(0).all():
        raise ValueError("Vérifier les mois et les volumes Medic’AM.")
    depart = pd.Period(date_depart, freq="M")
    historique = []
    futur = []
    for produit, groupe in volumes.groupby("produit"):
        serie = groupe.set_index(groupe.mois.dt.to_period("M")).boites.sort_index()
        for horizon in HORIZONS_VOLUMES:
            for cible, boites in serie.items():
                estimation = estimer(serie, cible - horizon, horizon)
                if estimation is not None:
                    historique.append(dict(estimation, produit=produit, observe=int(boites)))
            estimation = estimer(serie, depart, horizon)
            if estimation is not None:
                futur.append(dict(estimation, produit=produit))
    test = pd.DataFrame(historique)
    projection = pd.DataFrame(futur)
    predictions = previsions_timestral(volumes, pd.concat([test, projection], ignore_index=True), moteur)
    return predictions.iloc[:len(test)].assign(observe=lambda df: df.observe.astype(int)), (
        predictions.iloc[len(test):].drop(columns="observe")
    )
