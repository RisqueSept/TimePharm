"""Prévision mensuelle des boîtes remboursées par forme buvable."""

import pandas as pd
import numpy as np

from .inference_timestral import cycle


HORIZONS_VOLUMES = (1, 2, 3)
DELAI_MOIS = 2


def _cutoff_pour(origine, delais):
    """Mois de coupure pour une origine (``pd.Period`` freq M).

    ``delais`` vaut ``None`` (délai fixe ``DELAI_MOIS``, comportement
    historique), une fonction ``mois -> coupure`` (ex. ``cutoff_pour`` de
    `src/delai_medicam.py`), ou un dict/Series ``mois -> coupure``. Un mois
    absent du dict retombe sur le délai fixe ``DELAI_MOIS``.
    """
    if delais is None:
        return origine - DELAI_MOIS
    if callable(delais):
        return pd.Period(delais(origine), freq="M")
    if not hasattr(delais, "__getitem__"):
        raise ValueError("delais doit être None, une fonction mois -> coupure, ou un dict/Series mois -> coupure.")
    for cle in (origine, str(origine)):
        if cle in delais:
            return pd.Period(delais[cle], freq="M")
    return origine - DELAI_MOIS


def estimer(serie, origine, horizon, delais=None):
    """Référence saisonnière ajustée à ``horizon`` mois de ``origine``.

    ``delais`` : calendrier de disponibilité optionnel (voir `_cutoff_pour`) ;
    sans lui, la coupure est ``origine - DELAI_MOIS`` comme avant.
    """
    cible = origine + horizon
    reference = serie.get(cible - 12)
    connus = serie.loc[:_cutoff_pour(origine, delais)]
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


def previsions_timestral(volumes, lignes, moteur, delais=None):
    """Quantiles Timestral par origine ; ``delais`` fixe le mois de coupure (voir `_cutoff_pour`)."""
    table = volumes.pivot(index="mois", columns="produit", values="boites")
    table.index = table.index.to_period("M")
    resultats = []
    for origine, groupe in lignes.groupby("origine", sort=True):
        cutoff = _cutoff_pour(origine.to_period("M"), delais)
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


def prevoir_volumes(volumes, date_depart, moteur, delais=None):
    """Backtest et projection des boîtes ; ``delais`` (fonction ou dict mois -> coupure,
    ex. `src/delai_medicam.delais_reel`) remplace le délai fixe ``DELAI_MOIS``.
    Sans ``delais``, le comportement est identique à l'historique."""
    if volumes.duplicated(["produit", "mois"]).any() or not volumes.boites.ge(0).all():
        raise ValueError("Vérifier les mois et les volumes Medic’AM.")
    depart = pd.Period(date_depart, freq="M")
    historique = []
    futur = []
    for produit, groupe in volumes.groupby("produit"):
        serie = groupe.set_index(groupe.mois.dt.to_period("M")).boites.sort_index()
        for horizon in HORIZONS_VOLUMES:
            for cible, boites in serie.items():
                estimation = estimer(serie, cible - horizon, horizon, delais)
                if estimation is not None:
                    historique.append(dict(estimation, produit=produit, observe=int(boites)))
            estimation = estimer(serie, depart, horizon, delais)
            if estimation is not None:
                futur.append(dict(estimation, produit=produit))
    test = pd.DataFrame(historique)
    projection = pd.DataFrame(futur)
    predictions = previsions_timestral(volumes, pd.concat([test, projection], ignore_index=True), moteur, delais)
    return predictions.iloc[:len(test)].assign(observe=lambda df: df.observe.astype(int)), (
        predictions.iloc[len(test):].drop(columns="observe")
    )
