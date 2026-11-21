"""Projection rétrospective de l'occupation hospitalière avec SARS-CoV-2."""

import os

import numpy as np
import pandas as pd
from table_mensuelle import PROCESSED, REGIONS, charger_prevalences
from inference_timestral import cycle

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
HORIZON = 24
HORIZONS_TEST = (1, 3, 6, 12)
MIN_MOIS_BACKTEST = 24
CIBLE = "lits_covid_occupes"


def timestral_mensuel(moteur, historique, dates):
    calendrier = cycle(historique.index.append(dates).month)
    calendrier = np.repeat(calendrier[None, :, :], historique.shape[1], axis=0)
    return moteur.quantiles(historique.to_numpy().T, len(dates), calendrier=calendrier)[:, 0]


def evaluer(historique, moteur):
    blocs = []
    for i in range(MIN_MOIS_BACKTEST, len(historique)):
        train = historique.iloc[:i]
        test = historique.iloc[i:i + max(HORIZONS_TEST)]
        horizons = np.arange(1, len(test) + 1)
        cibles = test.stack().rename("lits_reel").to_frame()
        cibles["origine"] = train.index[-1]
        cibles["horizon_mois"] = np.repeat(horizons, test.shape[1])
        cibles["pred_Persistance"] = np.tile(train.iloc[-1].to_numpy(), len(test))
        cibles["pred_Saisonnier"] = train.iloc[-12:].head(len(test)).to_numpy().ravel()
        quantiles = timestral_mensuel(moteur, train, test.index)
        cibles[["timestral_q10", "pred_Timestral", "timestral_q90"]] = (
            quantiles.transpose(1, 0, 2).reshape(-1, 3)
        )
        blocs.append(cibles.reset_index())
    if not blocs:
        raise ValueError("Historique insuffisant pour le backtest mensuel")
    bt = pd.concat(blocs, ignore_index=True)
    bt = bt[bt.horizon_mois.isin(HORIZONS_TEST)].copy()
    candidats = bt.filter(like="pred_").columns
    erreurs = bt[candidats].sub(bt.lits_reel, axis=0)
    court_terme = bt.horizon_mois.le(6)
    rmse = erreurs[court_terme].pow(2).mean().pow(0.5)
    bt["modele"] = "Timestral"
    bt["region_nom"] = bt.reg.map(REGIONS)
    bt.to_csv(os.path.join(OUT, "backtest_mensuel.csv"), index=False)
    print("[backtest mensuel] Timestral et références, horizons 1, 3 et 6 mois")
    print(rmse.round(1).to_string())
    par_horizon = erreurs.pow(2).groupby(bt.horizon_mois).mean().pow(0.5)
    par_horizon["origines"] = bt.groupby("horizon_mois").origine.nunique()
    print(par_horizon.round(1).to_string())


def exporter_previsions(historique, moteur):
    mois_futurs = pd.date_range(historique.index[-1] + pd.offsets.MonthBegin(),
                                periods=HORIZON, freq="MS", name="mois")
    quantiles = timestral_mensuel(moteur, historique, mois_futurs)
    index_futur = pd.MultiIndex.from_product([mois_futurs, historique.columns], names=["mois", "reg"])
    prev = pd.DataFrame(quantiles.transpose(1, 0, 2).reshape(-1, 3), index=index_futur,
                        columns=["scenario_bas", "lits_pred", "scenario_haut"])
    full = pd.concat([historique.stack().rename("lits_pred").to_frame(), prev]).reset_index()
    full = full.sort_values(["reg", "mois"])
    full["est_prevision"] = full.mois.isin(mois_futurs).astype(int)
    full["region_nom"] = full.reg.map(REGIONS)
    full["modele"] = np.where(full.est_prevision.eq(1), "Timestral", "")
    full.to_csv(os.path.join(OUT, "prevision_mensuelle_regions.csv"), index=False)
    france = (full.groupby(["mois", "est_prevision", "modele"], as_index=False)
              .lits_pred.sum())
    france.to_csv(os.path.join(OUT, "prevision_mensuelle_france.csv"), index=False)
    return full, france


def projeter_prevalences(moteur):
    prev = charger_prevalences()
    lignes = []
    series = []
    for patho in prev.filter(like="prev_"):
        for reg, g in prev.groupby("reg"):
            g = g.loc[g.annee.le(2024), ["annee", patho]].dropna()
            reference = g.loc[g.annee.eq(2024), patho]
            if len(g) < 2 or reference.empty:
                continue
            series.append(g.set_index("annee")[patho].reindex(range(2015, 2025)).to_numpy())
            lignes.append({"reg": reg, "region_nom": REGIONS[reg], "patho": patho,
                           "prev_2024": reference.item()})
    proj = pd.DataFrame(lignes)
    quantiles = moteur.quantiles(series, 6, bornes=(0, 100))[:, 0, -1, :]
    proj[["timestral_q10", "prev_2030", "timestral_q90"]] = quantiles
    proj["delta_pts"] = proj.prev_2030 - proj.prev_2024
    proj["modele"] = "Timestral"
    proj.to_csv(os.path.join(OUT, "prevalence_projections_2030.csv"), index=False)
    return proj


def run(moteur):
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(PROCESSED, "table_mensuelle_regions.csv"), parse_dates=["mois"])
    historique = df.pivot(index="mois", columns="reg", values=CIBLE)
    historique = historique.asfreq("MS")
    if set(historique.columns) != set(REGIONS) or not np.isfinite(historique.to_numpy()).all():
        raise ValueError("La série SPF doit être complète pour chaque couple mois-région")
    evaluer(historique, moteur)
    full, france = exporter_previsions(historique, moteur)
    return full, france, projeter_prevalences(moteur)


if __name__ == "__main__":
    from inference_timestral import MistralTimestral

    run(MistralTimestral())
