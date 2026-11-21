"""Construction de la table mensuelle régionale (2020-03 -> 2023-06).

Cible : lits_covid_occupes = nombre moyen de lits occupés par des patients COVID
         dans la région sur le mois (tx_prev_hosp * population / 100 000, moyenné).
Features structurelles croisées :
  - population régionale INSEE (au 1er janvier) + part des 60+ et 75+
  - prévalences régionales annuelles CNAM (Ameli) : diabète, cancers,
    maladies cardio-neurovasculaires, psychiatriques, respiratoires, neurologiques
  - calendrier : mois, sin/cos (saisonnalité), indicateur hiver

Sortie : data/processed/table_mensuelle_regions.csv
"""

import io
import os
import zipfile

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(BASE, "data", "raw", "mensuel")
PROCESSED = os.path.join(BASE, "data", "processed")

REGIONS = {
    1: "Guadeloupe", 2: "Martinique", 3: "Guyane", 4: "La Réunion", 6: "Mayotte",
    11: "Île-de-France", 24: "Centre-Val de Loire", 27: "Bourgogne-Franche-Comté",
    28: "Normandie", 32: "Hauts-de-France", 44: "Grand Est", 52: "Pays de la Loire",
    53: "Bretagne", 75: "Nouvelle-Aquitaine", 76: "Occitanie",
    84: "Auvergne-Rhône-Alpes", 93: "Provence-Alpes-Côte d'Azur", 94: "Corse",
}

PATHOLOGIES = {  # codes agrégés Ameli
    "DIA_CAT_CAT": "prev_diabete",
    "CAN_CAT_CAT": "prev_cancer",
    "MCV_CAT_CAT": "prev_cardio",
    "PSY_CAT_CAT": "prev_psy",
    "RES_CAT_EXC": "prev_resp",
    "NEU_CAT_CAT": "prev_neuro",
}


def charger_covid() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW, "covid_hosp_txad_reg.csv"), sep=";")
    df = df[df.PourAvec == 0].copy()  # tous motifs d'hospitalisation avec infection SARS-CoV-2
    df["reg"] = df["reg"].astype(int)
    df["jour"] = pd.to_datetime(df["jour"])
    df["tx_prev_hosp"] = pd.to_numeric(df["tx_prev_hosp"], errors="coerce")
    df["tx_indic_7J_hosp"] = pd.to_numeric(df["tx_indic_7J_hosp"], errors="coerce")
    df = df[df.reg.isin(REGIONS)]
    if df.duplicated(["reg", "jour"]).any():
        raise ValueError("SPF : doublons région-jour")
    return df[["reg", "jour", "tx_prev_hosp", "tx_indic_7J_hosp"]]


def charger_population() -> pd.DataFrame:
    """Population régionale INSEE au 1er janvier + parts 60+ / 75+ (niveau REG)."""
    z = zipfile.ZipFile(os.path.join(RAW, "insee_estimation_population.zip"))
    nom = [n for n in z.namelist() if n.endswith("_data.csv")][0]
    raw = pd.read_csv(io.BytesIO(z.read(nom)), sep=";", dtype=str)
    raw = raw[raw.GEO_OBJECT == "REG"]
    raw["reg"] = raw.GEO.astype(int)
    raw["annee"] = raw.TIME_PERIOD.astype(int)
    raw["val"] = pd.to_numeric(raw.OBS_VALUE, errors="coerce")

    def serie(ep, age):
        s = raw[(raw.EP_MEASURE == ep) & (raw.SEX == "_T") & (raw.AGE == age)]
        return s.set_index(["reg", "annee"])["val"]

    pop = serie("POP_JAN_1ST", "_T")
    p75 = serie("PT_IN_POP", "Y_GE75")   # part des 75+ dans la pop, en %
    p60 = serie("PT_IN_POP", "Y_GE60")   # part des 60+ dans la pop, en %
    out = pd.DataFrame({"population": pop, "part_75plus": p75, "part_60plus": p60}).reset_index()
    return out[out.reg.isin(REGIONS)]


def charger_prevalences() -> pd.DataFrame:
    """Prévalences régionales annuelles (Ameli) pour les pathologies chroniques clés."""
    df = pd.read_csv(os.path.join(RAW, "ameli_pathologies_regions.csv"), sep=";", encoding="utf-8-sig")
    df["region"] = df["region"].astype(int)
    df = df[df.region.isin(REGIONS) & df.top.isin(PATHOLOGIES)].copy()
    df["prev"] = pd.to_numeric(df["prev"], errors="coerce")
    df["Ntop"] = pd.to_numeric(df["Ntop"], errors="coerce")
    df["Npop"] = pd.to_numeric(df["Npop"], errors="coerce")
    df["prev"] = df["prev"].fillna(100 * df.Ntop / df.Npop.where(df.Npop > 0))
    df["col"] = df.top.map(PATHOLOGIES)
    return (df.pivot(index=["region", "annee"], columns="col", values="prev")
            .reindex(columns=PATHOLOGIES.values()).reset_index().rename(columns={"region": "reg"}))


def _interpole_annuelle(df, cols, mens):
    """Interpole linéairement les valeurs annuelles de `df` (index: reg, annee)
    aux positions mensuelles de `mens` (colonnes reg et t)."""
    out = pd.DataFrame({"reg": mens.reg.values, "t": mens["t"].values})
    for reg, g in df.groupby("reg"):
        g = g.set_index("annee")[cols].sort_index()
        x = g.index.values.astype(float)
        mask = (out.reg == reg).values
        tsub = out.loc[mask, "t"].values.astype(float)
        for c in cols:
            y = g[c].values.astype(float)
            valid = np.isfinite(y)
            out.loc[mask, c] = np.interp(tsub, x[valid], y[valid]) if valid.any() else np.nan
    return out


def construire_table() -> pd.DataFrame:
    os.makedirs(PROCESSED, exist_ok=True)
    covid = charger_covid()
    pop = charger_population()
    prev = charger_prevalences()

    covid["annee"] = covid.jour.dt.year
    covid = covid.merge(pop, on=["reg", "annee"], how="left", validate="many_to_one")
    if covid.population.isna().any():
        raise ValueError("Population INSEE manquante pour un couple région-année SPF")
    # lits occupés COVID ce jour-là = taux de prévalence (p100k) x population / 100 000
    covid["lits_covid_jour"] = covid.tx_prev_hosp * covid.population / 100_000
    # admissions du jour ~ taux glissant 7J / 7
    covid["admissions_jour"] = covid.tx_indic_7J_hosp / 7 * covid.population / 100_000

    covid["mois"] = covid.jour.values.astype("datetime64[M]")
    mens = (covid.groupby(["reg", "mois"])
                 .agg(lits_covid_occupes=("lits_covid_jour", "mean"),
                      jours_observes=("lits_covid_jour", "count"),
                      admissions_covid=("admissions_jour", lambda s: s.sum(min_count=1)))
                 .reset_index())

    # features annuelles interpolées au mois
    mens["annee"] = mens.mois.dt.year
    mens["t"] = mens.mois.dt.year + (mens.mois.dt.month - 1) / 12.0
    pop_i = _interpole_annuelle(pop, ["population", "part_75plus", "part_60plus"], mens)
    prev_cols = [c for c in prev.columns if c.startswith("prev_")]
    prev_i = _interpole_annuelle(prev, prev_cols, mens)
    mens = mens.merge(pop_i, on=["reg", "t"], how="left").merge(prev_i, on=["reg", "t"], how="left")
    mens = mens.drop(columns=["annee"])

    # calendrier / saisonnalité
    m = mens.mois.dt.month
    mens["mois_num"] = m
    mens["sin_mois"] = np.sin(2 * np.pi * (m - 1) / 12)
    mens["cos_mois"] = np.cos(2 * np.pi * (m - 1) / 12)
    mens["hiver"] = m.isin([12, 1, 2, 3]).astype(int)
    mens["region_nom"] = mens.reg.map(REGIONS)
    mens = mens.sort_values(["reg", "mois"]).reset_index(drop=True)

    out = os.path.join(PROCESSED, "table_mensuelle_regions.csv")
    mens.to_csv(out, index=False)
    print(f"[table] -> {out} ({len(mens)} lignes, {mens.reg.nunique()} régions, "
          f"{mens.mois.min().date()} -> {mens.mois.max().date()})")
    print(mens.groupby("region_nom").lits_covid_occupes.agg(["mean", "max"]).round(0).to_string())
    return mens


if __name__ == "__main__":
    construire_table()
