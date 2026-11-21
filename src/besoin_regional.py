#!/usr/bin/env python3
"""Indicateur de besoin relatif par région (chantier #12).

Croise trois sources pour produire ``outputs/besoin_regional.csv`` :

- ``outputs/previsions_boites_pharma.csv`` : boîtes d'antibiotiques buvables
  prévues au niveau national, par forme (colonne ``timestral``, médiane du
  modèle retenu, à l'horizon 1 mois = prévision courante) ;
- ``outputs/prevalence_projections_2030.csv`` : prévalences régionales projetées
  à 2030 pour six pathologies chroniques (médiane Timestral) ;
- ``data/raw/mensuel/insee_estimation_population.zip`` : population régionale
  INSEE et part des 75 ans et plus (dernière année disponible, niveau REG).

Méthode (déterministe, documentée) :

1. les boîtes prévues par forme sont réparties entre régions au prorata de la
   population des 75 ans et plus (``population × part_75plus / 100``), proxy
   démographique retenu pour ce chantier ;
2. la prévalence pondérée d'une région est la moyenne de ses prévalences
   projetées à 2030, pondérée par la fréquence nationale moyenne 2024 de chaque
   pathologie (les pathologies les plus fréquentes pèsent davantage) ;
3. l'indice de besoin relatif brut vaut ``boîtes prévues par habitant ×
   prévalence pondérée`` ; il est normalisé entre 0 et 100 (min–max) et chaque
   région reçoit un rang unique (1 = besoin relatif le plus élevé).

Lancement : ``.venv-timestral/bin/python src/besoin_regional.py``
"""

import io
import zipfile
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
POP_ZIP = BASE / "data" / "raw" / "mensuel" / "insee_estimation_population.zip"

COLONNE_PREVISION = "timestral"  # médiane Timestral = prévision retenue du pipeline
HORIZON = 1                       # horizon 1 mois = prévision courante
PATHOLOGIES = ["prev_diabete", "prev_cancer", "prev_cardio", "prev_psy", "prev_resp", "prev_neuro"]

REGIONS = {
    1: "Guadeloupe", 2: "Martinique", 3: "Guyane", 4: "La Réunion", 6: "Mayotte",
    11: "Île-de-France", 24: "Centre-Val de Loire", 27: "Bourgogne-Franche-Comté",
    28: "Normandie", 32: "Hauts-de-France", 44: "Grand Est", 52: "Pays de la Loire",
    53: "Bretagne", 75: "Nouvelle-Aquitaine", 76: "Occitanie",
    84: "Auvergne-Rhône-Alpes", 93: "Provence-Alpes-Côte d'Azur", 94: "Corse",
}

# Régions DOM = un seul département : la part des 75+ s'y lit au niveau DEP.
REG_DOM_DEP = {1: "971", 2: "972", 3: "973", 4: "974", 6: "976"}


def normaliser_0_100(valeurs):
    """Normalisation min–max entre 0 et 100 (série constante -> tout à 0)."""
    v = [float(x) for x in valeurs]
    if not v:
        return []
    mini, maxi = min(v), max(v)
    if maxi == mini:
        return [0.0 for _ in v]
    return [100.0 * (x - mini) / (maxi - mini) for x in v]


def attribuer_rangs(indices, cles):
    """Rangs uniques 1..N ; le rang 1 va à l'indice le plus élevé.

    En cas d'égalité, `cles` (p. ex. le nom de la région) départage de façon
    déterministe, ce qui garantit des rangs toujours uniques.
    """
    indices = [float(x) for x in indices]
    ordre = sorted(range(len(indices)), key=lambda i: (-indices[i], str(cles[i])))
    rangs = [0] * len(indices)
    for rang, i in enumerate(ordre, start=1):
        rangs[i] = rang
    return rangs


def poids_prevalences(prevalence):
    """Poids de chaque pathologie = fréquence nationale moyenne 2024, normalisé (somme = 1)."""
    moyennes = prevalence.groupby("patho")["prev_2024"].mean()
    total = float(moyennes.sum())
    if total <= 0:
        raise ValueError("prévalences 2024 toutes nulles : poids indéfinis")
    return {patho: float(v) / total for patho, v in moyennes.items()}


def charger_population_regions(chemin=None):
    """Population et part des 75 ans et plus par région (INSEE, dernière année complète).

    Niveau REG ; pour les régions DOM, dont la part des 75+ n'existe pas au
    niveau REG, elle est lue au niveau DEP du département correspondant.
    """
    chemin = Path(chemin) if chemin is not None else POP_ZIP
    with zipfile.ZipFile(chemin) as z:
        nom = [n for n in z.namelist() if n.endswith("_data.csv")][0]
        raw = pd.read_csv(io.BytesIO(z.read(nom)), sep=";", dtype=str,
                          usecols=["GEO", "GEO_OBJECT", "EP_MEASURE", "AGE", "SEX", "TIME_PERIOD", "OBS_VALUE"])
    raw["annee"] = raw.TIME_PERIOD.astype(int)
    raw["val"] = pd.to_numeric(raw.OBS_VALUE, errors="coerce")

    reg = raw[raw.GEO_OBJECT == "REG"].copy()
    reg["reg"] = reg.GEO.astype(int)

    def serie_reg(ep, age):
        s = reg[(reg.EP_MEASURE == ep) & (reg.AGE == age) & (reg.SEX == "_T")]
        return s.set_index(["reg", "annee"])["val"]

    out = pd.DataFrame({"population": serie_reg("POP_JAN_1ST", "_T"),
                        "part_75plus": serie_reg("PT_IN_POP", "Y_GE75")}).reset_index()
    out = out[out.reg.isin(REGIONS)]

    dep = raw[raw.GEO_OBJECT == "DEP"]

    def p75_dep(dep_code, annee):
        s = dep[(dep.GEO == dep_code) & (dep.EP_MEASURE == "PT_IN_POP") & (dep.AGE == "Y_GE75")
                & (dep.SEX == "_T") & (dep.annee == annee)]
        return float(s.val.iloc[0]) if len(s) and pd.notna(s.val.iloc[0]) else None

    masque = out.part_75plus.isna() & out.reg.map(REG_DOM_DEP).notna()
    for idx, row in out[masque].iterrows():
        valeur = p75_dep(REG_DOM_DEP[int(row.reg)], int(row.annee))
        if valeur is not None:
            out.loc[idx, "part_75plus"] = valeur

    derniere = int(out.annee.max())
    out = out[out.annee == derniere].copy()
    out["region_nom"] = out.reg.map(REGIONS)
    if len(out) != len(REGIONS) or out.population.isna().any() or out.part_75plus.isna().any():
        raise ValueError(f"population ou part des 75 ans et plus manquante pour l'année {derniere}")
    return (out[["reg", "region_nom", "annee", "population", "part_75plus"]]
            .sort_values("reg").reset_index(drop=True))


def calculer_besoin_regional(previsions, prevalence, population,
                             colonne=COLONNE_PREVISION, horizon=HORIZON):
    """DataFrame du besoin relatif par région (voir le docstring du module).

    Pur : les entrées sont trois DataFrames (`previsions_boites_pharma.csv`,
    `prevalence_projections_2030.csv`, population régionale INSEE).
    """
    futur = previsions[previsions.horizon == horizon]
    if futur.empty:
        raise ValueError(f"pas de prévision à l'horizon {horizon}")
    if colonne not in futur.columns:
        raise ValueError(f"colonne de prévision inconnue : {colonne!r}")
    nationale = futur.set_index("produit")[colonne].astype(float)
    if nationale.isna().any() or (nationale < 0).any():
        raise ValueError(f"prévisions {colonne!r} manquantes ou négatives à l'horizon {horizon}")
    origine = futur.origine.iloc[0]
    mois_cible = futur.mois.iloc[0]

    pop = population.copy()
    for col in ("reg", "region_nom", "population", "part_75plus"):
        if col not in pop.columns:
            raise ValueError(f"colonne manquante dans la population régionale : {col!r}")
    pop["reg"] = pop.reg.astype(int)
    pop["pop_75plus"] = pop.population.astype(float) * pop.part_75plus.astype(float) / 100.0
    total_75 = float(pop.pop_75plus.sum())
    if total_75 <= 0:
        raise ValueError("population des 75 ans et plus nulle ou négative")
    pop["part_pop_75plus"] = pop.pop_75plus / total_75

    prev = prevalence.copy()
    prev["reg"] = prev.reg.astype(int)
    manquantes = sorted(set(prev.reg.unique()) - set(pop.reg.unique()))
    if manquantes:
        raise ValueError(f"régions sans population INSEE (codes reg) : {manquantes}")
    pivot = prev.pivot(index="reg", columns="patho", values="prev_2030")
    manquants = [p for p in PATHOLOGIES if p not in pivot.columns]
    if manquants:
        raise ValueError(f"pathologies manquantes dans les projections 2030 : {manquants}")
    poids = poids_prevalences(prev)
    prev_ponderee = sum(poids[p] * pivot[p] for p in PATHOLOGIES).rename("prevalence_ponderee_2030").reset_index()

    df = (prev[["reg", "region_nom"]].drop_duplicates("reg")
          .merge(pop[["reg", "population", "part_75plus", "pop_75plus", "part_pop_75plus"]], on="reg")
          .merge(prev_ponderee, on="reg"))
    produits = sorted(nationale.index)
    for produit in produits:
        df[f"boites_{produit}"] = nationale[produit] * df.part_pop_75plus
    df["boites_prevues_total"] = df[[f"boites_{p}" for p in produits]].sum(axis=1)
    df["indice_brut"] = df.boites_prevues_total / df.population * df.prevalence_ponderee_2030
    df["indice_besoin_0_100"] = normaliser_0_100(df.indice_brut.tolist())
    df["rang"] = attribuer_rangs(df.indice_besoin_0_100.tolist(), df.region_nom.tolist())
    df["origine"] = origine
    df["mois_cible"] = mois_cible
    df["horizon"] = horizon
    colonnes = ["origine", "mois_cible", "horizon", "reg", "region_nom", "population", "part_75plus",
                "pop_75plus", "boites_prevues_total", *[f"boites_{p}" for p in produits],
                "prevalence_ponderee_2030", "indice_besoin_0_100", "rang"]
    return df[colonnes].sort_values("rang").reset_index(drop=True)


def main(dest=None):
    dest = Path(dest) if dest is not None else OUT / "besoin_regional.csv"
    previsions = pd.read_csv(OUT / "previsions_boites_pharma.csv")
    prevalence = pd.read_csv(OUT / "prevalence_projections_2030.csv")
    population = charger_population_regions()
    df = calculer_besoin_regional(previsions, prevalence, population)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False, float_format="%.4f")
    print(f"[besoin_regional] -> {dest} ({len(df)} régions, prévision {df.mois_cible.iloc[0]} "
          f"à {df.horizon.iloc[0]} mois, origine {df.origine.iloc[0]})")
    print(df[["rang", "region_nom", "boites_prevues_total", "prevalence_ponderee_2030",
              "indice_besoin_0_100"]].head(5).to_string(index=False))
    return dest


if __name__ == "__main__":
    main()
