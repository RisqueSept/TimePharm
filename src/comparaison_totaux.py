"""Comparaison « total » vs « somme des prévisions individuelles » (chantier #8 de docs/ROADMAP.md).

Les backtests existants appliquent Timestral aux niveaux individuels — les six
formes buvables (`outputs/backtest_boites_pharma.csv`) et les 18 régions
(`outputs/backtest_mensuel.csv`) — jamais au total directement. Ce module mesure,
en lecture seule, l'erreur de l'agrégation « somme des prévisions individuelles »
par rapport au total observé :

- **pharma** : total prédit = somme des `timestral` des 6 formes, total observé =
  somme des `observe`, agrégés par (origine, horizon) ;
- **mensuel** : total France = somme des `pred_Timestral` régionales, vs somme
  des `lits_reel`, agrégés par (origine, horizon_mois).

La prévision « directe du total » (Timestral appliqué à la série agrégée) n'est
pas présente dans ces backtests : ce rapport publie le volet « somme des
prévisions individuelles » ; une ligne `stratégie="total_direct"` pourra être
ajoutée quand un tel backtest existera.

Rapport : MAE (unités natives), WAPE (%) et biais signé (%) par domaine et par
horizon, écrits dans `outputs/comparaison_totaux.csv` (colonnes : domaine,
horizon, stratégie, mae, wape, biais). Comme `wape_pct` / `biais_pct` de
`src/recherche_volumes_pharma.py` : WAPE = 100 * somme|erreur| / somme(observé)
et biais = 100 * somme(erreur signée) / somme(observé), positif = surestimation.

Exécution : `.venv-timestral/bin/python -m src.comparaison_totaux` (ou
`python src/comparaison_totaux.py`) depuis la racine du dépôt.
"""

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "COLONNES_RAPPORT",
    "DOMAINE_MENSUEL",
    "DOMAINE_PHARMA",
    "STRATEGIE_SOMME",
    "aggreger_total",
    "metriques_total",
    "comparaison_par_horizon",
    "comparaison_pharma",
    "comparaison_mensuel",
    "rapport",
    "main",
]

COLONNES_RAPPORT = ["domaine", "horizon", "stratégie", "mae", "wape", "biais"]
DOMAINE_PHARMA = "pharma"
DOMAINE_MENSUEL = "mensuel"
STRATEGIE_SOMME = "somme_individuelle"


def aggreger_total(df, cles, prevision, observe):
    """Somme `prevision` et `observe` par groupe de `cles`.

    Retourne un DataFrame `[cles..., total_predit, total_observe]` : le total
    prédit par somme des prévisions individuelles et le total observé sur le
    même groupe. Un groupe contenant une ligne sans `prevision` ou sans
    `observe` est exclu : un total partiel n'est pas un total.
    """
    manquants = [c for c in [*cles, prevision, observe] if c not in df.columns]
    if manquants:
        raise ValueError(f"Colonnes absentes du DataFrame : {manquants}")
    cles = list(cles)
    tailles = df.groupby(cles, sort=True).size()
    donnees = df.dropna(subset=[prevision, observe])
    complets = donnees.groupby(cles, sort=True).size()
    # groupes dont toutes les lignes ont `prevision` ET `observe`
    index_complets = tailles.index[tailles.to_numpy() == complets.reindex(tailles.index).fillna(0).to_numpy()]
    agg = (donnees.groupby(cles, sort=True)[[prevision, observe]].sum()
           .loc[index_complets]
           .rename(columns={prevision: "total_predit", observe: "total_observe"})
           .reset_index())
    return agg


def metriques_total(total_predit, total_observe):
    """MAE / WAPE (%) / biais signé (%) entre deux séries agrégées de même longueur.

    - `mae` = moyenne des |prédit - observé|, en unités natives ;
    - `wape` = 100 * somme|prédit - observé| / somme(observé) ;
    - `biais` = 100 * somme(prédit - observé) / somme(observé), positif = surestimation.

    WAPE et biais sont NaN si la somme observée est nulle. Retourne aussi `n`,
    le nombre de points agrégés.
    """
    p = np.asarray(total_predit, dtype=float)
    o = np.asarray(total_observe, dtype=float)
    if p.ndim != 1 or p.size == 0 or p.shape != o.shape:
        raise ValueError("total_predit et total_observe doivent être des séries 1D non vides de même longueur.")
    erreur = p - o
    somme_obs = float(o.sum())
    if somme_obs == 0.0:
        wape = biais = float("nan")
    else:
        wape = float(100.0 * np.abs(erreur).sum() / somme_obs)
        biais = float(100.0 * erreur.sum() / somme_obs)
    return {"n": int(p.size), "mae": float(np.abs(erreur).mean()), "wape": wape, "biais": biais}


def comparaison_par_horizon(agrege, domaine, strategie):
    """Une ligne par `horizon` de `agrege` (colonnes `cles` + total_predit/total_observe)."""
    lignes = []
    for horizon, g in agrege.groupby("horizon", sort=True):
        m = metriques_total(g.total_predit.to_numpy(), g.total_observe.to_numpy())
        lignes.append({
            "domaine": domaine,
            "horizon": int(horizon),
            "stratégie": strategie,
            "mae": m["mae"],
            "wape": m["wape"],
            "biais": m["biais"],
            "n": m["n"],
        })
    return pd.DataFrame(
        lignes,
        columns=["domaine", "horizon", "stratégie", "mae", "wape", "biais", "n"],
    )


def comparaison_pharma(boites, prevision="timestral", observe="observe"):
    """Volumes pharma : somme des prévisions des formes vs total observé, par (origine, horizon).

    `boites` est le contenu de `outputs/backtest_boites_pharma.csv` (une ligne par
    produit, origine et horizon).
    """
    agrege = aggreger_total(boites, ["origine", "horizon"], prevision, observe)
    return comparaison_par_horizon(agrege, DOMAINE_PHARMA, STRATEGIE_SOMME)


def comparaison_mensuel(mensuel, prevision="pred_Timestral", observe="lits_reel"):
    """Mensuel : somme des prévisions régionales vs total France observé, par (origine, horizon_mois).

    `mensuel` est le contenu de `outputs/backtest_mensuel.csv` (une ligne par
    région, origine et horizon).
    """
    agrege = aggreger_total(mensuel, ["origine", "horizon_mois"], prevision, observe)
    agrege = agrege.rename(columns={"horizon_mois": "horizon"})
    return comparaison_par_horizon(agrege, DOMAINE_MENSUEL, STRATEGIE_SOMME)


def rapport(boites, mensuel):
    """Rapport complet `[domaine, horizon, stratégie, mae, wape, biais]` (pharma d'abord)."""
    table = pd.concat([
        comparaison_pharma(boites),
        comparaison_mensuel(mensuel),
    ], ignore_index=True)
    ordre = {DOMAINE_PHARMA: 0, DOMAINE_MENSUEL: 1}
    table = table.assign(_ordre=table.domaine.map(ordre))
    table = table.sort_values(["_ordre", "horizon"], ignore_index=True).drop(columns="_ordre")
    return table[COLONNES_RAPPORT].reset_index(drop=True)


def main(racine=None):
    """Lit les deux backtests (lecture seule) et écrit `outputs/comparaison_totaux.csv`.

    Retourne le rapport écrit (DataFrame `COLONNES_RAPPORT`).
    """
    if racine is None:
        racine = Path(__file__).resolve().parents[1]
    sorties = Path(racine) / "outputs"
    boites = pd.read_csv(sorties / "backtest_boites_pharma.csv")
    mensuel = pd.read_csv(sorties / "backtest_mensuel.csv")
    table = rapport(boites, mensuel)
    destination = sorties / "comparaison_totaux.csv"
    table.to_csv(destination, index=False)
    print("[comparaison totaux] somme des prévisions individuelles vs total observé, par horizon")
    print(table.to_string(index=False))
    print(destination)
    return table


if __name__ == "__main__":
    main()
