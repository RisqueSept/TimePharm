"""Calibration mesurée des backtests pharma (chantier #4 de docs/ROADMAP.md).

Ce module ne réentraîne aucun modèle : il lit les backtests existants en
lecture seule (`outputs/backtest_boites_pharma.csv` et
`outputs/backtest_pharma.csv`) et en déduit des métriques de calibration :

- **volumes** : couverture empirique de la bande de quantiles [q10, q90]
  (globale, par produit, par horizon ; couverture visée de 80 % d'après
  `src/seuils.py: COUVERTURE_VISEE`) et biais signé (%) par produit et horizon ;
- **tensions** : Brier, courbe de fiabilité en 10 bins, taux de détection et
  de fausses alertes au seuil fixé a priori (`src/seuils.py:
  SEUIL_ALERTE_RISQUE` ; alerte si score >= seuil, comme dans `src/pharma.js`)
  et PR-AUC (average precision, calculée en NumPy pur, sans sklearn), par
  horizon et par produit, avec un volet séparé pour les nouvelles
  dégradations (lignes où le score `degradation` est renseigné).

Exécution : `python -m src.calibration` écrit `outputs/calibration_volumes.csv`
et `outputs/calibration_tensions.csv`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .seuils import COUVERTURE_VISEE, SEUIL_ALERTE_RISQUE

__all__ = [
    "NB_BINS_FIABILITE",
    "biais_par_forme",
    "couverture_quantiles",
    "metriques_binaires",
    "metriques_degradations",
    "table_volumes",
    "table_tensions",
    "main",
]

NB_BINS_FIABILITE = 10
TOUS = "tous"

COLONNES_VOLUMES = ["metrique", "niveau", "produit", "horizon", "n", "valeur"]
COLONNES_TENSIONS = [
    "sous_ensemble", "metrique", "niveau", "produit", "horizon", "n", "valeur",
    "bin", "proba_moyenne", "frequence_observee",
]
METRIQUES_RESUME = [
    "brier", "pr_auc", "taux_detection", "taux_fausses_alertes", "seuil",
    "n_positifs", "n_alertes", "n_detections", "n_fausses_alertes",
]


def _perimetres(df):
    """Yield (niveau, produit, horizon, sous-ensemble) : global, horizon, produit, produit x horizon."""
    yield ("global", TOUS, TOUS, df)
    if "horizon" in df.columns:
        for horizon, g in df.groupby("horizon", sort=True):
            yield ("horizon", TOUS, horizon, g)
    if "produit" in df.columns:
        for produit, g in df.groupby("produit", sort=True):
            yield ("produit", produit, TOUS, g)
    if {"produit", "horizon"} <= set(df.columns):
        for (produit, horizon), g in df.groupby(["produit", "horizon"], sort=True):
            yield ("horizon_produit", produit, horizon, g)


def couverture_quantiles(df, q_low="timestral_q10", q_high="timestral_q90", cible="observe"):
    """Fraction des observations dans la bande [q_low, q_high].

    Retourne une ligne par périmètre : global, par `produit`, par `horizon`
    et par produit x horizon. Les lignes sans quantile ni cible sont ignorées.
    """
    donnees = df.dropna(subset=[q_low, q_high, cible])
    lignes = []
    for niveau, produit, horizon, g in _perimetres(donnees):
        dans_bande = g[cible].between(g[q_low], g[q_high])
        lignes.append({
            "niveau": niveau,
            "produit": produit,
            "horizon": horizon,
            "n": int(len(g)),
            "couverture": float(dans_bande.mean()) if len(g) else float("nan"),
        })
    return pd.DataFrame(lignes, columns=["niveau", "produit", "horizon", "n", "couverture"])


def _brier(scores, y):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(s) == 0:
        return float("nan")
    return float(np.mean((s - y) ** 2))


def _pr_auc(scores, y):
    """Average precision (aire sous la courbe précision-rappel, par paliers).

    Même définition que sklearn.metrics.average_precision_score, calculée en
    NumPy pur : on parcourt les seuils décroissants (groupes d'égalités
    traités ensemble) et on somme (rappel_n - rappel_{n-1}) * précision_n.
    NaN s'il n'y a aucun positif ou que des positifs.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    n_pos = float(y.sum())
    if len(y) == 0 or n_pos == 0.0 or n_pos == len(y):
        return float("nan")
    ordre = np.argsort(-s, kind="stable")
    y, s = y[ordre], s[ordre]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    fin_de_groupe = np.r_[s[1:] != s[:-1], True]  # fin de chaque égalité de score
    tp, fp = tp[fin_de_groupe], fp[fin_de_groupe]
    precision = tp / (tp + fp)
    rappel = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, rappel]) * precision))


def _courbe_fiabilite(scores, y, nb_bins=NB_BINS_FIABILITE):
    """Courbe de fiabilité : `nb_bins` bins réguliers sur [0, 1] du score.

    Bin i = [i/nb_bins, (i+1)/nb_bins] (le premier bin est fermé à gauche, le
    dernier à droite ; un score hors [0, 1] est rabattu dans le bin extrême).
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = np.linspace(0.0, 1.0, nb_bins + 1)
    indices = np.clip(np.searchsorted(bins, s, side="left") - 1, 0, nb_bins - 1)
    lignes = []
    for b in range(nb_bins):
        masque = indices == b
        n = int(masque.sum())
        lignes.append({
            "bin": b,
            "bin_min": float(bins[b]),
            "bin_max": float(bins[b + 1]),
            "n": n,
            "proba_moyenne": float(s[masque].mean()) if n else float("nan"),
            "frequence_observee": float(y[masque].mean()) if n else float("nan"),
        })
    return pd.DataFrame(lignes, columns=["bin", "bin_min", "bin_max", "n", "proba_moyenne", "frequence_observee"])


def metriques_binaires(df, score="timestral", cible="observé", seuil=None):
    """Métriques de calibration d'un score binaire, par périmètre.

    Retourne {"resume": DataFrame, "fiabilite": DataFrame} :
    - resume : une ligne par périmètre (global, par `horizon`, par `produit`,
      produit x horizon) avec le Brier, le PR-AUC, et au `seuil` (par défaut
      `SEUIL_ALERTE_RISQUE` ; alerte si score >= seuil, comme `src/pharma.js`) :
      taux de détection (TP / positifs), taux de fausses alertes (FP / négatifs)
      et les effectifs n_alertes, n_detections, n_fausses_alertes ;
    - fiabilite : courbe de fiabilité en `NB_BINS_FIABILITE` bins par périmètre.

    Les lignes sans score ni cible sont ignorées.
    """
    if seuil is None:
        seuil = SEUIL_ALERTE_RISQUE
    donnees = df.dropna(subset=[score, cible])
    resume, fiabilite = [], []
    for niveau, produit, horizon, g in _perimetres(donnees):
        s = g[score].to_numpy(dtype=float)
        y = g[cible].to_numpy(dtype=float)
        alertes = s >= seuil
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        tp = int(np.sum(alertes & (y == 1)))
        fp = int(np.sum(alertes & (y == 0)))
        resume.append({
            "niveau": niveau,
            "produit": produit,
            "horizon": horizon,
            "n": int(len(g)),
            "n_positifs": n_pos,
            "brier": _brier(s, y),
            "pr_auc": _pr_auc(s, y),
            "seuil": float(seuil),
            "n_alertes": int(alertes.sum()),
            "n_detections": tp,
            "n_fausses_alertes": fp,
            "taux_detection": tp / n_pos if n_pos else float("nan"),
            "taux_fausses_alertes": fp / n_neg if n_neg else float("nan"),
        })
        courbe = _courbe_fiabilite(s, y)
        courbe.insert(0, "horizon", horizon)
        courbe.insert(0, "produit", produit)
        courbe.insert(0, "niveau", niveau)
        fiabilite.append(courbe)
    return {
        "resume": pd.DataFrame(resume),
        "fiabilite": pd.concat(fiabilite, ignore_index=True),
    }


def metriques_degradations(df, score="degradation", cible="observé", seuil=None):
    """Métriques du volet « nouvelles dégradations ».

    Se restreint aux lignes où le score `degradation` est renseigné (départ en
    couverture acceptable ou satisfaisante) et évalue ce score sur ce
    sous-ensemble, via `metriques_binaires`.
    """
    if score not in df.columns:
        raise KeyError(f"Colonne de score absente : {score!r}")
    subset = df.loc[df[score].notna()]
    return metriques_binaires(subset, score=score, cible=cible, seuil=seuil)


def biais_par_forme(df, prevision="timestral", cible="observe"):
    """Biais signé (%) des prévisions de volumes, par produit et par horizon.

    biais_pct = 100 * somme(prevision - observe) / somme(observe) : même
    pondération que la WAPE, un mois à gros volume pèse davantage. Un biais
    positif = surestimation.
    """
    donnees = df.dropna(subset=[prevision, cible])
    lignes = []
    for (produit, horizon), g in donnees.groupby(["produit", "horizon"], sort=True):
        somme_obs = float(g[cible].sum())
        erreur = float((g[prevision] - g[cible]).sum())
        lignes.append({
            "produit": produit,
            "horizon": horizon,
            "n": int(len(g)),
            "somme_observe": somme_obs,
            "somme_prevision": float(g[prevision].sum()),
            "erreur_signee": erreur,
            "biais_pct": 100.0 * erreur / somme_obs if somme_obs else float("nan"),
        })
    return pd.DataFrame(
        lignes,
        columns=["produit", "horizon", "n", "somme_observe", "somme_prevision", "erreur_signee", "biais_pct"],
    )


def table_volumes(boites):
    """Table longue prête pour `outputs/calibration_volumes.csv`."""
    lignes = []
    for r in couverture_quantiles(boites).itertuples(index=False):
        lignes.append({
            "metrique": "couverture", "niveau": r.niveau, "produit": r.produit,
            "horizon": r.horizon, "n": r.n, "valeur": r.couverture,
        })
    lignes.append({
        "metrique": "couverture_visee", "niveau": "reference", "produit": TOUS,
        "horizon": TOUS, "n": pd.NA, "valeur": COUVERTURE_VISEE,
    })
    for r in biais_par_forme(boites).itertuples(index=False):
        lignes.append({
            "metrique": "biais_pct", "niveau": "produit_horizon", "produit": r.produit,
            "horizon": r.horizon, "n": r.n, "valeur": r.biais_pct,
        })
    return pd.DataFrame(lignes, columns=COLONNES_VOLUMES)


def table_tensions(resultats, sous_ensemble="tous"):
    """Table longue prête pour `outputs/calibration_tensions.csv`.

    `resultats` est le dict renvoyé par `metriques_binaires` (ou
    `metriques_degradations`) ; `sous_ensemble` le étiquette ("tous" pour le
    score `timestral`, "degradations" pour le volet nouvelles dégradations).
    """
    lignes = []
    for r in resultats["resume"].itertuples(index=False):
        for m in METRIQUES_RESUME:
            lignes.append({
                "sous_ensemble": sous_ensemble, "metrique": m, "niveau": r.niveau,
                "produit": r.produit, "horizon": r.horizon, "n": r.n,
                "valeur": getattr(r, m), "bin": pd.NA, "proba_moyenne": pd.NA,
                "frequence_observee": pd.NA,
            })
    for r in resultats["fiabilite"].itertuples(index=False):
        lignes.append({
            "sous_ensemble": sous_ensemble, "metrique": "fiabilite", "niveau": r.niveau,
            "produit": r.produit, "horizon": r.horizon, "n": r.n,
            "valeur": r.frequence_observee, "bin": r.bin,
            "proba_moyenne": r.proba_moyenne, "frequence_observee": r.frequence_observee,
        })
    return pd.DataFrame(lignes, columns=COLONNES_TENSIONS)


def main(racine=None):
    """Lit les backtests existants et écrit les deux CSV de calibration.

    Lecture seule des entrées ; aucun réentraînement. Retourne les deux tables
    écrites (volumes, tensions).
    """
    if racine is None:
        racine = Path(__file__).resolve().parents[1]
    sorties = Path(racine) / "outputs"
    boites = pd.read_csv(sorties / "backtest_boites_pharma.csv")
    tensions = pd.read_csv(sorties / "backtest_pharma.csv")

    volumes = table_volumes(boites)
    volumes.to_csv(sorties / "calibration_volumes.csv", index=False)

    table = pd.concat([
        table_tensions(metriques_binaires(tensions, score="timestral"), sous_ensemble="tous"),
        table_tensions(metriques_degradations(tensions), sous_ensemble="degradations"),
    ], ignore_index=True)
    table.to_csv(sorties / "calibration_tensions.csv", index=False)
    return volumes, table


if __name__ == "__main__":
    main()
