"""Incertitude du gain sur la période de vérification réservée (chantier #6).

Toutes les fonctions sont pures : mêmes entrées et même graine -> mêmes
sorties. Elles ne servent jamais à régler un modèle ; le choix du candidat et
le poids du mélange restent cantonnés à la période de développement (voir
`recherche_delai_pharma.poids_du_melange`), la période >= DEBUT_VERIFICATION
étant strictement réservée à la mesure.
"""

import math

import numpy as np
import pandas as pd


def _serie(valeurs, nom):
    serie = np.asarray(valeurs, dtype=float)
    if serie.ndim != 1 or serie.size == 0:
        raise ValueError(f"{nom} doit être une série 1D non vide.")
    if not np.isfinite(serie).all():
        raise ValueError(f"{nom} contient des valeurs non finies.")
    return serie


def block_bootstrap_gain(erreurs_a, erreurs_b, taille_bloc=3, n_bootstrap=2000, seed=0,
                         denominateur=None):
    """Bootstrap par blocs du gain du modèle A sur le modèle B.

    `erreurs_a` et `erreurs_b` sont les erreurs (signées ou absolues) des deux
    modèles, une par mois de la période évaluée. Le gain est `mean(e_b - e_a)`
    — un gain de MAE en boîtes si les erreurs sont absolues — ou, si
    `denominateur` est fourni (ex. les volumes observés `y`), le ratio
    `sum(e_b - e_a) / sum(y)`, c.-à-d. un gain de WAPE avec des erreurs
    absolues. Des blocs de `taille_bloc` mois consécutifs sont
    rééchantillonnés avec remise, ce qui préserve l'autocorrélation mensuelle.

    Retourne un dict : `gain` observé, intervalle de confiance à 95 %
    [`ic95_bas`, `ic95_haut`], `proba_gain_positif` (fraction des
    rééchantillons où le gain est positif ou nul) et les paramètres utilisés.
    Reproductible : même `seed` -> même intervalle.
    """
    e_a = _serie(erreurs_a, "erreurs_a")
    e_b = _serie(erreurs_b, "erreurs_b")
    if e_a.shape != e_b.shape:
        raise ValueError("erreurs_a et erreurs_b doivent avoir la même longueur.")
    n = e_a.size
    if denominateur is None:
        y = None
    else:
        y = _serie(denominateur, "denominateur")
        if y.shape != e_a.shape:
            raise ValueError("denominateur doit avoir la même longueur que les erreurs.")
        if (y < 0).any() or y.sum() <= 0:
            raise ValueError("denominateur doit être positif, de somme strictement positive.")
    taille_bloc = int(taille_bloc)
    if taille_bloc < 1:
        raise ValueError("taille_bloc doit être >= 1.")
    taille_bloc = min(taille_bloc, n)
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap doit être >= 1.")

    def gain(idx):
        diff = e_b[idx] - e_a[idx]
        if y is None:
            return float(diff.mean())
        return float(diff.sum() / y[idx].sum())

    rng = np.random.default_rng(seed)
    n_blocs = -(-n // taille_bloc)
    dernier_debut = n - taille_bloc
    echantillon = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        debuts = rng.integers(0, dernier_debut + 1, size=n_blocs)
        idx = np.concatenate([np.arange(d, d + taille_bloc) for d in debuts])[:n]
        echantillon[i] = gain(idx)
    bas, haut = np.percentile(echantillon, [2.5, 97.5])
    return {
        "gain": gain(np.arange(n)),
        "ic95_bas": float(bas),
        "ic95_haut": float(haut),
        "proba_gain_positif": float((echantillon >= 0).mean()),
        "n_bootstrap": n_bootstrap,
        "taille_bloc": taille_bloc,
    }


def diebold_mariano(erreurs_a, erreurs_b, horizon=1):
    """Test de Diebold-Mariano classique entre deux séries d'erreurs signées.

    La différentielle de perte est `d_t = |e_b,t|^h - |e_a,t|^h` avec
    `h = horizon` ; sous H0 (mêmes précisions), `DM = mean(d) / sqrt(S/n)` suit
    asymptotiquement N(0, 1), avec la variance HAC tronquée à h-1 retards :
    `S = gamma_0 + 2 * sum_{j=1}^{h-1} gamma_j`. `DM > 0` signifie que A est
    meilleur que B. Retourne un dict : statistique, p-value bilatérale
    (loi normale), gain moyen et paramètres.
    """
    e_a = _serie(erreurs_a, "erreurs_a")
    e_b = _serie(erreurs_b, "erreurs_b")
    if e_a.shape != e_b.shape:
        raise ValueError("erreurs_a et erreurs_b doivent avoir la même longueur.")
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon doit être >= 1.")
    n = e_a.size
    d = np.abs(e_b) ** h - np.abs(e_a) ** h
    dbar = float(d.mean())
    s = float(((d - dbar) ** 2).sum() / n)
    for j in range(1, h):
        s += 2.0 * float(((d[:n - j] - dbar) * (d[j:] - dbar)).sum() / n)
    if s <= 0:
        # S nul : différentielle constante. Nulle -> égalité parfaite (p = 1) ;
        # non nulle -> écart déterministe (p = 0).
        if dbar == 0:
            dm, p = 0.0, 1.0
        else:
            dm, p = math.copysign(math.inf, dbar), 0.0
    else:
        dm = dbar / math.sqrt(s / n)
        p = math.erfc(abs(dm) / math.sqrt(2.0))
    return {
        "dm_stat": float(dm),
        "p_value": float(p),
        "gain_moyen": dbar,
        "horizon": h,
        "n": int(n),
        "significatif_5pct": bool(p < 0.05),
    }


def scores_avec_ic(df, modeles, cible="observe", bloc=3, meilleur=None,
                   n_bootstrap=2000, seed=0):
    """Scores usuels + intervalles de confiance du gain du meilleur modèle.

    Applique `scores` de `recherche_volumes_pharma` (import tardif pour éviter
    l'import circulaire) sur `df` restreint à `modeles`, puis ajoute pour
    chaque modèle :
    - `gain_mae_boites` avec son IC à 95 % [`ic95_mae_bas`, `ic95_mae_haut`] :
      gain de MAE du meilleur modèle sur ce modèle, bootstrap par blocs de
      `bloc` mois ;
    - `gain_wape_pct` avec son IC à 95 % [`ic95_wape_bas`, `ic95_wape_haut`] :
      même gain exprimé en points de WAPE (bootstrap pondéré par `cible`) ;
    - `proba_gain_positif`, `dm_stat`, `dm_p_value` : probabilité bootstrap et
      test de Diebold-Mariano (horizon 1) du gain.

    `meilleur` force le modèle de référence du gain (ex. le candidat retenu
    sur le développement) ; par défaut c'est le meilleur au sens de la WAPE
    sur `df`. La ligne du meilleur modèle porte `est_meilleur=True` et des
    gains nuls. Retourne un DataFrame trié par WAPE croissante.
    """
    from .recherche_volumes_pharma import scores  # import tardif : évite le cycle

    modeles = list(modeles)
    if not modeles:
        raise ValueError("modeles ne doit pas être vide.")
    manquants = [c for c in [cible, *modeles] if c not in df.columns]
    if manquants:
        raise ValueError(f"Colonnes absentes du DataFrame : {manquants}")
    base = scores(df if cible == "observe" else df.assign(observe=df[cible]), modeles)
    if meilleur is None:
        meilleur = base.modele.iloc[0]
    elif meilleur not in modeles:
        raise ValueError(f"meilleur={meilleur!r} doit figurer dans modeles.")
    reel = df[cible].to_numpy(dtype=float)
    erreur = {m: df[m].to_numpy(dtype=float) - reel for m in modeles}
    lignes = []
    for _, ligne in base.iterrows():
        m = ligne.modele
        ligne = ligne.to_dict()
        if m == meilleur:
            ligne.update({
                "est_meilleur": True,
                "gain_mae_boites": 0.0, "ic95_mae_bas": 0.0, "ic95_mae_haut": 0.0,
                "gain_wape_pct": 0.0, "ic95_wape_bas": 0.0, "ic95_wape_haut": 0.0,
                "proba_gain_positif": 1.0, "dm_stat": 0.0, "dm_p_value": 1.0,
            })
        else:
            mae = block_bootstrap_gain(np.abs(erreur[meilleur]), np.abs(erreur[m]),
                                       taille_bloc=bloc, n_bootstrap=n_bootstrap, seed=seed)
            wape = block_bootstrap_gain(np.abs(erreur[meilleur]), np.abs(erreur[m]),
                                        taille_bloc=bloc, n_bootstrap=n_bootstrap, seed=seed,
                                        denominateur=reel)
            dm = diebold_mariano(erreur[meilleur], erreur[m], horizon=1)
            ligne.update({
                "est_meilleur": False,
                "gain_mae_boites": mae["gain"],
                "ic95_mae_bas": mae["ic95_bas"], "ic95_mae_haut": mae["ic95_haut"],
                "gain_wape_pct": 100.0 * wape["gain"],
                "ic95_wape_bas": 100.0 * wape["ic95_bas"],
                "ic95_wape_haut": 100.0 * wape["ic95_haut"],
                "proba_gain_positif": mae["proba_gain_positif"],
                "dm_stat": dm["dm_stat"], "dm_p_value": dm["p_value"],
            })
        lignes.append(ligne)
    return pd.DataFrame(lignes)
