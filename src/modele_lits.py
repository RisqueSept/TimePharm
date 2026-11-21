"""Projection nationale des capacités SAE et scénarios de charge en MCO."""

import io
import calendar
import os
import zipfile
from itertools import product

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # pas d'affichage, on sauvegarde des PNG
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from extract_lits import DISCIPLINES_TOTAL
from table_mensuelle import REGIONS

BASE = os.path.join(os.path.dirname(__file__), "..")
PROCESSED = os.path.join(BASE, "data", "processed")
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "outputs")

HORIZON = list(range(2026, 2036))  # 2026 -> 2035
HORIZONS_TEST = (1, 2, 3)
HYPOTHESES_MCO = tuple(product((-0.05, 0.0, 0.05), (0.80, 0.85, 0.90)))


# ---------------------------------------------------------------- population
def charger_population() -> pd.DataFrame:
    """Population France (GEO='F', fallback 'FM') et part des 75 ans et +."""
    z = zipfile.ZipFile(os.path.join(RAW, "insee_pop_struct_series.zip"))
    nom = [n for n in z.namelist() if n.endswith("_data.csv")][0]
    df = pd.read_csv(io.BytesIO(z.read(nom)), sep=";", dtype=str)
    df = df[df.EP_MEASURE == "POP_JAN_1ST"]
    df["annee"] = df.TIME_PERIOD.astype(int)
    df["val"] = pd.to_numeric(df.OBS_VALUE, errors="coerce")

    def serie(geo, age):
        s = df[(df.GEO == geo) & (df.SEX == "_T") & (df.AGE == age)]
        return s.set_index("annee")["val"]

    pop_f, pop_fm = serie("F", "_T"), serie("FM", "_T")
    p75_f, p75_fm = serie("F", "Y_GE75"), serie("FM", "Y_GE75")

    # France entière estimée pour les années où seul FM est publié :
    # pop_F ~= pop_FM + delta, delta = écart moyen F-FM sur les années communes récentes
    communes = pop_f.index.intersection(pop_fm.index)
    delta = float((pop_f[communes] - pop_fm[communes]).sort_index().tail(6).mean())
    communes_75 = p75_f.dropna().index.intersection(p75_fm.dropna().index)
    delta_75 = float((p75_f[communes_75] - p75_fm[communes_75]).sort_index().tail(6).mean())
    annees = sorted(set(pop_f.index) | set(pop_fm.index))
    pop, p75 = {}, {}
    for a in annees:
        if a in pop_f.index:
            pop[a], p75[a] = pop_f[a], p75_f.get(a, np.nan)
        else:
            pop[a], p75[a] = pop_fm[a] + delta, p75_fm.get(a, np.nan) + delta_75
    out = pd.DataFrame({"annee": annees, "population": pd.Series(pop), "pop_75plus": pd.Series(p75)})
    out["part_75plus"] = 100 * out.pop_75plus / out.population
    return out.sort_values("annee").reset_index(drop=True)


# ---------------------------------------------------------------- dataset
def construire_dataset() -> pd.DataFrame:
    lits = pd.read_csv(os.path.join(PROCESSED, "lits_sae_annuel.csv"))
    pop = charger_population()
    df = lits.merge(pop, on="annee", how="left", validate="one_to_one")
    if df.population.isna().any():
        raise ValueError("Population INSEE manquante pour une année SAE")
    df["lits_pour_1000"] = 1000 * df.lits_total / df.population
    out = os.path.join(PROCESSED, "dataset_annuel.csv")
    df.to_csv(out, index=False)
    print(f"[dataset] -> {out}")
    print(df[["annee", "lits_total", "population", "part_75plus", "lits_pour_1000"]]
          .round(2).to_string(index=False))
    return df


# ---------------------------------------------------------------- modèles
def _lin(x, y):
    m = LinearRegression().fit(np.asarray(x, float).reshape(-1, 1), np.asarray(y, float))
    return m


def _projeter_population(df, annees_futures):
    """Extrapole la population à partir des années d’apprentissage."""
    modele = _lin(df.annee, df.population)
    return modele.predict(np.asarray(annees_futures, float).reshape(-1, 1))


def _predictions(hist, annees, moteur=None):
    x = np.asarray(annees).reshape(-1, 1)
    tendance = _lin(hist.annee, hist.lits_total).predict(x)
    recent = hist.tail(5)
    tendance_recente = _lin(recent.annee, recent.lits_total).predict(x)
    ratio = _lin(hist.annee, hist.lits_pour_1000).predict(x)
    pop = _projeter_population(hist, annees)
    resultats = pd.DataFrame({
        "pred_Persistance": hist.lits_total.iloc[-1],
        "pred_A_tendance": tendance,
        "pred_B_ratio_x_pop": ratio * pop / 1000,
        "pred_C_tendance_recente": tendance_recente,
    })
    if moteur is not None:
        horizons = np.asarray(annees) - hist.annee.iloc[-1]
        quantiles = moteur.quantiles(hist.lits_total.to_numpy(), int(horizons.max()))
        resultats[["timestral_q10", "pred_Timestral", "timestral_q90"]] = quantiles[0, 0, horizons - 1]
    return resultats


def _backtest(hist, moteur=None):
    blocs = []
    for i in range(5, len(hist)):
        apprentissage = hist.iloc[:i]
        origine = apprentissage.annee.iloc[-1]
        test = hist[hist.annee.sub(origine).isin(HORIZONS_TEST)]
        pred = _predictions(apprentissage, test.annee, moteur)
        pred["annee"] = test.annee.to_numpy()
        pred["origine"] = origine
        pred["horizon_ans"] = pred.annee - origine
        pred["lits_reel"] = test.lits_total.to_numpy()
        blocs.append(pred)
    bt = pd.concat(blocs, ignore_index=True)
    erreurs = bt.filter(like="pred_").sub(bt.lits_reel, axis=0)
    rmse = erreurs.pow(2).mean().pow(0.5)
    principal = "pred_Timestral" if moteur is not None else rmse.idxmin()
    bt["modele"] = principal.removeprefix("pred_")
    print("\n[backtest annuel] RMSE par horizon (lits)")
    par_horizon = erreurs.pow(2).groupby(bt.horizon_ans).mean().pow(0.5)
    par_horizon["origines"] = bt.groupby("horizon_ans").origine.nunique()
    print(par_horizon.round().to_string())
    print(f"[backtest annuel] {principal.removeprefix('pred_')} : RMSE agrégée {rmse[principal]:.0f} lits")
    return bt, principal


def scenarios_mco(hist, prev):
    """Convertit des hypothèses de journées et d'occupation en lits ouverts moyens."""
    derniere = hist.iloc[-1]
    annee_ref = int(derniere.annee)
    jours_ref = 365 + calendar.isleap(annee_ref)
    activite_jour = derniere.jou_mco / jours_ref
    lignes = []
    for annee in prev.itertuples(index=False):
        for variation, occupation in HYPOTHESES_MCO:
            lits_equiv = int(np.ceil(activite_jour * (1 + variation) / occupation))
            lignes.append({
                "annee": annee.annee,
                "annee_reference": annee_ref,
                "journees_mco_reference": int(derniere.jou_mco),
                "taux_occupation_observe_ref": derniere.jou_mco / derniere.jli_mco,
                "variation_activite": variation,
                "taux_occupation_cible": occupation,
                "lits_ouverts_equivalents": lits_equiv,
                "lits_sae_mco_projetes": annee.lits_mco,
                "ecart_equivalents_moins_capacite": lits_equiv - annee.lits_mco,
            })
    scenarios = pd.DataFrame(lignes)
    scenarios.to_csv(os.path.join(OUT, "scenarios_capacite_mco.csv"), index=False)
    return scenarios


def scenarios_mco_regionaux():
    mco = pd.read_csv(os.path.join(PROCESSED, "mco_regional_reference.csv"))
    mco["territoire"] = mco.reg.map({**REGIONS, 5: "Saint-Pierre-et-Miquelon"})
    if mco.territoire.isna().any() or not mco.jli_mco.gt(0).all():
        raise ValueError("Territoire ou journées de capacité MCO de référence invalides")
    mco = mco.rename(columns={"annee": "annee_reference", "jou_mco": "journees_mco_reference",
                              "lit_mco": "lits_sae_mco_au_31_decembre"})
    mco["taux_occupation_observe_ref"] = mco.journees_mco_reference / mco.pop("jli_mco")
    hypotheses = pd.DataFrame(HYPOTHESES_MCO, columns=["variation_activite", "taux_occupation_cible"])
    scenarios = mco.merge(hypotheses, how="cross")
    jours = 365 + scenarios.annee_reference.map(calendar.isleap)
    charge = scenarios.journees_mco_reference * (1 + scenarios.variation_activite)
    scenarios["lits_ouverts_equivalents"] = np.ceil(charge / jours / scenarios.taux_occupation_cible).astype(int)
    scenarios["ecart_equivalents_moins_capacite"] = (
        scenarios.lits_ouverts_equivalents - scenarios.lits_sae_mco_au_31_decembre
    ).astype(int)
    scenarios.to_csv(os.path.join(OUT, "scenarios_capacite_mco_regions.csv"), index=False)
    return scenarios


# --------------------------------------------------- scénarios MCO saisonniers
# Le SAE ne publie l'activité MCO (JOU_MCO) qu'au niveau annuel. La seule
# série mensuelle territoriale d'hospitalisation disponible localement est
# l'hospitalisation SARS-CoV-2 (SPF, 2020-03 -> 2023-06) : on en déduit un
# profil saisonnier par région (niveau journalier moyen par mois calendaire,
# moyenne sur les années observées). À défaut de série mensuelle pour une
# région, un profil plat documenté est utilisé (charge journalière constante :
# part_m = jours_m / jours_annee).


def _jours_par_mois(annee: int) -> dict:
    """Nombre de jours de chaque mois de l'année de référence."""
    return {m: calendar.monthrange(annee, m)[1] for m in range(1, 13)}


def _niveau_mensuel_hosp_regional() -> pd.DataFrame:
    """Niveau journalier moyen d'hospitalisation par (région, mois calendaire).

    Source privilégiée : data/processed/table_mensuelle_regions.csv (lits COVID
    occupés ramenés à un taux pour 100 000 habitants). Sinon repli sur l'export
    brut SPF data/raw/mensuel/covid_hosp_txad_reg.csv (PourAvec=0, tx_prev_hosp).
    Renvoie un DataFrame (reg, mois, niveau) vide si aucune source n'existe.
    """
    chemin = os.path.join(PROCESSED, "table_mensuelle_regions.csv")
    if os.path.exists(chemin):
        df = pd.read_csv(chemin)
        niveau = (pd.to_numeric(df.lits_covid_occupes, errors="coerce")
                  / pd.to_numeric(df.population, errors="coerce") * 1e5)
        out = pd.DataFrame({
            "reg": df.reg.astype(int),
            "mois": pd.to_datetime(df.mois).dt.month,
            "niveau": niveau,
        }).replace([np.inf, -np.inf], np.nan).dropna(subset=["niveau"])
        return out.groupby(["reg", "mois"], as_index=False).niveau.mean()
    brut = os.path.join(RAW, "mensuel", "covid_hosp_txad_reg.csv")
    if os.path.exists(brut):
        df = pd.read_csv(brut, sep=";")
        df = df[df.PourAvec == 0]
        out = pd.DataFrame({
            "reg": pd.to_numeric(df.reg, errors="coerce").astype("Int64").astype(int),
            "mois": pd.to_datetime(df.jour).dt.month,
            "niveau": pd.to_numeric(df.tx_prev_hosp, errors="coerce"),
        }).dropna(subset=["reg", "niveau"])
        return out.groupby(["reg", "mois"], as_index=False).niveau.mean()
    return pd.DataFrame(columns=["reg", "mois", "niveau"])


def charger_profil_saisonnier_mco(annee_ref: int, regions=None) -> pd.DataFrame:
    """Profil saisonnier des journées MCO : part de chaque mois dans l'année.

    Pour chaque région, la part du mois m vaut
        niveau_m * jours_m / somme_k(niveau_k * jours_k)
    où niveau_m est le niveau journalier moyen du mois calendaire m (moyenne
    sur les années observées de la série mensuelle territoriale). Les parts
    somment à 1 par région. Une région sans série mensuelle (ou de niveau nul)
    reçoit le profil plat documenté : part_m = jours_m / jours_annee.

    Renvoie un DataFrame (reg, mois, part, source_profil).
    """
    jours = _jours_par_mois(annee_ref)
    jours_annee = sum(jours.values())
    niveau = _niveau_mensuel_hosp_regional()
    if regions is None:
        regions = sorted(niveau.reg.unique()) if not niveau.empty else []
    lignes = []
    for reg in regions:
        sous = niveau[niveau.reg == reg] if not niveau.empty else niveau
        if not sous.empty:
            moy = sous.groupby("mois").niveau.mean()
            poids = {m: float(moy.get(m, 0.0)) * jours[m] for m in range(1, 13)}
            total = sum(poids.values())
        else:
            poids, total = {}, 0.0
        if total <= 0:  # profil plat documenté : charge journalière constante
            parts = {m: jours[m] / jours_annee for m in range(1, 13)}
            source = "profil_plat"
        else:
            parts = {m: poids[m] / total for m in range(1, 13)}
            source = "spf_covid_mensuel"
        for m in range(1, 13):
            lignes.append({"reg": int(reg), "mois": m, "part": parts[m], "source_profil": source})
    return pd.DataFrame(lignes)


def scenarios_mco_saisonnier_regionaux(profil=None) -> pd.DataFrame:
    """Scénarios MCO territoriaux avec distribution saisonnière mensuelle.

    Pour chaque région et chaque hypothèse (variation d'activité x taux
    d'occupation cible), la charge annuelle de référence (JOU_MCO) est
    répartie entre les 12 mois selon le profil saisonnier ; le mois m exige
        lits_equivalents_pic = ceil(journees_m * (1+variation) / (jours_m * taux))
    lits_equivalents_moyen reste la moyenne annuelle du scénario
    (identique à scenarios_capacite_mco_regions.csv). Le pic est donc toujours
    >= la moyenne. Écrit outputs/scenarios_capacite_mco_saisonnier_regions.csv.
    """
    mco = pd.read_csv(os.path.join(PROCESSED, "mco_regional_reference.csv"))
    mco["territoire"] = mco.reg.map({**REGIONS, 5: "Saint-Pierre-et-Miquelon"})
    if mco.territoire.isna().any() or not mco.jli_mco.gt(0).all():
        raise ValueError("Territoire ou journées de capacité MCO de référence invalides")
    annee_ref = int(mco.annee.max())
    if profil is None:
        profil = charger_profil_saisonnier_mco(annee_ref, regions=sorted(mco.reg.unique()))
    jours = _jours_par_mois(annee_ref)
    jours_annee = sum(jours.values())
    hypotheses = pd.DataFrame(HYPOTHESES_MCO, columns=["variation_activite", "taux_occupation_cible"])
    base = (mco.rename(columns={"annee": "annee_reference", "jou_mco": "journees_mco_reference",
                               "lit_mco": "lits_sae_mco_au_31_decembre"})
               .merge(profil, on="reg", how="left", validate="one_to_many")
               .merge(hypotheses, how="cross"))
    if base.part.isna().any():
        raise ValueError("Région sans profil saisonnier MCO")
    base["jours_du_mois"] = base.mois.map(jours)
    base["journees_mco_mois"] = base.journees_mco_reference * base.part
    charge = base.journees_mco_reference * (1 + base.variation_activite)
    charge_mois = base.journees_mco_mois * (1 + base.variation_activite)
    # même ordre d'opérations que scenarios_mco_regionaux pour la moyenne annuelle
    base["lits_equivalents_moyen"] = np.ceil(charge / jours_annee / base.taux_occupation_cible).astype(int)
    base["lits_equivalents_pic"] = np.ceil(charge_mois / base.jours_du_mois / base.taux_occupation_cible).astype(int)
    cols = ["annee_reference", "reg", "territoire", "lits_sae_mco_au_31_decembre",
            "journees_mco_reference", "variation_activite", "taux_occupation_cible", "mois",
            "jours_du_mois", "part_saisonnier", "journees_mco_mois",
            "lits_equivalents_pic", "lits_equivalents_moyen", "source_profil"]
    out = (base.assign(part_saisonnier=base.part.round(6), journees_mco_mois=base.journees_mco_mois.round(1))
              .sort_values(["reg", "variation_activite", "taux_occupation_cible", "mois"])
              .reset_index(drop=True)[cols])
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "scenarios_capacite_mco_saisonnier_regions.csv"), index=False)
    return out


def scenarios_mco_saisonnier_national(saisonnier=None) -> pd.DataFrame:
    """Version nationale du profil saisonnier MCO (somme des régions).

    Les journées mensuelles et les lits SAE de référence sont sommés sur les
    régions ; les lits équivalents sont recalculés sur ces totaux. Écrit
    outputs/scenarios_capacite_mco_saisonnier_national.csv (reg=0,
    territoire='France entière').
    """
    if saisonnier is None:
        saisonnier = scenarios_mco_saisonnier_regionaux()
    g = (saisonnier.groupby(["annee_reference", "variation_activite", "taux_occupation_cible", "mois"],
                            as_index=False)
         .agg(journees_mco_reference=("journees_mco_reference", "sum"),
              lits_sae_mco_au_31_decembre=("lits_sae_mco_au_31_decembre", "sum"),
              jours_du_mois=("jours_du_mois", "first"),
              journees_mco_mois=("journees_mco_mois", "sum")))
    annee_ref = int(g.annee_reference.iloc[0])
    jours_annee = 365 + calendar.isleap(annee_ref)
    g["part_saisonnier"] = g.journees_mco_mois / g.journees_mco_reference
    charge = g.journees_mco_reference * (1 + g.variation_activite)
    g["lits_equivalents_moyen"] = np.ceil(charge / jours_annee / g.taux_occupation_cible).astype(int)
    g["lits_equivalents_pic"] = np.ceil(
        g.journees_mco_mois * (1 + g.variation_activite) / g.jours_du_mois / g.taux_occupation_cible).astype(int)
    g["reg"] = 0
    g["territoire"] = "France entière"
    g["source_profil"] = "somme_regions"
    cols = ["annee_reference", "reg", "territoire", "lits_sae_mco_au_31_decembre",
            "journees_mco_reference", "variation_activite", "taux_occupation_cible", "mois",
            "jours_du_mois", "part_saisonnier", "journees_mco_mois",
            "lits_equivalents_pic", "lits_equivalents_moyen", "source_profil"]
    out = (g.assign(part_saisonnier=g.part_saisonnier.round(6), journees_mco_mois=g.journees_mco_mois.round(1))
              .sort_values(["variation_activite", "taux_occupation_cible", "mois"])
              .reset_index(drop=True)[cols])
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "scenarios_capacite_mco_saisonnier_national.csv"), index=False)
    return out


def scenarios_mco_saisonnier() -> dict:
    """Scénarios MCO saisonniers territoriaux + version nationale."""
    regions = scenarios_mco_saisonnier_regionaux()
    national = scenarios_mco_saisonnier_national(regions)
    return {"regions": regions, "national": national}


def entrainer_et_prevoir(df: pd.DataFrame, moteur):
    hist = df.sort_values("annee")
    if len(hist) < 6 or not hist.annee.diff().dropna().eq(1).all():
        raise ValueError("Le modèle annuel exige au moins six années SAE consécutives")
    os.makedirs(OUT, exist_ok=True)
    bt, principal = _backtest(hist, moteur)
    bt.to_csv(os.path.join(OUT, "backtest_predictions.csv"), index=False)
    annees = np.arange(hist.annee.iloc[-1] + 1, HORIZON[-1] + 1)
    hypotheses = _predictions(hist, annees, moteur).clip(lower=0)
    pred = hypotheses[principal]
    pop = moteur.quantiles(hist.population.to_numpy(), len(annees))[0, 0, :, 1]
    prev = pd.DataFrame({
        "annee": annees,
        "lits_pred": pred.round().astype(int),
        "scenario_bas": hypotheses.timestral_q10.round().astype(int),
        "scenario_haut": hypotheses.timestral_q90.round().astype(int),
        "population_proj": np.round(pop).astype(int),
        "modele": principal.removeprefix("pred_"),
    })
    # Parts des trois dernières années, arrondies avec conservation du total.
    parts = hist.tail(3)[DISCIPLINES_TOTAL].sum()
    allocation = np.outer(prev.lits_pred, parts / parts.sum())
    arrondi = np.floor(allocation).astype(int)
    for i, reste in enumerate(prev.lits_pred.to_numpy() - arrondi.sum(axis=1)):
        ordre = np.argsort(-(allocation[i] - arrondi[i]), kind="stable")
        arrondi[i, ordre[:reste]] += 1
    prev[DISCIPLINES_TOTAL] = arrondi
    prev.to_csv(os.path.join(OUT, "prediction_lits_2026_2035.csv"), index=False)
    scenarios_mco(hist, prev)
    scenarios_mco_regionaux()
    scenarios_mco_saisonnier()
    print(prev.to_string(index=False))
    return {"df": hist, "test": bt, "meilleur": principal.removeprefix("pred_"),
            "pred_test_meilleur": bt[principal], "prev": prev}


# ---------------------------------------------------------------- graphiques
def tracer(ctx):
    os.makedirs(OUT, exist_ok=True)
    df, prev = ctx["df"], ctx["prev"]
    hist = df

    # fig 1 : historique et plage des scénarios
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(hist.annee, hist.lits_total, "o-", color="#1f77b4", label="lits installés (SAE, historique)")
    ax.plot(prev.annee, prev.lits_pred, "s--", color="#d62728", label=f"prévision ({ctx['meilleur']})")
    ax.fill_between(prev.annee, prev.scenario_bas, prev.scenario_haut, color="#d62728", alpha=0.2,
                    label="quantiles 10 % et 90 % de Timestral")
    ax.set_title("Nombre total de lits hospitaliers en France — historique et prévision 2026-2035")
    ax.set_xlabel("année")
    ax.set_ylabel("lits installés")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_forecast_lits_total.png"), dpi=150)
    plt.close(fig)

    # fig 2 : lits par discipline (historique fiable)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    disciplines = ["lits_mco", "lits_ssr", "lits_psy", "lits_usld"]
    labels = ["MCO", "SSR/SMR", "Psychiatrie", "USLD (longue durée)"]
    bottom = np.zeros(len(hist))
    for d, lab in zip(disciplines, labels):
        ax.fill_between(hist.annee, bottom, bottom + hist[d].fillna(0), alpha=0.75, label=lab)
        bottom += hist[d].fillna(0).values
    ax.set_title("Lits installés par discipline (SAE)")
    ax.set_xlabel("année")
    ax.set_ylabel("lits")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_lits_par_discipline.png"), dpi=150)
    plt.close(fig)

    # fig 3 : backtest réel vs prédit (meilleur modèle)
    test, p = ctx["test"], ctx["pred_test_meilleur"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test.lits_reel, p, color="#2ca02c", zorder=3)
    lim = [min(test.lits_reel.min(), p.min()) * 0.98, max(test.lits_reel.max(), p.max()) * 1.02]
    ax.plot(lim, lim, "k--", alpha=0.5, label="prédiction parfaite")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("lits réels (SAE)")
    ax.set_ylabel("lits prédits (backtest)")
    ax.set_title(f"Backtest à 1–3 ans — modèle {ctx['meilleur']}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_backtest.png"), dpi=150)
    plt.close(fig)

    # fig 4 : ratio lits/1000 hab + part des 75+
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df.annee, df.lits_pour_1000, "o-", color="#9467bd", label="lits pour 1000 habitants")
    ax.set_xlabel("année")
    ax.set_ylabel("lits / 1000 hab.", color="#9467bd")
    ax2 = ax.twinx()
    ax2.plot(df.annee, df.part_75plus, "s-", color="#ff7f0e", label="part des 75 ans et + (%)")
    ax2.set_ylabel("75 ans et + (%)", color="#ff7f0e")
    ax.set_title("Ratio de lits et vieillissement de la population")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_ratio_et_vieillissement.png"), dpi=150)
    plt.close(fig)
    print("[plots] 4 figures PNG dans outputs/")


if __name__ == "__main__":
    df = construire_dataset()
    ctx = entrainer_et_prevoir(df)
    tracer(ctx)
