"""Tableau mensuel de l’occupation hospitalière avec le SARS-CoV-2."""

from pathlib import Path

import pandas as pd

from dashboard import MODELES, enregistrements, fraicheur_dashboard, generer_html

OUT = Path(__file__).resolve().parents[1] / "outputs"
FRANCE = "France entière"
PATHOLOGIES = {
    "prev_diabete": "Diabète",
    "prev_cancer": "Cancers",
    "prev_cardio": "Maladies cardioneurovasculaires",
    "prev_psy": "Maladies psychiatriques",
    "prev_resp": "Maladies respiratoires chroniques",
    "prev_neuro": "Maladies neurologiques",
}


def _comparaisons(bt, territoires):
    resultats = {}
    for territoire in territoires:
        region = bt[bt.region_nom.eq(territoire)]
        resultats[territoire] = {
            int(h): enregistrements(groupe.rename(columns={"mois": "date"}))
            for h, groupe in region.groupby("horizon_mois")
        }
    return resultats


def _prevalences(prev):
    mesures = ["prev_2024", "prev_2030"]
    moyennes = prev.groupby("patho", as_index=False)[mesures].mean()
    prev = pd.concat([moyennes.assign(region_nom=FRANCE), prev])
    prev["pathologie"] = prev.patho.map(PATHOLOGIES)
    return {
        territoire: enregistrements(groupe[["pathologie", *mesures]].sort_values("pathologie"))
        for territoire, groupe in prev.groupby("region_nom")
    }


def generer_dashboard_mensuel() -> str:
    full = pd.read_csv(OUT / "prevision_mensuelle_regions.csv")
    france = pd.read_csv(OUT / "prevision_mensuelle_france.csv")
    bt = pd.read_csv(OUT / "backtest_mensuel.csv")
    prev = pd.read_csv(OUT / "prevalence_projections_2030.csv")
    modeles = {nom: label for nom, label in MODELES.items() if f"pred_{nom}" in bt}

    cles = ["mois", "origine", "horizon_mois"]
    valeurs = ["lits_reel", *(f"pred_{nom}" for nom in modeles)]
    france_bt = bt.groupby(cles, as_index=False)[valeurs].sum()
    territoires = [FRANCE, *sorted(full.region_nom.unique().tolist())]
    bt = pd.concat([france_bt.assign(region_nom=FRANCE), bt]).sort_values("mois")
    full = pd.concat([france.assign(region_nom=FRANCE), full]).sort_values("mois")
    projections = {
        territoire: enregistrements(groupe[["mois", "lits_pred", "est_prevision", "scenario_bas", "scenario_haut"]].rename(columns={"mois": "date"}))
        for territoire, groupe in full.groupby("region_nom")
    }

    donnees = {
        "territoire_defaut": FRANCE, "territoires": territoires,
        "unite_horizon": "mois", "horizons": sorted(bt.horizon_mois.unique().tolist()),
        "modele_retenu": bt.modele.dropna().iloc[0], "modeles": modeles,
        "comparaisons": _comparaisons(bt[["region_nom", *cles, *valeurs]], territoires),
        "projections": projections,
        "contexte": [],
        "prevalences": _prevalences(prev),
    }
    analyses = """
      <article class="panel">
        <h3>Prévalence des maladies chroniques</h3>
        <p id="prevalences-contexte"></p>
        <div id="prevalences-chart" class="chart" role="img" aria-label="Prévalences observées en 2024 et extrapolées à 2030"></div>
      </article>
    """
    return generer_html(
        OUT / "dashboard_mensuel.html", "Occupation hospitalière avec le SARS-CoV-2",
        "Compare les observations et les valeurs prédites par région ou pour la France entière. La série couvre les patients positifs hospitalisés, tous motifs confondus.",
        donnees,
        "Timestral-209M prolonge la série historique de Santé publique France. Les courbes régionales affichent ses quantiles 10 % et 90 %. Le total national additionne les médianes régionales.",
        analyses,
        [("backtest_mensuel.csv", "Backtest régional (CSV)"),
         ("prevision_mensuelle_regions.csv", "Projection par région (CSV)"),
         ("prevision_mensuelle_france.csv", "Projection France (CSV)"),
         ("prevalence_projections_2030.csv", "Prévalences à 2030 (CSV)")],
        fraicheur=fraicheur_dashboard(OUT.parent, **{"Dernier mois observé": str(full.mois.max())}),
    )


if __name__ == "__main__":
    generer_dashboard_mensuel()
