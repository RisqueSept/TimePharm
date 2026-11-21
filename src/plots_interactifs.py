"""Tableau annuel des capacités SAE et des scénarios MCO."""

from pathlib import Path

import pandas as pd

from dashboard import MODELES, enregistrements, fraicheur_dashboard, generer_html

BASE = Path(__file__).resolve().parents[1]
PROCESSED = BASE / "data" / "processed"
OUT = BASE / "outputs"
DISCIPLINES = [
    ("lits_mco", "MCO", "#245bba"),
    ("lits_ssr", "SSR / SMR", "#087472"),
    ("lits_psy", "Psychiatrie", "#9467bd"),
    ("lits_usld", "Longue durée", "#a7633c"),
]


def _contexte(hist, prev, scenarios):
    par_discipline = [{
        "type": "scatter", "mode": "lines", "stackgroup": "disciplines",
        "x": hist.annee.tolist(), "y": hist[colonne].tolist(), "name": label,
        "line": {"color": couleur, "width": 1},
    } for colonne, label, couleur in DISCIPLINES]
    charge = [{
        "type": "scatter", "mode": "lines+markers", "name": "Capacité MCO projetée",
        "x": prev.annee.tolist(), "y": prev.lits_mco.tolist(), "line": {"color": "#245bba"},
    }]
    for variation, groupe in scenarios[scenarios.taux_occupation_cible.eq(0.85)].groupby("variation_activite"):
        charge.append({
            "type": "scatter", "mode": "lines", "x": groupe.annee.tolist(),
            "y": groupe.lits_ouverts_equivalents.tolist(),
            "name": f"Activité {variation:+.0%}" if variation else "Activité de référence",
            "line": {"dash": "dot"},
        })
    demographie = [
        {"type": "scatter", "mode": "lines+markers", "name": "Lits pour 1 000 habitants",
         "x": hist.annee.tolist(), "y": hist.lits_pour_1000.tolist()},
        {"type": "scatter", "mode": "lines+markers", "name": "Part des 75 ans et plus",
         "x": hist.annee.tolist(), "y": hist.part_75plus.tolist(), "yaxis": "y2"},
    ]
    return [
        {"id": "mco-chart", "traces": charge, "axe_y": "Lits", "axe_temps": {"tickformat": "d"}},
        {"id": "disciplines-chart", "traces": par_discipline, "axe_y": "Lits", "axe_temps": {"tickformat": "d"}},
        {"id": "demographie-chart", "traces": demographie, "axe_y": "Lits pour 1 000 habitants",
         "axe_temps": {"tickformat": "d"},
         "layout": {"yaxis2": {"overlaying": "y", "side": "right", "ticksuffix": " %", "showgrid": False},
                    "margin": {"t": 52, "b": 52, "l": 70, "r": 65}}},
    ]


def _table_disciplines(prev):
    colonnes = ["annee", "lits_pred", *(colonne for colonne, _, _ in DISCIPLINES)]
    titres = ["Année", "Total", *(label for _, label, _ in DISCIPLINES)]
    entetes = "".join(f'<th scope="col">{titre}</th>' for titre in titres)
    lignes = []
    for annee, *valeurs in prev[colonnes].itertuples(index=False, name=None):
        cellules = "".join(f"<td>{valeur:,}</td>".replace(",", " ") for valeur in valeurs)
        lignes.append(f"<tr><td>{annee}</td>{cellules}</tr>")
    return f"""
        <details class="detail">
          <summary>Voir les capacités projetées par discipline</summary>
          <div class="table-wrap">
            <table><thead><tr>{entetes}</tr></thead><tbody>{''.join(lignes)}</tbody></table>
          </div>
        </details>
    """


def generer_dashboard() -> str:
    hist = pd.read_csv(PROCESSED / "dataset_annuel.csv").sort_values("annee")
    prev = pd.read_csv(OUT / "prediction_lits_2026_2035.csv").sort_values("annee")
    bt = pd.read_csv(OUT / "backtest_predictions.csv").sort_values("annee")
    scenarios = pd.read_csv(OUT / "scenarios_capacite_mco.csv")

    territoire = "France"
    historique = hist[["annee", "lits_total"]].rename(columns={"annee": "date", "lits_total": "lits_pred"})
    projection = prev[["annee", "lits_pred", "scenario_bas", "scenario_haut"]].rename(columns={"annee": "date"})
    donnees = {
        "territoire_defaut": territoire, "territoires": [],
        "unite_horizon": "an", "horizons": sorted(bt.horizon_ans.unique().tolist()),
        "modele_retenu": bt.modele.iloc[0],
        "modeles": {nom: label for nom, label in MODELES.items() if f"pred_{nom}" in bt},
        "comparaisons": {territoire: {
            int(h): enregistrements(groupe.rename(columns={"annee": "date"}))
            for h, groupe in bt.groupby("horizon_ans")
        }},
        "projections": {territoire: enregistrements(pd.concat([
            historique.assign(est_prevision=0), projection.assign(est_prevision=1)
        ]))},
        "contexte": _contexte(hist, prev, scenarios),
    }
    analyses = f"""
      <article class="panel">
        <h3>Charge en MCO selon l’activité</h3>
        <p>Les lignes pointillées montrent les lits ouverts équivalents à 85 % d’occupation moyenne. Le CSV propose aussi les hypothèses à 80 % et 90 %.</p>
        <div id="mco-chart" class="chart" role="img" aria-label="Scénarios de charge et capacité MCO projetée"></div>
      </article>
      <article class="panel">
        <h3>Capacités observées par discipline</h3>
        <div id="disciplines-chart" class="chart" role="img" aria-label="Lits observés par discipline"></div>
        {_table_disciplines(prev)}
      </article>
      <article class="panel">
        <h3>Capacité rapportée à la population</h3>
        <div id="demographie-chart" class="chart" role="img" aria-label="Lits pour 1 000 habitants et part des 75 ans et plus"></div>
      </article>
    """
    return generer_html(
        OUT / "dashboard_lits.html", "Capacité hospitalière en France",
        "Lits d’hospitalisation complète recensés par la DREES. Les scénarios MCO permettent aussi de comparer la capacité projetée et la charge selon l’activité.",
        donnees,
        "La médiane de Timestral-209M est prolongée jusqu’en 2035. La bande montre ses quantiles 10 % et 90 %. Le backtest couvre les horizons de 1 à 3 ans.",
        analyses,
        [("backtest_predictions.csv", "Backtest annuel (CSV)"),
         ("prediction_lits_2026_2035.csv", "Projection nationale (CSV)"),
         ("scenarios_capacite_mco.csv", "Scénarios MCO nationaux (CSV)"),
         ("scenarios_capacite_mco_regions.csv", "Scénarios MCO par territoire (CSV)")],
        fraicheur=fraicheur_dashboard(BASE, **{"Dernière année SAE observée": int(hist.annee.max())}),
    )


if __name__ == "__main__":
    generer_dashboard()
