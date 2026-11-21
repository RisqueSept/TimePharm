"""Composants partagés des tableaux HTML autonomes."""

import json
from html import escape
from pathlib import Path

from plotly.offline import get_plotlyjs


MODELES = {
    "Timestral": "Timestral-209M · Mistral AI",
    "Persistance": "Dernière valeur observée",
    "A_tendance": "Tendance de long terme",
    "B_ratio_x_pop": "Tendance par 1 000 habitants",
    "C_tendance_recente": "Tendance des cinq dernières années",
    "Saisonnier": "Même mois de l’année précédente",
    "Ridge": "Régression Ridge",
    "HistGB": "HistGradientBoosting",
    "XGBoost": "XGBoost",
}


def enregistrements(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _options(options):
    return "".join(f'<option value="{escape(str(k))}">{escape(v)}</option>' for k, v in options)


def generer_html(dest, titre, accroche, donnees, note_projection, analyses, liens):
    dossiers = Path(__file__).resolve().parent
    css = (dossiers / "dashboard.css").read_text(encoding="utf-8")
    script = (dossiers / "dashboard.js").read_text(encoding="utf-8")
    donnees_json = json.dumps(donnees, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    unite = donnees["unite_horizon"]
    horizons = _options(
        (h, f"{h} {unite}{'s' if unite == 'an' and h > 1 else ''}")
        for h in donnees["horizons"]
    )
    modele_retenu = donnees["modele_retenu"]
    modeles = _options(
        (nom, label + (" · projection" if nom == modele_retenu else ""))
        for nom, label in donnees["modeles"].items()
    )
    territoires = ""
    if donnees["territoires"]:
        regions = _options((nom, nom) for nom in donnees["territoires"])
        territoires = f'<label>Territoire<select id="territoire">{regions}</select></label>'
    liens_html = "".join(f'<a href="{escape(url)}">{escape(texte)}</a>' for url, texte in liens)
    html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(titre)} | Timestral-209M</title>
  <style>{css}</style>
</head>
<body>
  <header class="top"><div class="shell mast">
    <a class="brand" href="#">Prévisions santé</a>
    <nav aria-label="Navigation du tableau de bord">
      <a href="#comparaison">Réel et prédit</a>
      <a href="#projection">Projection</a>
      <a href="#contexte">Contexte et données</a>
    </nav>
  </div></header>
  <main class="shell">
    <header class="intro"><h1>{escape(titre)}</h1><p>{escape(accroche)}</p></header>

    <section id="comparaison" class="panel" aria-labelledby="titre-comparaison">
      <div class="section-head">
        <h2 id="titre-comparaison">Observé et prédit</h2>
        <p>Les prédictions du backtest ont été calculées avant les observations auxquelles elles sont comparées.</p>
      </div>
      <div class="filters">
        {territoires}
        <label>Horizon<select id="horizon">{horizons}</select></label>
        <label class="model-filter">Modèle<select id="modele">{modeles}</select></label>
      </div>
      <div class="metrics" aria-live="polite">
        <div><span>Écart moyen en valeur absolue</span><strong id="mae">—</strong><small>Taille moyenne de l’écart</small></div>
        <div><span>Biais moyen</span><strong id="biais">—</strong><small id="direction">Sens de l’écart</small></div>
        <div><span>Observations comparées</span><strong id="n">—</strong><small id="periode"></small></div>
      </div>
      <div id="compare-chart" class="chart compare" role="img" aria-label="Chronologie du backtest et écarts entre prédictions et observations"></div>
      <p class="hint">Un écart positif correspond à une surestimation. Survole la prévision ou une barre d’écart pour voir sa date de départ.</p>
      <details class="detail">
        <summary>Voir les valeurs du backtest</summary>
        <div class="table-wrap">
          <table>
            <thead><tr><th scope="col">Période observée</th><th scope="col">Date de départ</th><th scope="col">Observé</th><th scope="col">Prédit</th><th scope="col">Écart</th></tr></thead>
            <tbody id="comparaison-lignes"></tbody>
          </table>
        </div>
      </details>
    </section>

    <section id="projection" class="panel" aria-labelledby="titre-projection">
      <div class="section-head">
        <h2 id="titre-projection">Historique et projection</h2>
        <p>{escape(note_projection)}</p>
        <p class="hint" id="projection-contexte"></p>
      </div>
      <div id="projection-chart" class="chart" role="img" aria-label="Historique observé et projection après la dernière observation"></div>
    </section>

    <section id="contexte" class="context" aria-labelledby="titre-contexte">
      <h2 id="titre-contexte">Contexte et données</h2>
      {analyses}
      <div class="downloads" aria-label="Télécharger les données">{liens_html}</div>
    </section>
  </main>
  <script type="application/json" id="dashboard-data">{donnees_json}</script>
  <script>{get_plotlyjs()}</script>
  <script>{script}</script>
</body>
</html>
"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    print(f"[dashboard] -> {dest} ({dest.stat().st_size // 1024} Ko, HTML autonome)")
    return str(dest)
