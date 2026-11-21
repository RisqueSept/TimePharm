"use strict";

const data = JSON.parse(document.getElementById("dashboard-data").textContent);
const element = id => document.getElementById(id);
const territoire = element("territoire");
const horizon = element("horizon");
const modele = element("modele");
modele.value = data.modele_retenu;

const couleur = {
  observe: "#087472",
  predit: "#245bba",
  surestimation: "#b44f26",
  sousEstimation: "#53798d",
  grille: "#e4ebf0",
};
const entier = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });
const signe = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0, signDisplay: "exceptZero" });
const mois = new Intl.DateTimeFormat("fr-FR", { month: "short", year: "numeric", timeZone: "UTC" });
const formatDate = valeur => data.unite_horizon === "mois" ? mois.format(new Date(valeur)) : String(valeur);
const config = { responsive: true, displayModeBar: false, locale: "fr" };

function axesTemps() {
  return data.unite_horizon === "mois"
    ? { type: "date", tickformat: "%m/%Y" }
    : { type: "linear", tickformat: "d", dtick: 1 };
}

function miseEnPage(height) {
  return {
    height,
    margin: { t: 52, b: 52, l: 65, r: 12 },
    font: { color: "#263d51", family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 12 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    hovermode: "closest",
    separators: ", ",
    legend: { orientation: "h", x: 0, y: 1.04, yanchor: "bottom", font: { size: 12 } },
  };
}

function axeLits(titre) {
  return { title: { text: titre, font: { size: 12 } }, tickformat: "~s", gridcolor: couleur.grille, automargin: true };
}

function ligne(rows, cle, nom, color, tirets, alt = null, opacite = 1) {
  return {
    type: "scatter", mode: "lines+markers", name: nom,
    x: rows.map(row => row.date), y: rows.map(row => row[cle]),
    customdata: rows.map(row => [
      formatDate(row.date),
      entier.format(row[cle]),
      row.origine === undefined ? "" : formatDate(row.origine),
    ]),
    line: { color, width: 2.7, dash: tirets },
    marker: { size: 6 },
    opacity: opacite,
    hovertemplate: "<b>%{customdata[0]}</b><br>%{fullData.name} · %{customdata[1]} lits" +
      (alt || "") + "<extra></extra>",
  };
}

// Chantier #13 : horizons hors validation (au-delà de l'horizon du backtest).
const DELAI_VALIDATION = { an: 3, mois: 12 };

function estHorsValidation(date, derniere, unite, delai) {
  if (unite === "an") {
    const d = typeof date === "number" ? date : new Date(date).getUTCFullYear();
    const o = typeof derniere === "number" ? derniere : new Date(derniere).getUTCFullYear();
    return d - o > delai;
  }
  const d = new Date(date), o = new Date(derniere);
  return (d.getUTCFullYear() - o.getUTCFullYear()) * 12 + (d.getUTCMonth() - o.getUTCMonth()) > delai;
}

function legendeHorsValidation(afficher) {
  let legende = document.getElementById("legende-projection");
  if (!legende) {
    legende = document.createElement("p");
    legende.id = "legende-projection";
    legende.className = "legende-projection";
    const valide = document.createElement("span");
    valide.className = "chip";
    valide.textContent = "dans l’horizon de validation";
    const hors = document.createElement("span");
    hors.className = "chip chip-hors-validation";
    hors.textContent = "hors validation";
    legende.append(valide, hors);
    element("projection-contexte").after(legende);
  }
  legende.style.display = afficher ? "" : "none";
}

function diagrammeComparaison() {
  const nomTerritoire = territoire ? territoire.value : data.territoire_defaut;
  const lignes = data.comparaisons[nomTerritoire][horizon.value];
  const nomModele = modele.value;
  const rows = lignes.map(row => ({
    ...row,
    predit: row["pred_" + nomModele],
    ecart: row["pred_" + nomModele] - row.lits_reel,
  }));
  const n = rows.length;
  const biais = rows.reduce((total, row) => total + row.ecart, 0) / n;
  const mae = rows.reduce((total, row) => total + Math.abs(row.ecart), 0) / n;
  element("mae").textContent = entier.format(mae) + " lits";
  element("biais").textContent = signe.format(biais) + " lits";
  element("direction").textContent = Math.round(Math.abs(biais)) === 0
    ? "Biais moyen proche de zéro"
    : (biais > 0 ? "Le modèle surestime en moyenne" : "Le modèle sous-estime en moyenne");
  element("n").textContent = entier.format(n);
  element("periode").textContent = [rows[0].date, rows.at(-1).date].map(formatDate).join(" → ");

  const observations = ligne(rows, "lits_reel", "Observé", couleur.observe, "solid");
  const prediction = ligne(rows, "predit", "Prédit", couleur.predit, "dash",
    "<br>Calculé en %{customdata[2]}");
  const valeurs = {
    type: "bar", x: rows.map(row => row.date), y: rows.map(row => row.ecart), yaxis: "y2",
    name: "Écart", showlegend: false,
    marker: { color: rows.map(row => row.ecart > 0 ? couleur.surestimation : couleur.sousEstimation) },
    customdata: rows.map(row => [formatDate(row.date), signe.format(row.ecart), formatDate(row.origine)]),
    hovertemplate: "<b>%{customdata[0]}</b><br>Prédit − observé · %{customdata[1]} lits" +
      "<br>Calculé en %{customdata[2]}<extra></extra>",
  };
  Plotly.react(element("compare-chart"), [observations, prediction, valeurs], {
    ...miseEnPage(460),
    xaxis: { ...axesTemps(), anchor: "y2", showgrid: false, automargin: true },
    yaxis: { ...axeLits("Lits"), domain: [0.34, 1], anchor: "x" },
    yaxis2: { ...axeLits("Écart en lits"), domain: [0, 0.24], anchor: "x",
      zerolinecolor: "#64788a", zerolinewidth: 1.5 },
  }, config);

  const table = document.createDocumentFragment();
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const valeur of [formatDate(row.date), formatDate(row.origine), entier.format(row.lits_reel),
      entier.format(row.predit), signe.format(row.ecart)]) {
      const td = document.createElement("td");
      td.textContent = valeur;
      tr.append(td);
    }
    table.append(tr);
  }
  element("comparaison-lignes").replaceChildren(table);
}

function diagrammeProjection() {
  const nomTerritoire = territoire ? territoire.value : data.territoire_defaut;
  const rows = data.projections[nomTerritoire];
  const obs = rows.filter(row => !row.est_prevision);
  const futurs = rows.filter(row => row.est_prevision);
  const derniereDate = obs.at(-1).date;
  const delai = DELAI_VALIDATION[data.unite_horizon] ?? 3;
  const hors = row => row.hors_validation != null
    ? Number(row.hors_validation) === 1
    : estHorsValidation(row.date, derniereDate, data.unite_horizon, delai);
  const valides = futurs.filter(row => !hors(row));
  const horsValidation = futurs.filter(hors);
  const traces = [];
  const bande = (segment, opacite, nom) => {
    if (!segment.length || segment[0].scenario_bas == null) return [];
    const base = {
      type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip",
      x: segment.map(row => row.date), opacity: opacite,
    };
    return [
      { ...base, y: segment.map(row => row.scenario_haut), showlegend: false },
      { ...base, y: segment.map(row => row.scenario_bas), fill: "tonexty",
        fillcolor: "rgba(36, 91, 186, 0.14)", name: nom },
    ];
  };
  traces.push(...bande(valides, 1, "Timestral · quantiles 10–90 %"));
  traces.push(...bande(horsValidation, 0.35, "Quantiles 10–90 % · hors validation"));
  traces.push(ligne(obs, "lits_pred", "Observé", couleur.observe, "solid"));
  if (valides.length) traces.push(ligne(valides, "lits_pred", "Projection", couleur.predit, "dash"));
  if (horsValidation.length) {
    traces.push(ligne(horsValidation, "lits_pred", "Projection · hors validation", couleur.predit, "dash", null, 0.35));
  }
  element("projection-contexte").textContent =
    "Dernière observation en " + formatDate(derniereDate) +
    " · Modèle de projection : " + data.modeles[data.modele_retenu] +
    (horsValidation.length
      ? " · au-delà de " + formatDate(horsValidation[0].date) + " : hors validation (grisé)"
      : "");
  legendeHorsValidation(horsValidation.length > 0);
  const temps = axesTemps();
  delete temps.dtick;
  Plotly.react(element("projection-chart"), traces, {
    ...miseEnPage(380),
    xaxis: { ...temps, showgrid: false, automargin: true },
    yaxis: axeLits("Lits"),
    shapes: [{ type: "line", x0: derniereDate, x1: derniereDate, y0: 0, y1: 1,
      yref: "paper", line: { color: "#7d90a0", width: 1, dash: "dot" } }],
  }, config);
}

function diagrammePrevalences() {
  if (!data.prevalences) return;
  const nomTerritoire = territoire.value;
  const rows = data.prevalences[nomTerritoire] ?? [];
  element("prevalences-contexte").textContent = nomTerritoire === data.territoire_defaut
    ? "Moyenne des régions, sans pondération par la population. Extrapolation à partir des tendances 2015–2024."
    : nomTerritoire + " · extrapolation à partir des tendances 2015–2024.";
  const traces = [["prev_2024", "Observé en 2024", couleur.observe],
    ["prev_2030", "Timestral · 2030", couleur.predit]].map(([cle, nom, color]) => ({
    type: "bar", orientation: "h", name: nom,
    y: rows.map(row => row.pathologie), x: rows.map(row => row[cle]),
    marker: { color }, hovertemplate: "%{y}<br>%{x:.1f} %<extra>" + nom + "</extra>",
  }));
  Plotly.react(element("prevalences-chart"), traces, {
    ...miseEnPage(360), barmode: "group",
    xaxis: { title: { text: "Prévalence (%)", font: { size: 12 } }, gridcolor: couleur.grille, automargin: true },
    yaxis: { automargin: true, tickfont: { size: 11 }, autorange: "reversed" },
    margin: { t: 52, b: 52, l: 12, r: 15 },
  }, config);
}

for (const chart of data.contexte) {
  Plotly.newPlot(element(chart.id), chart.traces, {
    ...miseEnPage(chart.hauteur ?? 340),
    xaxis: { ...chart.axe_temps, automargin: true },
    yaxis: axeLits(chart.axe_y),
    ...chart.layout,
  }, config);
}

horizon.addEventListener("change", diagrammeComparaison);
modele.addEventListener("change", diagrammeComparaison);
if (territoire) {
  territoire.addEventListener("change", () => {
    diagrammeComparaison();
    diagrammeProjection();
    diagrammePrevalences();
  });
}
diagrammeComparaison();
diagrammeProjection();
diagrammePrevalences();
