"use strict";

const data = JSON.parse(document.getElementById("data").textContent);
const byId = id => document.getElementById(id);
const labels = ["Insuffisante", "Acceptable", "Satisfaisante"];
const statusColors = ["#be3f5c", "#ae6e14", "#167661"];
const color = {context: "#6944c8", reference: "#4c637f", observed: "#bd4260"};
const number = new Intl.NumberFormat("fr-FR");
const decimal = new Intl.NumberFormat("fr-FR", {maximumFractionDigits: 3, minimumFractionDigits: 3});
const percent = new Intl.NumberFormat("fr-FR", {style: "percent", maximumFractionDigits: 1});
const dateFormat = new Intl.DateTimeFormat("fr-FR", {day:"numeric", month:"short", year:"numeric", timeZone:"UTC"});
const dateTimeFormat = new Intl.DateTimeFormat("fr-FR", {day:"numeric", month:"short", year:"numeric", hour:"2-digit", minute:"2-digit", timeZone:"UTC"});
const monthFormat = new Intl.DateTimeFormat("fr-FR", {month:"long", year:"numeric", timeZone:"UTC"});
const formatDate = value => dateFormat.format(new Date(value.slice(0, 10) + "T00:00:00Z"));
const formatMonth = value => monthFormat.format(new Date(value.slice(0, 10) + "T00:00:00Z"));
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const chartConfig = {responsive: true, displayModeBar: false, scrollZoom: false};
const compact = window.matchMedia("(max-width:590px)");
const allDates = [...new Set(data.stocks.map(row => row.observation))].sort();
const lastReport = new Date(allDates.at(-1).slice(0, 10) + "T00:00:00Z");
lastReport.setUTCDate(lastReport.getUTCDate() + 18);
const seasons = [...new Set(data.stocks.map(row => row.saison))].sort();

function options(id, items) {
  byId(id).replaceChildren(...items.map(([value, name]) => new Option(name, value)));
}

options("product", [["all", "Toutes les formes"], ...Object.entries(data.produits)]);
options("season", [["all", "Toutes les saisons"], ["2022–23", "2022–23 · Archives"],
  ...seasons.map(season => [season, season])]);
options("horizon", data.horizons.map(days => [days, days + " jours"]));
options("box-product", [["all", "Total des six formes"], ...Object.entries(data.produits)]);
options("box-horizon", [...new Set(data.volumes_futur.map(row => row.horizon))]
  .sort((a, b) => a - b).map(months => [months, months + " mois"]));
byId("product-count").textContent = Object.keys(data.produits).length;
byId("report-count").textContent = allDates.length;
byId("last-report").textContent = formatDate(allDates.at(-1));

function plot(id, traces, extra = {}) {
  const {xaxis, yaxis, ...other} = extra;
  return Plotly.react(byId(id), traces, {
    paper_bgcolor: "white", plot_bgcolor: "white", font: {family: "system-ui", size: 11, color: "#424965"},
    margin: {l:compact.matches ? 47 : 68,r:compact.matches ? 13 : 25,t:12,b:66},
    separators: ", ",
    xaxis: {type:"date", gridcolor:"#eef0f5", fixedrange:true, ...xaxis},
    yaxis: {gridcolor:"#eef0f5", fixedrange:true, zeroline:false, ...yaxis},
    legend: {orientation:"h", y:-.22, font:{size:compact.matches ? 9 : 11}},
    hovermode:"closest", ...other,
  }, chartConfig);
}

function stats(rows, field, threshold) {
  const hit = rows.filter(row => row[field] >= threshold && row.observé);
  return {
    brier: mean(rows.map(row => (row[field] - row.observé) ** 2)),
    hit: hit.length,
    falseAlarms: rows.filter(row => row[field] >= threshold && !row.observé).length,
  };
}

function renderMetrics(rows, threshold) {
  const reference = stats(rows, "contexte", threshold);
  const context = stats(rows, "timestral", threshold);
  const events = rows.filter(row => row.observé);
  const brierDiff = context.brier - reference.brier;
  const result = Math.abs(brierDiff) < .0005 ? "Scores proches"
    : brierDiff < 0 ? "Score amélioré" : "Score dégradé";
  byId("brier").textContent = rows.length ? decimal.format(context.brier) : "—";
  byId("brier-note").textContent = rows.length ? result + " · logistique " + decimal.format(reference.brier) : "Aucune prévision pour ces filtres";
  byId("detected").textContent = events.length ? context.hit + " / " + events.length : "—";
  byId("detected-note").textContent = events.length ? "Logistique " + reference.hit + " / " + events.length : "Aucun cas insuffisant dans la sélection";
  byId("false-alarms").textContent = rows.length ? number.format(context.falseAlarms) : "—";
  byId("false-note").textContent = rows.length ? "Logistique " + reference.falseAlarms + " · seuil " + percent.format(threshold) : "Timestral-209M";
  byId("sample").textContent = number.format(rows.length);
  const uniqueCases = new Set(events.map(row => row.cible + row.produit)).size;
  byId("sample-note").textContent = events.length ? uniqueCases + " relevé(s) insuffisant(s) distinct(s)" : "Prévisions hors échantillon";
  const persistence = stats(rows.map(row => ({...row, persistence: Number(row.grossistes === 0)})), "persistence", threshold);
  byId("persistence").textContent = rows.length
    ? "Dernier état reconduit · Brier " + decimal.format(persistence.brier)
      + " · " + persistence.hit + " cas détecté(s) · " + persistence.falseAlarms + " fausse(s) alerte(s)"
    : "Le backtest commence à l’hiver 2024–25 sur les formes suivies.";
}

function forecast(rows, threshold, aggregate, newOnly) {
  const targetDates = new Map();
  for (const row of rows) {
    if (!targetDates.has(row.cible)) targetDates.set(row.cible, []);
    targetDates.get(row.cible).push(row);
  }
  const points = [...targetDates].sort(([left], [right]) => left.localeCompare(right));
  const x = points.map(([day]) => day);
  const details = points.map(([, group]) => {
    const origins = [...new Set(group.map(row => row.origine))].sort();
    const first = formatDate(origins[0]);
    const last = formatDate(origins.at(-1));
    return [group.length, first === last ? first : first + " au " + last];
  });
  const forecastSeries = [
    ["timestral", "Timestral-209M", color.context, "circle"],
    ["contexte", "Logistique · avec signaux", color.reference, "circle-open"],
    ["observé", "Relevé ANSM", color.observed, "diamond"],
  ];
  if (newOnly) forecastSeries.push(["degradation", "Prédit · nouvelles dégradations", "#167661", "square"]);
  const traces = forecastSeries.map(([field, name, shade, symbol]) => ({
    x, y: points.map(([, group]) => {
      const values = group.map(row => row[field]).filter(value => value !== null);
      return values.length ? mean(values) : null;
    }),
    name, mode:"markers", type:"scatter",
    marker:{color:shade, size:field === "observé" ? 8 : 9, symbol, line:{width:1.3}},
    customdata:details, hovertemplate:name + " %{y:.1%}<br>Départ %{customdata[1]}<br>%{customdata[0]} prévision(s)<extra></extra>",
  }));
  plot("forecast", traces, {
    yaxis:{range:[-.07,1.07], tickvals:[0,.25,.5,.75,1], tickformat:".0%"},
    shapes:rows.length ? [{type:"line", xref:"paper", x0:0,x1:1,y0:threshold,y1:threshold,line:{color:"#8e799e",width:1,dash:"dot"}}] : [],
    annotations:rows.length ? [] : [{text:"Aucune prévision pour ces filtres",xref:"paper",yref:"paper",x:.5,y:.5,showarrow:false}],
  });
  byId("chart-caption").textContent = aggregate
    ? "Par date, les cercles donnent la moyenne des probabilités et le losange donne la part des relevés en couverture insuffisante. Les alertes sont comptées sur chaque prévision individuelle."
    : "Pour le relevé ANSM, 100 % signifie « couverture insuffisante » et 0 % signifie « couverture acceptable ou satisfaisante ». Le trait pointillé marque le seuil d’alerte.";
}

function sourceCell(row, field, urlField) {
  const cell = document.createElement("td");
  const link = document.createElement("a");
  link.href = row[urlField];
  link.textContent = formatDate(row[field]);
  link.title = "Ouvrir le relevé ANSM";
  cell.append(link);
  return cell;
}

function renderTable(rows, newOnly) {
  byId("onset-column").hidden = !newOnly;
  byId("rows").replaceChildren(...[...rows].reverse().map(row => {
    const tr = document.createElement("tr");
    tr.append(sourceCell(row, "origine", "source_initiale"), sourceCell(row, "cible", "source_cible"));
    const values = [data.produits[row.produit], row.observé ? "Oui" : "Non", percent.format(row.timestral), percent.format(row.reference), percent.format(row.contexte)];
    if (newOnly) values.push(row.degradation === null ? "—" : percent.format(row.degradation));
    values.forEach(value => { tr.insertCell().textContent = value; });
    return tr;
  }));
}

function renderSignals(product, season) {
  const stocks = data.stocks.filter(row => (product === "all" || row.produit === product)
    && (season === "all" || row.saison === season));
  const originYear = season === "all" ? 2022 : Number(season.slice(0,4));
  const start = originYear + "-09-01";
  const end = season === "all" ? lastReport.toISOString().slice(0, 10) : (originYear + 1) + "-05-01";
  const xaxis = {range:[start,end]};
  const grouped = product === "all";
  const traces = labels.map((label, level) => {
    const rows = stocks.filter(row => row.grossistes === level);
    return {
      type:"scatter", mode:"markers", name:label,
      x:rows.map(row => row.observation),
      y:rows.map(row => grouped ? data.produits[row.produit] : level),
      marker:{size:grouped ? 8 : 10,color:statusColors[level], symbol:["diamond","square","circle"][level]},
      customdata:rows.map(row => formatDate(row.disponible)),
      hovertemplate:label + "<br>%{x|%d/%m/%Y}<br>Archive datée du %{customdata}<extra></extra>",
    };
  });
  plot("stocks", traces, {
    xaxis,
    yaxis:grouped
      ? {type:"category", categoryorder:"array", categoryarray:Object.values(data.produits).reverse(), automargin:true, tickfont:{size:compact.matches ? 8 : 10}}
      : {range:[-.5,2.5],tickvals:[0,1,2],ticktext:labels,automargin:true},
  });
  plot("ira", [{
    x:data.ira.map(row => row.semaine), y:data.ira.map(row => row.inc100), type:"scatter",
    mode:"lines", fill:"tozeroy", fillcolor:"rgba(22,118,97,.13)", line:{color:"#167661",width:2},
    hovertemplate:"%{y} / 100 000<extra></extra>",
  }], {xaxis, margin:{l:48,r:13,t:10,b:39}});
  const products = grouped ? Object.keys(data.produits) : [product];
  const palette = ["#6944c8","#167661","#bd4260","#936613","#506d9b","#9c54a0"];
  plot("volumes", products.map(id => {
    const series = data.volumes.filter(row => row.produit === id);
    return {
      x:series.map(row => row.mois), y:series.map(row => row.boites),
      type:"scatter", mode:"lines", name:data.produits[id], line:{color:palette[Object.keys(data.produits).indexOf(id)],width:2},
      hovertemplate:"%{y:,} boîtes<extra>%{fullData.name}</extra>",
    };
  }), {xaxis, showlegend:false, margin:{l:54,r:13,t:10,b:39}});
}

function monthlyTotals(rows, product, fields) {
  if (product !== "all") {
    return rows.filter(row => row.produit === product).sort((a, b) => a.mois.localeCompare(b.mois));
  }
  const byMonth = new Map();
  for (const row of rows) {
    if (!byMonth.has(row.mois)) {
      byMonth.set(row.mois, {mois:row.mois, dernier_mois_utilise:row.dernier_mois_utilise, count:0,
        ...Object.fromEntries(fields.map(field => [field, 0]))});
    }
    const total = byMonth.get(row.mois);
    total.count++;
    if (row.dernier_mois_utilise < total.dernier_mois_utilise) {
      total.dernier_mois_utilise = row.dernier_mois_utilise;
    }
    for (const field of fields) total[field] += row[field];
  }
  return [...byMonth.values()].filter(row => row.count === Object.keys(data.produits).length)
    .sort((a, b) => a.mois.localeCompare(b.mois));
}

function boxesTrace(rows, field, name, shade, dash = "solid") {
  return {
    x:rows.map(row => row.mois), y:rows.map(row => row[field]), type:"scatter", mode:"lines+markers",
    name, marker:{size:5}, line:{color:shade, width:2, dash},
    hovertemplate:"%{x|%m/%Y}<br>%{y:,} boîtes<extra>" + name + "</extra>",
  };
}

function renderBoxes() {
  const product = byId("box-product").value;
  const horizon = Number(byId("box-horizon").value);
  const history = monthlyTotals(data.volumes, product, ["boites"]);
  const forecasts = monthlyTotals(data.volumes_futur, product, ["saisonnier", "timestral"]);
  const tests = monthlyTotals(data.volumes_test.filter(row => row.horizon === horizon), product,
    ["observe", "saisonnier", "timestral"]);
  const month = data.volumes_futur.find(row => row.horizon === horizon)?.mois;
  const selected = forecasts.find(row => row.mois === month);
  const latest = history.at(-1);
  const boxCount = value => value === undefined ? "—" : number.format(Math.round(value));
  const error = field => tests.length ? mean(tests.map(row => Math.abs(row[field] - row.observe))) : undefined;
  byId("box-target").textContent = month ? "Timestral · " + formatMonth(month) : "Prévision Timestral";
  byId("box-seasonal").textContent = boxCount(selected?.saisonnier);
  byId("box-timestral").textContent = boxCount(selected?.timestral);
  byId("box-error-seasonal").textContent = boxCount(error("saisonnier"));
  byId("box-error-timestral").textContent = boxCount(error("timestral"));
  byId("box-test-count").textContent = tests.length + " mois évalués à cet horizon";
  byId("box-last").textContent = latest && selected
    ? "Dernière observation : " + formatMonth(latest.mois) + ". Les pointillés indiquent les prévisions."
      + (product === "all" ? " Le total additionne les médianes par forme." : " La zone ombrée montre les quantiles 10 % et 90 %.")
    : "Prévision indisponible pour cette sélection.";
  const intervalle = product === "all" ? [] : [
    { ...boxesTrace(forecasts, "timestral_q90", "", color.context), mode:"lines", line:{width:0},
      showlegend:false, hoverinfo:"skip" },
    { ...boxesTrace(forecasts, "timestral_q10", "Quantiles 10–90 %", color.context), mode:"lines",
      line:{width:0}, fill:"tonexty", fillcolor:"rgba(105,68,200,.14)", hoverinfo:"skip" },
  ];
  plot("box-forecast", [
    ...intervalle,
    boxesTrace(history.slice(-18), "boites", "Medic’AM observé", "#167661"),
    boxesTrace(forecasts, "saisonnier", "Prévision saisonnière", color.reference, "dot"),
    boxesTrace(forecasts, "timestral", "Timestral-209M", color.context, "dash"),
  ], {yaxis:{rangemode:"tozero"}});
  plot("box-backtest", [
    boxesTrace(tests, "observe", "Medic’AM observé", "#167661"),
    boxesTrace(tests, "saisonnier", "Prévision saisonnière", color.reference, "dot"),
    boxesTrace(tests, "timestral", "Timestral-209M", color.context, "dash"),
  ], {yaxis:{rangemode:"tozero"}});
}

function selection() {
  const product = byId("product").value;
  const season = byId("season").value;
  const horizon = Number(byId("horizon").value);
  const threshold = Number(byId("threshold").value) / 100;
  const newOnly = byId("new-only").checked;
  const rows = data.predictions.filter(row => (product === "all" || row.produit === product)
    && (season === "all" || row.saison === season) && row.horizon === horizon
    && (!newOnly || row.grossistes > 0));
  return {product, season, threshold, rows, newOnly};
}

function falseAlarmsByDetection(rows, field) {
  const ordered = [...rows].sort((a, b) => b[field] - a[field]);
  let falseAlarms = 0;
  const counts = [];
  // Un même score implique une même décision pour toutes ses lignes.
  for (let start = 0; start < ordered.length;) {
    let end = start + 1;
    while (end < ordered.length && ordered[end][field] === ordered[start][field]) end++;
    const events = ordered.slice(start, end).filter(row => row.observé).length;
    falseAlarms += end - start - events;
    for (let event = 0; event < events; event++) counts.push(falseAlarms);
    start = end;
  }
  return counts;
}

function renderOnset(rows, threshold, newOnly) {
  byId("onset").hidden = !newOnly;
  if (!newOnly) return;
  const paired = rows.filter(row => row.degradation !== null);
  const cases = paired.filter(row => row.observé).length;
  const previous = stats(paired, "timestral", threshold);
  const dedicated = stats(paired, "degradation", threshold);
  byId("onset-threshold").textContent = paired.length
    ? `Au seuil de ${percent.format(threshold)}, Timestral détecte ${previous.hit} cas sur ${cases} avec ${previous.falseAlarms} fausses alertes. La régression dédiée en détecte ${dedicated.hit} avec ${dedicated.falseAlarms} fausses alertes. Comparaison sur ${paired.length} prévisions communes.`
    : "Aucune prévision du modèle dédié pour cette sélection.";
  const dedicatedCounts = falseAlarmsByDetection(paired, "degradation");
  const previousCounts = falseAlarmsByDetection(paired, "timestral");
  byId("onset-budget").textContent = cases
    ? "Fausses alertes selon le nombre de cas détectés (Timestral / régression dédiée) · "
      + dedicatedCounts.slice(0, 3).map((count, index) =>
        `${index + 1} cas : ${previousCounts[index]} / ${count}`).join(" · ")
    : "";
}

function renderFraicheur() {
  const f = data.fraicheur || {};
  byId("fresh-generation").textContent = f.date_generation
    ? dateTimeFormat.format(new Date(f.date_generation)) + " UTC" : "—";
  byId("fresh-ansm").textContent = f.dernier_releve_ansm
    ? formatDate(f.dernier_releve_ansm) + " (publié le " + formatDate(f.dernier_releve_disponible) + ")"
    : "—";
  byId("fresh-medicam").textContent = f.dernier_mois_medicam ? formatMonth(f.dernier_mois_medicam) : "—";
  const revision = byId("fresh-revision");
  if (f.revision_modele) {
    revision.textContent = String(f.revision_modele).slice(0, 12);
    revision.title = "Révision complète : " + f.revision_modele + " (outputs/manifest.json)";
  } else {
    revision.textContent = "—";
    revision.title = "Manifeste absent : révision du modèle inconnue";
  }
}

function renderAlertes() {
  const seuil = data.seuil_alerte ?? 0.5;
  byId("alertes-seuil").textContent = percent.format(seuil);
  const rows = data.alertes || [];
  const nouvelles = rows.filter(row => row.type_alerte === "nouvelle dégradation").length;
  byId("alertes-rows").replaceChildren(...rows.map(row => {
    const tr = document.createElement("tr");
    const score = tr.insertCell();
    score.textContent = percent.format(row.timestral);
    score.className = "score";
    const type = tr.insertCell();
    const badge = document.createElement("span");
    const isNew = row.type_alerte === "nouvelle dégradation";
    badge.className = "badge " + (isNew ? "badge-new" : "badge-known");
    badge.textContent = isNew ? "Nouvelle dégradation" : "Tension connue";
    type.append(badge);
    tr.insertCell().textContent = data.produits[row.produit] ?? row.produit;
    tr.insertCell().textContent = row.horizon + " j";
    tr.insertCell().textContent = formatDate(row.origine);
    const cible = tr.insertCell();
    if (row.source_cible) {
      const link = document.createElement("a");
      link.href = row.source_cible;
      link.textContent = formatDate(row.cible);
      link.title = "Ouvrir le relevé ANSM visé";
      cible.append(link);
    } else {
      cible.textContent = formatDate(row.cible);
    }
    tr.insertCell().textContent = row.observé ? "Insuffisance" : "Couverture suffisante";
    return tr;
  }));
  byId("alertes-caption").textContent = rows.length
    ? rows.length + " alerte(s) au seuil de " + percent.format(seuil) + ", triées par score décroissant — dont "
      + nouvelles + " nouvelle(s) dégradation(s) et " + (rows.length - nouvelles) + " tension(s) connue(s)."
    : "Aucune prévision hors échantillon n’atteint le seuil d’alerte de " + percent.format(seuil) + ".";
}

function renderBacktest(refreshTable) {
  const {product, threshold, rows, newOnly} = selection();
  byId("threshold-value").textContent = percent.format(threshold);
  renderMetrics(rows, threshold);
  renderOnset(rows, threshold, newOnly);
  forecast(rows, threshold, product === "all", newOnly);
  if (refreshTable) renderTable(rows, newOnly);
}

function renderAll() {
  const {product, season} = selection();
  renderBacktest(true);
  renderSignals(product, season);
}

["product", "season"].forEach(id => byId(id).addEventListener("change", renderAll));
byId("horizon").addEventListener("change", () => renderBacktest(true));
byId("new-only").addEventListener("change", () => renderBacktest(true));
byId("threshold").addEventListener("input", () => renderBacktest(false));
["box-product", "box-horizon"].forEach(id => byId(id).addEventListener("change", renderBoxes));
compact.addEventListener("change", () => { renderAll(); renderBoxes(); });
renderFraicheur();
renderAlertes();
renderAll();
renderBoxes();
