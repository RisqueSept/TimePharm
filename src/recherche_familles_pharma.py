"""Benchmark de familles de modèles sous deux scénarios de publication."""

import numpy as np
import pandas as pd

from .inference_timestral import MistralTimestral, cycle
from .recherche_archives_pharma import dates_avec_prescripteurs, dates_documentees, mois_documentes
from .recherche_delai_pharma import SAISONNIER, TIMESTRAL, poids_du_melange
from .recherche_signaux_pharma import lire_benchmark
from .recherche_volumes_pharma import (
    DEBUT_VERIFICATION,
    FIN_DEV,
    charger,
    scores,
    timestral,
)


CHRONOS = "amazon/chronos-2"
REVISION_CHRONOS = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def charger_chronos(racine):
    from chronos import Chronos2Pipeline

    dossier = racine / "data" / "models" / "Chronos-2"
    local = all((dossier / fichier).is_file() for fichier in ("config.json", "model.safetensors"))
    return Chronos2Pipeline.from_pretrained(
        str(dossier) if local else CHRONOS, revision=None if local else REVISION_CHRONOS,
        local_files_only=local, device_map="cpu",
    )


def chronos_par_horizon(moteur, table, coupures, total):
    previsions = {}
    for horizon, cibles in coupures.groupby(
        [cible.ordinal - connu.ordinal for cible, connu in coupures.items()]
    ):
        entrees = []
        for cible, dernier in cibles.items():
            contexte = table.loc[:dernier].tail(60)
            dates = pd.period_range(contexte.index[0], cible, freq="M")
            cos_sin = dict(zip(("sin_mois", "cos_mois"), cycle(dates.month)))
            entrees.append({
                "target": (contexte.sum(axis=1) if total else contexte).to_numpy(dtype=np.float32).T,
                "past_covariates": {nom: vals[:-horizon] for nom, vals in cos_sin.items()},
                "future_covariates": {nom: vals[-horizon:] for nom, vals in cos_sin.items()},
            })
        quantiles, _ = moteur.predict_quantiles(
            entrees, prediction_length=horizon, quantile_levels=[0.5],
            batch_size=32, cross_learning=False,
        )
        for cible, q in zip(cibles.index, quantiles, strict=True):
            previsions[cible] = round(float(np.clip(q.numpy()[:, -1, 0], 0, None).sum()))
    return pd.Series(previsions).reindex(coupures.index)


def holt_winters(table, coupures):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    total = table.sum(axis=1)
    predictions = []
    for cible, dernier in coupures.items():
        historique = total.loc[:dernier].tail(60)
        modele = ExponentialSmoothing(np.log(historique), trend="add", damped_trend=True,
                                      seasonal="add", seasonal_periods=12)
        ajuste = modele.fit(use_brute=False)
        pas = cible.ordinal - dernier.ordinal
        dernier_point = ajuste.forecast(pas).iloc[-1]
        predictions.append(round(float(np.exp(dernier_point))))
    return pd.Series(predictions, index=coupures.index)


def scores_hiver(predictions, modeles):
    stats = scores(predictions, modeles)
    hiver = predictions.loc[predictions.index.month.isin([10, 11, 12, 1, 2, 3])]
    if not hiver.empty:
        stats["wape_hiver_pct"] = stats.modele.map(scores(hiver, modeles).set_index("modele").wape_pct)
    return stats


def benchmark(table, lignes, premier_test, dernier_dev, moteur, scenario, poids_archives=None):
    lignes = lignes.copy()
    coupures = lignes.dernier_mois_utilise
    for nom, total in (("chronos_formes", False), ("chronos_total", True)):
        lignes[nom] = chronos_par_horizon(moteur, table, coupures, total)
    lignes["holt_winters"] = holt_winters(table, coupures)
    lignes["timestral_chronos"] = np.rint((lignes.timestral + lignes.chronos_formes) / 2)
    references = ["reference_actuelle", "timestral"]
    if poids_archives is not None:
        lignes["reference_actuelle"] = np.rint(lignes.timestral * poids_archives
                                               + lignes.saisonnier * (1 - poids_archives))
    variantes = ["chronos_formes", "chronos_total", "holt_winters", "timestral_chronos", *references]
    communes = lignes.dropna(subset=variantes)
    stats_dev = scores_hiver(communes.loc[:dernier_dev], variantes)
    candidat = stats_dev.modele.iloc[0]
    stats_test = scores_hiver(communes.loc[premier_test:], list(dict.fromkeys([candidat, *references])))
    sorties = pd.concat([
        stats_dev.assign(periode="developpement"),
        stats_test.assign(periode="verification_reutilisee"),
    ]).assign(scenario=scenario, candidat_retenu=candidat)
    return sorties, lignes.assign(scenario=scenario)


def donnees_archives(racine, table, jours, developpement, moteur):
    destination = racine / "outputs" / "recherche_pharma"
    comparaison = pd.read_csv(destination / "predictions_dates_archives.csv")
    comparaison = comparaison.query("archives == 'atc_et_prescripteurs' and jour_origine == 21").copy()
    comparaison["mois"] = pd.PeriodIndex(comparaison.mois, freq="M")
    comparaison["dernier_mois_utilise"] = pd.PeriodIndex(comparaison.dernier_mois_utilise, freq="M")
    comparaison = comparaison.set_index("mois")
    editions = dates_avec_prescripteurs(racine, table, dates_documentees(racine))
    origine_test = (DEBUT_VERIFICATION - 1).start_time + pd.Timedelta(days=20)
    dernier_dev = mois_documentes(table, editions, origine_test).index[-1]
    dates_dev = developpement.loc["2020-01":dernier_dev]
    from .recherche_archives_pharma import predictions_aux_dates

    archives_dev = predictions_aux_dates(table, jours, editions, dates_dev, 0.5, moteur, (21,))
    archives_dev["mois"] = pd.PeriodIndex(archives_dev.mois, freq="M")
    archives_dev["dernier_mois_utilise"] = pd.PeriodIndex(archives_dev.dernier_mois_utilise, freq="M")
    lignes = pd.concat([archives_dev.set_index("mois"), comparaison]).sort_index()
    lignes = lignes.rename(columns={"timestral_archives": "timestral", "saisonnier_archives": "saisonnier"})
    return lignes[["observe", "dernier_mois_utilise", "timestral", "saisonnier"]], dernier_dev


def experimenter_familles(racine):
    destination = racine / "outputs" / "recherche_pharma"
    moteur_timestral = MistralTimestral()
    moteur_chronos = charger_chronos(racine)
    table, jours, _, _ = charger(racine)
    developpement = lire_benchmark(destination, "developpement.csv")
    verification = lire_benchmark(destination, "verification.csv")
    lignes = pd.concat([developpement, verification]).sort_index()
    poids = poids_du_melange(developpement)
    lignes = lignes.assign(reference_actuelle=np.rint(lignes[TIMESTRAL] * poids
                                                       + lignes[SAISONNIER] * (1 - poids)))
    lignes = lignes.rename(columns={TIMESTRAL: "timestral"})
    print("Comparaison à délai simulé", flush=True)
    s1, p1 = benchmark(table, lignes[["observe", "dernier_mois_utilise", "timestral", "reference_actuelle"]],
                       DEBUT_VERIFICATION, FIN_DEV, moteur_chronos, "delai_fixe")
    print(s1.to_string(index=False), flush=True)
    print("Comparaison avec les archives documentées", flush=True)
    archives, dernier_dev = donnees_archives(racine, table, jours, developpement, moteur_timestral)
    poids_archives = pd.read_csv(destination / "scores_poids_archives.csv").poids_timestral_retenu.iloc[0]
    s2, p2 = benchmark(table, archives, DEBUT_VERIFICATION, dernier_dev, moteur_chronos,
                       "archives_documentees", poids_archives)
    print(s2.to_string(index=False))
    pd.concat([s1, s2]).to_csv(destination / "scores_familles.csv", index=False, float_format="%.4f")
    pd.concat([p1, p2]).to_csv(destination / "predictions_familles.csv", float_format="%.4f")
