"""Expériences à un mois sur les volumes de boîtes remboursées."""

from datetime import date

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .inference_timestral import MistralTimestral, cycle
from .modele_volumes_pharma import DELAI_MOIS


DEBUT = pd.Period("2018-01", freq="M")
DEBUT_VERIFICATION = pd.Period("2024-09", freq="M")
FIN_DEV = DEBUT_VERIFICATION - 1 - DELAI_MOIS
REFERENCE = "timestral_courant"
VARIANTES = {
    "timestral_long": (None, "formes", False),
    "timestral_60_mois": (60, "formes", False),
    "timestral_joint": (None, "joint", False),
    "timestral_total": (None, "total", False),
    "timestral_jours_ouvres": (None, "total", True),
}
COLONNES_RIDGE = ["sin_mois", "cos_mois", "jours_ouvres", "ratio_3m", "ratio_6m", "ratio_12m"]
COLONNES_RESP = ["ira_4s", "ira_4s_precedentes", "grippe_4s"]


def charger(racine):
    production = racine / "data" / "raw" / "pharma"
    recherche = racine / "data" / "research" / "pharma"
    fichiers = [recherche / "medicam_2012_2020.csv", production / "medicam_volumes.csv"]
    volumes = pd.concat([pd.read_csv(path, parse_dates=["mois"]) for path in fichiers])
    volumes["mois"] = volumes.mois.dt.to_period("M")
    table = volumes.pivot(index="mois", columns="produit", values="boites")
    dates = pd.period_range(table.index.min(), table.index.max(), freq="M")
    if not table.index.equals(dates) or table.isna().any().any() or (table <= 0).any().any():
        raise ValueError("La comparaison exige une série mensuelle complète et positive par forme.")
    feries = pd.to_datetime(pd.read_csv(recherche / "jours_feries_metropole.csv").date)
    jours = pd.Series({
        mois: np.busday_count(
            mois.start_time.date(), (mois + 1).start_time.date(),
            holidays=feries.to_numpy(dtype="datetime64[D]"),
        )
        for mois in dates
    })
    ira = charger_incidence(production / "sentinelles_ira.csv")
    grippe = charger_incidence(recherche / "sentinelles_grippe.csv")
    return table, jours, ira, grippe


def charger_incidence(fichier):
    donnees = pd.read_csv(fichier, comment="#", usecols=["week", "inc100"])
    donnees["inc100"] = pd.to_numeric(donnees.inc100, errors="coerce")
    donnees = donnees.dropna()
    semaines = pd.to_datetime([
        date.fromisocalendar(*divmod(int(semaine), 100), 1) for semaine in donnees.week
    ])
    # Approximation du premier mercredi de publication par rapport au lundi mesuré.
    return pd.Series(donnees.inc100.to_numpy(), index=semaines + pd.Timedelta(days=10)).sort_index()


def incidences_au_depart(origine, ira, grippe):
    date_depart = origine.start_time
    fin_recente = date_depart - pd.Timedelta(days=28)
    debut_precedent = fin_recente - pd.Timedelta(days=28)
    return {
        "ira_4s": np.log1p(ira.loc[(ira.index >= fin_recente) & (ira.index < date_depart)].mean()),
        "ira_4s_precedentes": np.log1p(ira.loc[(ira.index >= debut_precedent) & (ira.index < fin_recente)].mean()),
        "grippe_4s": np.log1p(grippe.loc[(grippe.index >= fin_recente) & (grippe.index < date_depart)].mean()),
    }


def calculer_features(table, jours, ira, grippe):
    lignes = []
    total = table.sum(axis=1)
    for cible in total.index[27:]:
        origine = cible - 1
        fin = origine - DELAI_MOIS
        connus = total.loc[:fin]
        base = float(np.median([connus[cible - retard] for retard in (12, 24)]))
        angle = 2 * np.pi * (cible.month - 1) / 12
        ratios = connus / connus.shift(12)
        lignes.append({
            "mois": cible, "origine": origine, "dernier_mois_utilise": fin,
            "observe": total[cible], "saisonnier": total[cible - 12],
            "mediane_saisonniere": base,
            "saisonnier_ajuste": (table.loc[cible - 12] * (
                table.loc[:fin] / table.shift(12).loc[:fin]
            ).tail(6).median()).round().sum(),
            "sin_mois": np.sin(angle), "cos_mois": np.cos(angle),
            "jours_ouvres": jours[cible], "ratio_3m": np.log(ratios.tail(3).median()),
            "ratio_6m": np.log(ratios.tail(6).median()), "ratio_12m": np.log(ratios.tail(12).median()),
            **incidences_au_depart(origine, ira, grippe),
        })
    return pd.DataFrame(lignes).set_index("mois")


def prevision_ridge(features, cible, colonnes):
    ligne = features.loc[[cible]]
    fin = ligne.dernier_mois_utilise.iloc[0]
    apprentissage = features.loc[:fin]
    modele = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=12),
    )
    y = np.log(apprentissage.observe / apprentissage.mediane_saisonniere)
    modele.fit(apprentissage[colonnes], y)
    return float(np.exp(modele.predict(ligne[colonnes])[0]) * ligne.mediane_saisonniere.iloc[0])


def quantiles_timestral(moteur, table, jours, cible, *, fenetre, mode, calendrier_ouvre,
                        delai_mois=DELAI_MOIS):
    origine = cible - 1
    historique = table.loc[:origine - delai_mois]
    if fenetre is not None:
        historique = historique.tail(fenetre)
    dates = pd.period_range(historique.index[0], cible, freq="M")
    pas = cible.ordinal - historique.index[-1].ordinal
    cov = cycle(dates.month)
    if calendrier_ouvre:
        cov = np.vstack([cov, jours.loc[dates].to_numpy()])
    if mode == "total":
        valeurs = historique.sum(axis=1).to_numpy()[None, None, :]
    elif mode == "joint":
        valeurs = historique.to_numpy().T[None, :, :]
    else:
        valeurs = historique.to_numpy().T[:, None, :]
    cov = np.repeat(cov[None].astype(np.float32), len(valeurs), axis=0)
    return moteur.quantiles(valeurs, pas, calendrier=cov)


def timestral(moteur, table, jours, cible, **options):
    quantiles = quantiles_timestral(moteur, table, jours, cible, **options)
    return float(np.rint(quantiles[..., -1, 1]).sum())


def scores(resultats, modeles):
    reel = resultats.observe.to_numpy()
    valeurs = []
    for modele in modeles:
        erreurs = resultats[modele].to_numpy() - reel
        valeurs.append({
            "modele": modele, "n_mois": len(reel), "mae_boites": np.abs(erreurs).mean(),
            "wape_pct": 100 * np.abs(erreurs).sum() / reel.sum(),
            "biais_pct": 100 * erreurs.sum() / reel.sum(),
            "erreur_max_boites": np.abs(erreurs).max(),
        })
    return pd.DataFrame(valeurs).sort_values("wape_pct", ignore_index=True)


def produire(table, jours, features, dates, modeles, moteur):
    lignes = features.loc[dates, [
        "origine", "dernier_mois_utilise", "observe", "saisonnier", "mediane_saisonniere", "saisonnier_ajuste"
    ]].copy()
    for modele in modeles:
        if modele in lignes.columns or modele == REFERENCE:
            continue
        print(f"  {modele} · {len(lignes)} mois", flush=True)
        if modele in VARIANTES:
            fenetre, mode, calendrier_ouvre = VARIANTES[modele]
            lignes[modele] = [
                timestral(moteur, table, jours, cible, fenetre=fenetre, mode=mode,
                          calendrier_ouvre=calendrier_ouvre)
                for cible in dates
            ]
        else:
            colonnes = COLONNES_RIDGE + (COLONNES_RESP if modele == "ridge_respiratoire" else [])
            lignes[modele] = [prevision_ridge(features, cible, colonnes) for cible in dates]
    return lignes


def experimenter(racine):
    table, jours, ira, grippe = charger(racine)
    features = calculer_features(table, jours, ira, grippe)
    moteur = MistralTimestral()
    candidats = ["saisonnier", "mediane_saisonniere", "saisonnier_ajuste",
                 "ridge_calendrier", "ridge_respiratoire", *VARIANTES]
    dates_dev = features.index[(features.index >= DEBUT) & (features.index <= FIN_DEV)]
    dates_test = features.index[features.index >= DEBUT_VERIFICATION]
    print("Sélection sur le développement", flush=True)
    developpement = produire(table, jours, features, dates_dev, candidats, moteur)
    stats_dev = scores(developpement, candidats)
    candidat = stats_dev.modele.iloc[0]
    print(stats_dev.to_string(index=False), flush=True)
    print(f"Vérification finale · candidat choisi : {candidat}", flush=True)
    verification = produire(table, jours, features, dates_test, [candidat], moteur)
    courant = pd.read_csv(racine / "outputs" / "backtest_boites_pharma.csv")
    courant = courant.loc[courant.horizon.eq(1)].assign(mois=lambda d: pd.PeriodIndex(d.mois, freq="M"))
    total_courant = courant.groupby("mois")[["observe", "timestral"]].sum().reindex(dates_test)
    if not total_courant.observe.eq(verification.observe).all():
        raise ValueError("Le backtest du modèle courant doit couvrir les mêmes mois et les mêmes volumes.")
    verification[REFERENCE] = total_courant.timestral
    comparaison = list(dict.fromkeys([candidat, REFERENCE, "saisonnier", "saisonnier_ajuste"]))
    stats_test = scores(verification, comparaison)
    print(stats_test.to_string(index=False), flush=True)
    destination = racine / "outputs" / "recherche_pharma"
    destination.mkdir(parents=True, exist_ok=True)
    developpement.to_csv(destination / "developpement.csv", float_format="%.4f")
    verification[["origine", "dernier_mois_utilise", "observe", *comparaison]].to_csv(
        destination / "verification.csv", float_format="%.4f"
    )
    pd.concat([
        stats_dev.assign(periode="developpement", candidat_retenu=candidat),
        stats_test.assign(periode="verification", candidat_retenu=candidat),
    ]).to_csv(destination / "scores.csv", index=False, float_format="%.4f")
