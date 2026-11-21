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

# Horizon maximal couvert par le backtest, par unité de temps (chantier #13) :
# lits -> horizons de 1 à 3 ans, mensuel -> horizons de 1 à 12 mois.
DELAI_VALIDATION = {"an": 3, "mois": 12}


def _annee(valeur):
    """Année d'une date (nombre = année seule, comme dans le JSON des lits)."""
    if isinstance(valeur, (int, float)) and not isinstance(valeur, bool):
        return int(valeur)
    return pd.Timestamp(valeur).year


def _indice_mois(valeur):
    """Indice mensuel d'une date (année * 12 + mois), p. ex. pour comparer des écarts en mois."""
    if isinstance(valeur, (int, float)) and not isinstance(valeur, bool):
        return int(valeur) * 12
    t = pd.Timestamp(valeur)
    return t.year * 12 + t.month


def est_hors_validation(date, derniere_observation, unite_horizon, delai_max=None):
    """Vrai si `date` est au-delà de l'horizon de validation du backtest (chantier #13).

    `unite_horizon` « an » : plus de `delai_max` années après la dernière observation ;
    « mois » : plus de `delai_max` mois après. Par défaut `delai_max` vaut
    `DELAI_VALIDATION[unite_horizon]` (3 ans pour les lits, 12 mois pour le mensuel).
    """
    if unite_horizon not in DELAI_VALIDATION:
        raise ValueError(f"unite_horizon inconnue : {unite_horizon!r}")
    if delai_max is None:
        delai_max = DELAI_VALIDATION[unite_horizon]
    if unite_horizon == "an":
        return _annee(date) - _annee(derniere_observation) > delai_max
    return _indice_mois(date) - _indice_mois(derniere_observation) > delai_max


def limite_validation(derniere_observation, unite_horizon, delai_max=None):
    """Première date hors validation : dernier horizon validé + 1 unité."""
    if unite_horizon not in DELAI_VALIDATION:
        raise ValueError(f"unite_horizon inconnue : {unite_horizon!r}")
    if delai_max is None:
        delai_max = DELAI_VALIDATION[unite_horizon]
    if unite_horizon == "an":
        return _annee(derniere_observation) + delai_max + 1
    periode = pd.Timestamp(derniere_observation).to_period("M") + delai_max + 1
    return periode.to_timestamp().strftime("%Y-%m")


def marquer_hors_validation(records, derniere_observation, unite_horizon, delai_max=None):
    """Ajoute le champ `hors_validation` (0/1) à chaque enregistrement (chantier #13)."""
    return [
        {**r, "hors_validation": int(est_hors_validation(r["date"], derniere_observation, unite_horizon, delai_max))}
        for r in records
    ]


def note_hors_validation(derniere_observation, unite_horizon, delai_max=None):
    """Légende « hors validation » ajoutée à la note de projection (chantier #13)."""
    if unite_horizon not in DELAI_VALIDATION:
        raise ValueError(f"unite_horizon inconnue : {unite_horizon!r}")
    if delai_max is None:
        delai_max = DELAI_VALIDATION[unite_horizon]
    limite = limite_validation(derniere_observation, unite_horizon, delai_max)
    if unite_horizon == "an":
        return (f"Le backtest couvre les horizons de 1 à {delai_max} ans : au-delà de {limite} "
                f"(plus de {delai_max} ans après la dernière observation, {_annee(derniere_observation)}), "
                "les projections sont grisées — « hors validation ».")
    return (f"Le backtest couvre les horizons jusqu'à {delai_max} mois : au-delà de {limite} "
            f"(plus de {delai_max} mois après la dernière observation, "
            f"{pd.Timestamp(derniere_observation).strftime('%Y-%m')}), "
            "les projections sont grisées — « hors validation ».")


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
    projections = {}
    for territoire, groupe in full.groupby("region_nom"):
        dernier_observe = groupe.loc[groupe.est_prevision == 0, "mois"].max()
        records = enregistrements(
            groupe[["mois", "lits_pred", "est_prevision", "scenario_bas", "scenario_haut"]]
            .rename(columns={"mois": "date"})
        )
        projections[territoire] = marquer_hors_validation(records, dernier_observe, "mois")
    dernier_mois = full.loc[full.est_prevision == 0, "mois"].max()

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
    note = ("Timestral-209M prolonge la série historique de Santé publique France. Les courbes régionales "
            "affichent ses quantiles 10 % et 90 %. Le total national additionne les médianes régionales. "
            + note_hors_validation(dernier_mois, "mois"))
    return generer_html(
        OUT / "dashboard_mensuel.html", "Occupation hospitalière avec le SARS-CoV-2",
        "Compare les observations et les valeurs prédites par région ou pour la France entière. La série couvre les patients positifs hospitalisés, tous motifs confondus.",
        donnees,
        note,
        analyses,
        [("backtest_mensuel.csv", "Backtest régional (CSV)"),
         ("prevision_mensuelle_regions.csv", "Projection par région (CSV)"),
         ("prevision_mensuelle_france.csv", "Projection France (CSV)"),
         ("prevalence_projections_2030.csv", "Prévalences à 2030 (CSV)")],
        fraicheur=fraicheur_dashboard(OUT.parent, **{"Dernier mois observé": str(full.mois.max())}),
    )


if __name__ == "__main__":
    generer_dashboard_mensuel()
