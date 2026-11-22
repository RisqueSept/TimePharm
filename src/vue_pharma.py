"""Tableau de bord interactif, consultable hors ligne."""

import json
from pathlib import Path

from plotly.offline import get_plotlyjs

from .dashboard import enregistrements
from .modele_pharma import HORIZONS, PRODUITS


def generer_dashboard(destination, stocks, ira, volumes, predictions, test_volumes, futur_volumes):
    dossier = Path(__file__).resolve().parent
    colonnes = [
        "origine", "observation_initiale", "cible", "cible_disponible", "saison",
        "produit", "horizon", "delai_reel", "observé", "grossistes", "reference",
        "contexte", "degradation", "timestral", "source_initiale", "source_cible", "ira_incidence",
    ]
    donnees = {
        "produits": PRODUITS,
        "horizons": HORIZONS,
        "stocks": enregistrements(stocks),
        "ira": enregistrements(ira.loc[ira.semaine.ge("2021-01-01"), ["semaine", "inc100"]]),
        "volumes": enregistrements(volumes),
        "volumes_test": enregistrements(test_volumes),
        "volumes_futur": enregistrements(futur_volumes),
        "predictions": enregistrements(predictions[colonnes]),
    }
    replacements = {
        "__CSS__": (dossier / "pharma.css").read_text(encoding="utf-8"),
        "__DATA__": json.dumps(donnees, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"),
        "__PLOTLY__": get_plotlyjs(),
        "__JS__": (dossier / "pharma.js").read_text(encoding="utf-8"),
    }
    template = (dossier / "pharma.html").read_text(encoding="utf-8")
    for key, content in replacements.items():
        template = template.replace(key, content, 1)
    destination.write_text(template, encoding="utf-8")
