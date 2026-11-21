"""Archive les prévisions émises et les rapproche des observations ultérieures."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from .dashboard import enregistrements
from .seuils import SEUIL_ALERTE_RISQUE


def figer_previsions(dossier, tensions, boites, manifest):
    emis = pd.Timestamp(manifest["date_utc"]).tz_convert("Europe/Paris")
    dossier.mkdir(parents=True, exist_ok=True)
    destination = dossier / f"{emis.date().isoformat()}.json"
    archive = {
        "emis": emis.isoformat(),
        "modele": manifest["modele"],
        "revision_modele": manifest["revision_modele"],
        "revision_code": manifest["revision_code"],
        "entrees": manifest["entrees"],
        "seuil": SEUIL_ALERTE_RISQUE,
        "tensions": enregistrements(tensions),
        "boites": enregistrements(boites),
    }
    with NamedTemporaryFile(mode="w", dir=dossier, encoding="utf-8") as fichier:
        json.dump(archive, fichier, ensure_ascii=False, indent=2, allow_nan=False)
        fichier.write("\n")
        fichier.flush()
        # La première émission de la journée prévaut lors de relances concurrentes.
        try:
            os.link(fichier.name, destination)
        except FileExistsError:
            pass
    return destination


def evaluer_previsions(dossier, stocks, volumes, date_evaluation):
    """Évalue chaque cible × horizon avec la première émission correspondante."""
    stocks = stocks.loc[stocks.disponible.le(date_evaluation) & stocks.grossistes.notna()]
    volumes = volumes.set_index(["mois", "produit"]).boites
    tensions, boites = {}, {}
    for chemin in sorted(Path(dossier).glob("*.json")):
        archive = json.loads(chemin.read_text(encoding="utf-8"))
        emis = pd.Timestamp(archive["emis"]).tz_localize(None).normalize()
        for ligne in archive["tensions"]:
            cible = pd.Timestamp(ligne["cible"])
            marge = pd.Timedelta(days=3)
            cle = (cible, ligne["produit"], ligne["horizon"])
            if cible - marge <= emis or cle in tensions:
                continue
            candidats = stocks.loc[
                stocks.produit.eq(ligne["produit"])
                & stocks.observation.between(cible - marge, cible + marge)
            ]
            observe = None
            if date_evaluation >= cible + marge and not candidats.empty:
                ecarts = (candidats.observation - cible).abs()
                observe = int(candidats.loc[ecarts.idxmin(), "grossistes"] == 0)
            tensions[cle] = dict(ligne, observe=observe, seuil=archive["seuil"], emis=archive["emis"])
        for ligne in archive["boites"]:
            mois = pd.Timestamp(ligne["mois"])
            cle = (mois, ligne["produit"], ligne["horizon"])
            if mois <= emis or cle in boites:
                continue
            observe = volumes.get((mois, ligne["produit"])) if mois <= date_evaluation else None
            boites[cle] = dict(ligne, observe=int(observe) if observe is not None else None, emis=archive["emis"])
    return {"tensions": list(tensions.values()), "boites": list(boites.values())}
