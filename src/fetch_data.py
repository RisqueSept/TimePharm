"""Téléchargement des données ouvertes via l'API publique de data.gouv.fr.

Sources :
  1. Dataset "Bases statistiques SAE" (DREES / Ministère des Solidarités et de la Santé)
     -> archives annuelles 2013-2025 (7z contenant les CSV par établissement).
  2. Dataset INSEE "Population et structure de la population - séries longues"
     -> zip contenant la population au 1er janvier par âge/sexe (1975-2026).

Le script est idempotent : il ne télécharge que les fichiers absents de data/raw/.
"""

import json
import os
import re
import sys
import urllib.request

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
UA = {"User-Agent": "timestral-test/1.0 (projet local de prévision de lits)"}

SAE_DATASET_API = "https://www.data.gouv.fr/api/1/datasets/bases-statistiques-sae/"
INSEE_DATASET_API = (
    "https://www.data.gouv.fr/api/1/datasets/"
    "population-et-structure-de-la-population-series-longues/"
)
ANNEES_SAE = list(range(2013, 2026))  # 2013 -> 2025 (CSV exploitables)


def _get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url: str, dest: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _exists_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1_000_000


def fetch_sae() -> dict:
    """Retourne {annee: chemin local du .7z} pour les années demandées."""
    meta = _get_json(SAE_DATASET_API)
    # On repère, pour chaque année, l'URL de l'archive "formats SAS-CSV.7z"
    urls = {}
    for res in meta.get("resources", []):
        title = res.get("title", "")
        m = re.search(r"SAE\s+(\d{4}).*SAS-CSV\.7z", title)
        if m:
            annee = int(m.group(1))
            if annee in ANNEES_SAE:
                urls.setdefault(annee, res["url"])  # setdefault: on garde la 1ère occurrence

    chemins = {}
    for annee in ANNEES_SAE:
        if annee not in urls:
            print(f"  [SAE] aucune archive CSV trouvée pour {annee}, on ignore.")
            continue
        dest = os.path.join(RAW_DIR, f"sae_{annee}.7z")
        if _exists_ok(dest):
            print(f"  [SAE] {annee}: déjà présent ({os.path.getsize(dest)//2**20} Mo)")
        else:
            print(f"  [SAE] {annee}: téléchargement de {urls[annee][:100]}...")
            _download(urls[annee], dest)
        chemins[annee] = dest
    return chemins


def fetch_insee() -> str:
    """Télécharge le zip INSEE des séries longues de population."""
    meta = _get_json(INSEE_DATASET_API)
    url = None
    for res in meta.get("resources", []):
        u = res.get("url", "")
        if "DS_POP_STRUCT_SERIES" in u:
            url = u
            break
    if not url:
        raise RuntimeError("Ressource INSEE DS_POP_STRUCT_SERIES introuvable via l'API data.gouv")
    dest = os.path.join(RAW_DIR, "insee_pop_struct_series.zip")
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print(f"  [INSEE] déjà présent ({os.path.getsize(dest)//1024} Ko)")
    else:
        print(f"  [INSEE] téléchargement de {url[:100]}...")
        _download(url, dest)
    return dest


def fetch_all():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("[fetch] Interrogation de l'API data.gouv.fr ...")
    sae = fetch_sae()
    insee = fetch_insee()
    print(f"[fetch] OK: {len(sae)} archives SAE + 1 fichier INSEE")
    return sae, insee


if __name__ == "__main__":
    fetch_all()
