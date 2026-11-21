"""Téléchargement des données mensuelles/régionales via l'API publique de data.gouv.fr.

Sources (voir docs/DATA.md pour le détail) :
  1. SPF - "Données hospitalières relatives à l'épidémie de COVID-19"
     -> covid-hosp-txad-reg (taux de prévalence/incidence par région et par jour, 2020-2023)
  2. CNAM - "Pathologies : effectif de patients par pathologie... et territoire"
     (data.ameli.fr, référencé sur data.gouv.fr) -> prévalences régionales annuelles
     (diabète, cancers, cardio, psychiatrie...) 2015-2024
  3. INSEE - "Estimations localisées de population - séries longues"
     -> population au 1er janvier par région + part des 60 ans et + / 75 ans et +, 1975-2026

Idempotent : ne télécharge que ce qui manque dans data/raw/mensuel/.
"""

import json
import os
import urllib.parse
import urllib.request

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(BASE, "data", "raw", "mensuel")
UA = {"User-Agent": "timestral-test/1.0 (projet local de prévision de lits)"}

DS_COVID = "donnees-hospitalieres-relatives-a-lepidemie-de-covid-19"
DS_AMELI = ("pathologies-effectif-de-patients-par-pathologie-sexe-classe-dage"
            "-et-territoire-departement-region")
DS_INSEE = "estimations-localisees-de-population-series-longues"


def _get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url, dest, min_size=10_000):
    if os.path.exists(dest) and os.path.getsize(dest) > min_size:
        print(f"  déjà présent: {os.path.basename(dest)} ({os.path.getsize(dest)//1024} Ko)")
        return dest
    print(f"  GET {url[:110]}...")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"  -> {dest} ({os.path.getsize(dest)//1024} Ko)")
    return dest


def fetch_covid():
    meta = _get_json(f"https://www.data.gouv.fr/api/1/datasets/{DS_COVID}/")
    # on prend la ressource régionale la plus récente (covid-hosp-txad-reg-YYYY-MM-DD...)
    cands = [r for r in meta["resources"] if r["title"].startswith("covid-hosp-txad-reg")]
    cands.sort(key=lambda r: r["title"])
    url = cands[-1]["url"]
    return _download(url, os.path.join(RAW, "covid_hosp_txad_reg.csv"))


def fetch_ameli():
    meta = _get_json(f"https://www.data.gouv.fr/api/1/datasets/{DS_AMELI}/")
    base = next(r["url"] for r in meta["resources"] if r["title"] == "effectifs.csv")
    # export CSV filtré : tous sexes (9), tous âges (tsage), niveau région (dept=999)
    where = "sexe='9' AND cla_age_5='tsage' AND dept='999'"
    url = base + "?" + urllib.parse.urlencode({"use_labels": "true", "where": where})
    return _download(url, os.path.join(RAW, "ameli_pathologies_regions.csv"), min_size=100_000)


def fetch_insee_pop():
    meta = _get_json(f"https://www.data.gouv.fr/api/1/datasets/{DS_INSEE}/")
    url = next(r["url"] for r in meta["resources"] if "DS_ESTIMATION_POPULATION" in r.get("url", ""))
    return _download(url, os.path.join(RAW, "insee_estimation_population.zip"), min_size=100_000)


def fetch_all():
    os.makedirs(RAW, exist_ok=True)
    print("[fetch_mensuel] API data.gouv.fr ...")
    return {
        "covid": fetch_covid(),
        "ameli": fetch_ameli(),
        "insee_pop": fetch_insee_pop(),
    }


if __name__ == "__main__":
    fetch_all()
