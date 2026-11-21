"""Actualise l'export national des infections respiratoires de Sentinelles."""

import csv
import json
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile


SENTINELLES = "https://www.sentiweb.fr/api/v1/datasets/rest/dataset?id=inc-25-PAY-ds2&span=all"
CHAMPS = ("week", "geo_insee", "indicator", "inc", "inc100", "inc_up", "inc_low", "inc100_up", "inc100_low")


def actualiser_sentinelles(destination):
    reponse = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "25", SENTINELLES],
        check=True, stdout=subprocess.PIPE,
    )
    export = json.loads(reponse.stdout)
    meta, lignes = export["meta"], export["data"]
    semaines = [ligne["week"] for ligne in lignes]
    if (not semaines or semaines != sorted(set(semaines))
        or any(ligne["indicator"] != 25 or ligne["geo_insee"] != "FR" for ligne in lignes)):
        raise ValueError("Export inattendu pour l'incidence nationale des infections respiratoires.")
    with destination.open(encoding="utf-8", newline="") as courant:
        entete = json.loads(courant.readline().removeprefix("# "))
        historique = list(csv.DictReader(courant))
    if meta["date"] < entete["date"] or semaines[-1] < int(historique[-1]["week"]):
        raise ValueError("L'export Sentinelles est plus ancien que celui du dépôt.")
    if meta == entete and historique == [
        {champ: "" if ligne[champ] is None else str(ligne[champ]) for champ in CHAMPS} for ligne in lignes
    ]:
        return meta["date"], False
    with NamedTemporaryFile(mode="w", dir=destination.parent, newline="", encoding="utf-8", delete=False) as f:
        temporaire = Path(f.name)
        try:
            f.write("# " + json.dumps(meta, ensure_ascii=False, separators=(",", ":")) + "\n")
            writer = csv.DictWriter(f, CHAMPS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(lignes)
            f.close()
            os.replace(temporaire, destination)
        finally:
            temporaire.unlink(missing_ok=True)
    return meta["date"], True


if __name__ == "__main__":
    chemin = Path(__file__).resolve().parents[1] / "data/raw/pharma/sentinelles_ira.csv"
    date_export, modifie = actualiser_sentinelles(chemin)
    print(f"Sentinelles {date_export} · {'actualisé' if modifie else 'export identique'}")
