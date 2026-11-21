"""Rejoue et archive les prévisions sur les antibiotiques buvables."""

from pathlib import Path

import pandas as pd

from src.delai_medicam import delais_pipeline
from src.fetch_pharma import actualiser_sentinelles
from src.inference_timestral import MistralTimestral
from src.manifest import ecrire_manifest, ecrire_sidecar
from src.modele_pharma import (
    HORIZONS, backtest, charger_donnees, metadonnees_fraicheur, previsions_timestral, prevoir_tensions,
)
from src.modele_volumes_pharma import DELAI_MOIS, HORIZONS_VOLUMES, prevoir_volumes
from src.suivi_pharma import evaluer_previsions, figer_previsions
from src.vue_pharma import generer_dashboard


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-sentinelles", action="store_true", help="télécharge l'export national IRA")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    dossier = root / "data" / "raw" / "pharma"
    if args.refresh_sentinelles:
        date_export, modifie = actualiser_sentinelles(dossier / "sentinelles_ira.csv")
        print(f"Sentinelles {date_export} · {'actualisé' if modifie else 'export identique'}")
    stocks, ira, volumes = charger_donnees(dossier)
    delais = delais_pipeline(root)
    entrees = sorted(dossier.glob("*.csv"))
    if delais is not None:
        entrees.append(root / "data/research/pharma/medicam_dates_archives.csv")
    manifest = ecrire_manifest(root, "pharma", entrees, {
        "horizons_tensions_jours": list(HORIZONS),
        "horizons_volumes_mois": list(HORIZONS_VOLUMES),
        "delai_publication": "archives_documentées" if delais is not None else f"fixe_{DELAI_MOIS}_mois",
    })
    moteur = MistralTimestral()
    date_generation = pd.Timestamp(manifest["date_utc"]).tz_convert("Europe/Paris").tz_localize(None).normalize()
    print("Backtest hebdomadaire avec Timestral-209M")
    predictions = previsions_timestral(stocks, ira, backtest(stocks, ira), moteur)
    futur_tensions = prevoir_tensions(stocks, ira, date_generation, moteur)
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    print("Backtest mensuel et projections des boîtes avec Timestral-209M"
          + (" (délai réel Medic'AM)" if delais is not None else ""))
    test_volumes, futur_volumes = prevoir_volumes(volumes, date_generation, moteur, delais=delais)
    predictions.to_csv(out / "backtest_pharma.csv", index=False, float_format="%.6f")
    test_volumes.to_csv(out / "backtest_boites_pharma.csv", index=False)
    futur_volumes.to_csv(out / "previsions_boites_pharma.csv", index=False)
    futur_tensions.to_csv(out / "previsions_tensions_pharma.csv", index=False, float_format="%.6f")
    for nom in ("backtest_pharma.csv", "backtest_boites_pharma.csv", "previsions_boites_pharma.csv",
                "previsions_tensions_pharma.csv"):
        ecrire_sidecar(out / nom, root, run="pharma")
    archives = out / "previsions_figees"
    archive = figer_previsions(archives, futur_tensions, futur_volumes, manifest)
    suivi = evaluer_previsions(archives, stocks, volumes, date_generation)
    for nom, lignes in suivi.items():
        pd.DataFrame(lignes).to_csv(out / f"suivi_{nom}_pharma.csv", index=False)
    fraicheur = metadonnees_fraicheur(stocks, volumes, manifest)
    generer_dashboard(out / "dashboard_pharma.html", stocks, ira, volumes, predictions,
                      test_volumes, futur_volumes, fraicheur=fraicheur,
                      futur_tensions=futur_tensions, suivi=suivi, archive=archive.name)
    print(f"{len(stocks)} relevés, {len(predictions)} prévisions hors échantillon")
    print(out / "dashboard_pharma.html")


if __name__ == "__main__":
    main()
