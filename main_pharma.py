"""Rejoue les prévisions de couverture des antibiotiques buvables."""

from pathlib import Path

from src.inference_timestral import MistralTimestral
from src.manifest import ecrire_manifest
from src.modele_pharma import HORIZONS, backtest, charger_donnees, previsions_timestral
from src.modele_volumes_pharma import DELAI_MOIS, HORIZONS_VOLUMES, prevoir_volumes
from src.vue_pharma import generer_dashboard


def main():
    root = Path(__file__).resolve().parent
    dossier = root / "data" / "raw" / "pharma"
    stocks, ira, volumes = charger_donnees(dossier)
    ecrire_manifest(root, "pharma", sorted(dossier.glob("*.csv")), {
        "horizons_tensions_jours": list(HORIZONS),
        "horizons_volumes_mois": list(HORIZONS_VOLUMES),
        "delai_publication_mois": DELAI_MOIS,
    })
    moteur = MistralTimestral()
    print("Backtest hebdomadaire avec Timestral-209M")
    predictions = previsions_timestral(stocks, ira, backtest(stocks, ira), moteur)
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    print("Backtest mensuel et projections des boîtes avec Timestral-209M")
    test_volumes, futur_volumes = prevoir_volumes(volumes, stocks.disponible.max(), moteur)
    predictions.to_csv(out / "backtest_pharma.csv", index=False, float_format="%.6f")
    test_volumes.to_csv(out / "backtest_boites_pharma.csv", index=False)
    futur_volumes.to_csv(out / "previsions_boites_pharma.csv", index=False)
    generer_dashboard(out / "dashboard_pharma.html", stocks, ira, volumes, predictions,
                      test_volumes, futur_volumes)
    print(f"{len(stocks)} relevés, {len(predictions)} prévisions hors échantillon")
    print(out / "dashboard_pharma.html")


if __name__ == "__main__":
    main()
