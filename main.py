#!/usr/bin/env python3
"""Timestral — pipeline complet de prévision des lits hospitaliers.

Étapes :
  1. fetch    : télécharge les données via l'API publique de data.gouv.fr (si absentes)
  2. extract  : lit les lits par discipline dans les archives SAE 2013-2025
  3. modèle   : backtest temporel, capacité 2026-2035 et scénarios de charge MCO

Lancement :  .venv/bin/python main.py [--skip-fetch]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import fetch_data
import extract_lits
import manifest
import modele_lits
import plots_interactifs
from inference_timestral import MistralTimestral


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true",
                        help="ne pas interroger l'API data.gouv (utilise data/raw tel quel)")
    args = parser.parse_args()

    if args.skip_fetch:
        sae = extract_lits.archives_locales()
    else:
        sae, _ = fetch_data.fetch_all()
    manifest.ecrire_manifest(os.path.dirname(os.path.abspath(__file__)), "lits",
                             sorted(str(v) for v in sae.values()),
                             {"skip_fetch": args.skip_fetch})

    print("\n=== 1. Extraction des lits (SAE) ===")
    extract_lits.extract_all(sae)

    print("\n=== 2. Dataset + modèles + prévision ===")
    df = modele_lits.construire_dataset()
    ctx = modele_lits.entrainer_et_prevoir(df, MistralTimestral())
    modele_lits.tracer(ctx)

    print("\n=== 3. Dashboard interactif ===")
    plots_interactifs.generer_dashboard()

    print("\n=== Terminé ===")
    print("  - data/processed/dataset_annuel.csv          (données annuelles fusionnées)")
    print("  - outputs/prediction_lits_2026_2035.csv       (prévisions)")
    print("  - outputs/backtest_predictions.csv            (Timestral et références)")
    print("  - outputs/scenarios_capacite_mco*.csv          (scénarios nationaux et territoriaux)")
    print("  - outputs/fig1..fig4*.png                     (graphiques statiques)")
    print("  - outputs/dashboard_lits.html                 (dashboard interactif -> ouvrir dans un navigateur)")


if __name__ == "__main__":
    main()
