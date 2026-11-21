#!/usr/bin/env python3
"""Timestral — projection régionale de l’occupation hospitalière avec SARS-CoV-2.

  1. fetch_mensuel     : API data.gouv.fr -> data/raw/mensuel (COVID régional journalier,
                         prévalences pathologies régionales CNAM, population régionale INSEE)
  2. table_mensuelle   : construit la table mensuelle régionale 2020-03 -> 2023-06
  3. modele_mensuel    : backtest glissant + projection rétrospective à 24 mois
                         + projection des prévalences chroniques à 2030
  4. plots_mensuel     : dashboard interactif HTML (dropdown région, range slider...)

Lancement :  .venv/bin/python main_mensuel.py [--skip-fetch]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true",
                        help="ne pas interroger l'API (utilise data/raw/mensuel tel quel)")
    args = parser.parse_args()

    if not args.skip_fetch:
        import fetch_mensuel
        print("=== 0. Fetch des données (API data.gouv.fr) ===")
        fetch_mensuel.fetch_all()
    else:
        print("=== 0. Fetch sauté (--skip-fetch) ===")

    import table_mensuelle
    import modele_mensuel
    import plots_mensuel
    from inference_timestral import MistralTimestral

    print("\n=== 1. Table mensuelle régionale ===")
    table_mensuelle.construire_table()

    print("\n=== 2. Backtest glissant et projection rétrospective ===")
    modele_mensuel.run(MistralTimestral())

    print("\n=== 3. Dashboard interactif ===")
    plots_mensuel.generer_dashboard_mensuel()

    print("\n=== Terminé ===")
    print("  - data/processed/table_mensuelle_regions.csv")
    print("  - outputs/prevision_mensuelle_regions.csv / _france.csv")
    print("  - outputs/backtest_mensuel.csv")
    print("  - outputs/prevalence_projections_2030.csv")
    print("  - outputs/dashboard_mensuel.html  (ouvrir dans un navigateur)")


if __name__ == "__main__":
    main()
