"""Mesure la sensibilité au délai Medic'AM et teste une combinaison des références."""

from argparse import ArgumentParser
from datetime import date
from pathlib import Path

from src.recherche_delai_pharma import experimenter_delai


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--date-depart", type=date.fromisoformat,
                        help="Date de départ YYYY-MM-DD pour figer la prochaine prévision.")
    args = parser.parse_args()
    experimenter_delai(Path(__file__).resolve().parent, args.date_depart)
