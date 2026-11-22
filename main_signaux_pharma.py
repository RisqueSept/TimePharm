"""Compare les profils saisonniers et les signaux pédiatriques au modèle combiné."""

from argparse import ArgumentParser
from pathlib import Path

from src.recherche_signaux_pharma import experimenter_signaux


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--jour-origine", type=int, choices=(1, 21), default=1)
    args = parser.parse_args()
    experimenter_signaux(Path(__file__).resolve().parent, args.jour_origine)
