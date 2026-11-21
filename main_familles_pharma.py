"""Compare Timestral, Chronos-2 et Holt-Winters pour les boîtes remboursées."""

from pathlib import Path

from src.recherche_familles_pharma import experimenter_familles


if __name__ == "__main__":
    experimenter_familles(Path(__file__).resolve().parent)
