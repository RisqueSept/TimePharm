"""Compare des modèles sur les volumes Medic’AM avec une période finale réservée."""

from pathlib import Path

from src.recherche_volumes_pharma import experimenter


if __name__ == "__main__":
    experimenter(Path(__file__).resolve().parent)
