"""Évalue le modèle en se limitant aux éditions Medic’AM encore vérifiables."""

from pathlib import Path

from src.recherche_archives_pharma import experimenter_archives


if __name__ == "__main__":
    experimenter_archives(Path(__file__).resolve().parent)
