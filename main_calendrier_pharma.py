"""Teste les variations de jours ouvrés et d'activité scolaire sur les prévisions."""

from pathlib import Path

from src.recherche_calendrier_pharma import experimenter_calendrier


if __name__ == "__main__":
    experimenter_calendrier(Path(__file__).resolve().parent)
