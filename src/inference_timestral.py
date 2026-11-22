"""Inférence locale avec les poids officiels de Mistral AI."""

import os
from pathlib import Path

import numpy as np


MODELE = "mistralai/Timestral-209M"
REVISION = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
DOSSIER_MODELE = Path(__file__).resolve().parents[1] / "data" / "models" / "Timestral-209M"


class MistralTimestral:
    """Réutilise les mêmes poids pour toutes les séries d'un pipeline."""

    def __init__(self):
        import torch
        from timestral import TimestralForecaster

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        local = all((DOSSIER_MODELE / nom).is_file() for nom in ("config.json", "model.safetensors"))
        source = str(DOSSIER_MODELE) if local else MODELE
        options = {"local_files_only": True} if local else {"revision": REVISION}
        self._modele = TimestralForecaster.from_pretrained(source, device="cpu", **options)

    def prevoir(self, contexte, horizon, *, passe=None, calendrier=None, bornes=(0, None)):
        contexte = np.asarray(contexte, dtype=np.float32)
        resultat = self._modele.forecast(
            contexte,
            prediction_length=horizon,
            past_covariates=passe,
            known_covariates=calendrier,
        )
        quantiles = np.asarray(resultat.quantiles)
        if not np.isfinite(quantiles).all():
            raise ValueError("Timestral a renvoyé des quantiles non finis.")
        bas, haut = bornes
        if bas is not None or haut is not None:
            quantiles = np.clip(quantiles, bas, haut)
        return quantiles, np.asarray(resultat.quantile_levels)

    def quantiles(self, contexte, horizon, **options):
        valeurs, niveaux = self.prevoir(contexte, horizon, **options)
        indices = [int(np.argmin(abs(niveaux - q))) for q in (0.1, 0.5, 0.9)]
        return valeurs[..., indices]


def cycle(mois):
    """Deux covariables calendaires connues sur le contexte et l'horizon."""
    angle = 2 * np.pi * (np.asarray(mois) - 1) / 12
    return np.stack([np.sin(angle), np.cos(angle)]).astype(np.float32)


def probabilite_depassement(quantiles, niveaux, seuil):
    """Approxime P(Y > seuil) par interpolation de la fonction quantile."""
    return float(1 - np.interp(seuil, quantiles, niveaux, left=0, right=1))
