"""Inférence locale avec les poids officiels de Mistral AI.

Cache disque optionnel (chantier #14) : chaque inférence est stockée sous
``data/cache/timestral/``, indexée par un SHA-256 de la révision du modèle et de
toutes les entrées (contexte, horizon, covariables passées, calendrier, bornes,
sérialisés en octets float32). Un backtest rejoué avec les mêmes entrées ne
recalcule rien. Désactivation : ``MistralTimestral(cache=False)`` ou
``TIMESTRAL_CACHE=0``. Une inférence en erreur n'est jamais mise en cache.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


MODELE = "mistralai/Timestral-209M"
REVISION = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
DOSSIER_MODELE = Path(__file__).resolve().parents[1] / "data" / "models" / "Timestral-209M"
DOSSIER_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "timestral"

_VALEURS_DESACTIVEES = {"0", "false", "no", "off"}


def _cache_voulu(cache):
    """``cache=None`` : activé par défaut, désactivé si ``TIMESTRAL_CACHE`` vaut 0/false/no/off."""
    if cache is not None:
        return bool(cache)
    return os.environ.get("TIMESTRAL_CACHE", "1").strip().lower() not in _VALEURS_DESACTIVEES


def _hacher(h, donnees):
    """Ajoute un segment au hash en le préfixant par sa longueur (pas d'ambiguïté)."""
    h.update(len(donnees).to_bytes(8, "little"))
    h.update(donnees)


def _hacher_tableau(h, valeur):
    """Sérialise un tableau en octets float32 (forme + données) ; ``None`` = covariable absente."""
    if valeur is None:
        _hacher(h, b"absent")
        return
    t = np.ascontiguousarray(np.asarray(valeur, dtype=np.float32))
    _hacher(h, np.asarray(t.shape, dtype=np.int64).tobytes())
    _hacher(h, t.tobytes())


def cle_cache(contexte, horizon, passe, calendrier, bornes):
    """SHA-256 de la révision du modèle et de toutes les entrées d'une inférence."""
    h = hashlib.sha256()
    _hacher(h, f"{MODELE}@{REVISION}".encode("utf-8"))
    _hacher_tableau(h, contexte)
    _hacher(h, int(horizon).to_bytes(8, "little", signed=True))
    _hacher_tableau(h, passe)
    _hacher_tableau(h, calendrier)
    bas, haut = bornes
    _hacher(h, json.dumps([None if b is None else float(b) for b in (bas, haut)]).encode("utf-8"))
    return h.hexdigest()


class MistralTimestral:
    """Réutilise les mêmes poids pour toutes les séries d'un pipeline.

    ``cache`` (activé par défaut, sauf ``TIMESTRAL_CACHE=0``) met en cache disque
    les quantiles calculés, sous ``dossier_cache`` (défaut : ``data/cache/timestral``).
    """

    def __init__(self, cache=None, dossier_cache=None):
        import torch
        from timestral import TimestralForecaster

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        local = all((DOSSIER_MODELE / nom).is_file() for nom in ("config.json", "model.safetensors"))
        source = str(DOSSIER_MODELE) if local else MODELE
        options = {"local_files_only": True} if local else {"revision": REVISION}
        self._modele = TimestralForecaster.from_pretrained(source, device="cpu", **options)
        self._cache_actif = _cache_voulu(cache)
        self._dossier_cache = Path(dossier_cache) if dossier_cache is not None else DOSSIER_CACHE

    def _lire_cache(self, cle):
        """Retourne (quantiles, niveaux) ou None si l'entrée est absente/illisible."""
        try:
            with np.load(self._dossier_cache / f"{cle}.npz", allow_pickle=False) as contenu:
                quantiles = np.array(contenu["quantiles"])
                niveaux = np.array(contenu["niveaux"])
        except Exception:
            return None
        if quantiles.size == 0 or not np.isfinite(quantiles).all():
            return None
        return quantiles, niveaux

    def _ecrire_cache(self, cle, quantiles, niveaux):
        """Écriture atomique (fichier temporaire + renommage) ; best-effort."""
        try:
            self._dossier_cache.mkdir(parents=True, exist_ok=True)
            chemin = self._dossier_cache / f"{cle}.npz"
            temporaire = chemin.with_name(f"{cle}.{os.getpid()}.tmp")
            with open(temporaire, "wb") as f:
                np.savez(f, quantiles=quantiles, niveaux=niveaux)
            os.replace(temporaire, chemin)
        except OSError:
            pass

    def prevoir(self, contexte, horizon, *, passe=None, calendrier=None, bornes=(0, None)):
        contexte = np.asarray(contexte, dtype=np.float32)
        try:
            cle = cle_cache(contexte, horizon, passe, calendrier, bornes) if self._cache_actif else None
        except Exception:
            cle = None  # entrées non sérialisables : pas de cache, inférence inchangée
        if cle is not None:
            en_cache = self._lire_cache(cle)
            if en_cache is not None:
                return en_cache
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
        niveaux = np.asarray(resultat.quantile_levels)
        if cle is not None:
            self._ecrire_cache(cle, quantiles, niveaux)
        return quantiles, niveaux

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
