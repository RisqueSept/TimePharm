"""CLI unifiée du projet Timestral (`python -m timestral ...`)."""

from pkgutil import extend_path

# Les sous-modules d'inférence résident dans le paquet officiel de Mistral.
__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"


def __getattr__(nom):
    if nom == "TimestralForecaster":
        from .forecaster import TimestralForecaster

        return TimestralForecaster
    raise AttributeError(nom)
