"""Vrai calendrier de disponibilité des éditions Medic'AM.

Le pipeline courant simule un délai de publication fixe de ``DELAI_MOIS`` mois
entre le mois de remboursement et sa disponibilité. Ce module construit à la
place un calendrier réel depuis les dates documentées des archives Medic'AM
(``data/research/pharma/medicam_dates_archives.csv``) : pour une date d'origine
donnée, :func:`cutoff_pour` renvoie le dernier mois de remboursement dont
l'édition documentée est strictement antérieure à l'origine.

La logique de disponibilité est celle de ``mois_documentes`` et
``dates_avec_prescripteurs`` (`src/recherche_archives_pharma.py`), importée
sans modification : on s'arrête au premier mois dont l'édition n'est pas
encore attestée à la date d'origine. Le résultat se branche directement comme
paramètre ``delais`` de :func:`src.modele_volumes_pharma.prevoir_volumes` via
:func:`delais_reel`.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .modele_volumes_pharma import DELAI_MOIS
from .recherche_archives_pharma import mois_documentes


_REGEX_ARCHIVE = re.compile(r"(\d{4})-(\d{2})-a-(\d{2})")


def _normaliser_editions(editions):
    """Normalise ``editions`` en Series mois (PeriodIndex M) -> date d'édition (Timestamp)."""
    if isinstance(editions, pd.DataFrame):
        # Plusieurs sources d'éditions (ex. ATC et prescripteurs) : la plus
        # ancienne attestation gagne, comme dans `dates_avec_prescripteurs`.
        editions = editions.apply(pd.to_datetime).min(axis=1)
    if not isinstance(editions, pd.Series):
        editions = pd.Series(editions)
    editions = editions.copy()
    editions.index = pd.PeriodIndex(editions.index, freq="M")
    editions = pd.to_datetime(editions)
    return editions.sort_index()


def cutoff_pour(origine, editions):
    """Dernier mois de remboursement dont l'édition documentée précède ``origine``.

    ``origine`` : date d'origine (Timestamp, ``datetime.date``, chaîne) ou mois
    (``pd.Period``, pris au premier jour du mois). ``editions`` : mois -> date
    d'édition documentée (Series, dict ou DataFrame), comme celles produites
    par ``dates_documentees`` / ``dates_avec_prescripteurs`` de
    `src/recherche_archives_pharma.py` ou par :func:`editions_archives` ici.

    Retourne un ``pd.Period`` (freq M) : la fin de la série contiguë de mois
    dont l'édition est strictement antérieure à ``origine`` (logique de
    ``mois_documentes``). Repli : mois d'origine - ``DELAI_MOIS`` quand aucune
    édition n'est documentée à cette date.
    """
    editions = _normaliser_editions(editions)
    if isinstance(origine, pd.Period):
        date_origine = origine.to_timestamp()
    else:
        date_origine = pd.Timestamp(origine)
    table = pd.DataFrame({"boites": np.ones(len(editions), dtype=float)}, index=editions.index)
    documentes = mois_documentes(table, editions, date_origine)
    if len(documentes) == 0:
        return date_origine.to_period(freq="M") - DELAI_MOIS
    return pd.Period(documentes.index[-1], freq="M")


def editions_archives(chemin):
    """Éditions par mois de remboursement depuis le CSV des archives Medic'AM.

    Chaque ligne du CSV documente une archive semestrielle dont le nom de
    fichier porte la période couverte (``AAAA-MM-a-MM``) ; l'édition retenue
    est la plus récente des dates de création et de mise à jour. Le résultat
    est une Series mois (PeriodIndex M) -> date d'édition, directement
    utilisable comme ``editions`` de :func:`cutoff_pour` (équivalent à
    ``dates_documentees``, qui recoupe les sources des CSV de volumes).
    """
    archives = pd.read_csv(Path(chemin))
    entrees = {}
    for _, ligne in archives.iterrows():
        motif = _REGEX_ARCHIVE.search(str(ligne["source"]))
        if motif is None:
            continue
        debut = pd.Period(f"{motif.group(1)}-{motif.group(2)}", freq="M")
        fin = pd.Period(f"{motif.group(1)}-{motif.group(3)}", freq="M")
        dates = pd.to_datetime(
            pd.Series([ligne["date_creation"], ligne["date_mise_a_jour"]]), errors="coerce"
        )
        edition = dates.max()
        for mois in pd.period_range(debut, fin, freq="M"):
            entrees[mois] = edition
    editions = pd.Series(entrees)
    editions.index = pd.PeriodIndex(editions.index, freq="M")
    return editions.sort_index()


def delais_reel(editions, jour_origine=21):
    """Fonction ``mois -> coupure`` utilisable comme paramètre ``delais`` de ``prevoir_volumes``.

    Le mois d'origine est daté au ``jour_origine`` du mois (21 par défaut,
    comme les départs du pipeline courant) avant de consulter les éditions
    documentées. Exemple : ``prevoir_volumes(volumes, date_depart, moteur,
    delais=delais_reel(editions))``.
    """
    editions = _normaliser_editions(editions)

    def delais(mois):
        mois = pd.Period(mois, freq="M")
        date_origine = mois.start_time + pd.Timedelta(days=jour_origine - 1)
        return cutoff_pour(date_origine, editions)

    return delais
