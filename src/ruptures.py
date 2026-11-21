"""Détection de ruptures de définition dans les séries d'entrée.

Les sources changent parfois de méthode (ANSM entre saisons, Sentinelles en 2020,
révisions rétrospectives Medic'AM/Odissé). Une rupture de moyenne ou de variance
est un signal qu'il ne faut plus comparer les périodes telles quelles.

Méthode : pour chaque point t, on compare sa valeur à la moyenne et à l'écart-type
des `fenetre` points précédents (strictement antérieurs, donc causal) ; un |z| > seuil
déclenche une alerte. Exécution : `python -m src.ruptures` écrit `outputs/ruptures_series.csv`.
"""

import numpy as np
import pandas as pd


def detecter_ruptures(valeurs, fenetre=12, seuil=3.0, min_points=24):
    """Dates (index) où la série s'écarte de son régime passé.

    `valeurs` : Series pandas indexée par date. Retourne une Series booléenne
    (True = rupture) de même index, fausse tant que l'historique est insuffisant.
    """
    s = pd.Series(valeurs).astype(float)
    ruptures = pd.Series(False, index=s.index)
    if len(s) < min_points:
        return ruptures
    for i in range(min_points, len(s)):
        fenetre_vals = s.iloc[i - fenetre:i].dropna()
        if len(fenetre_vals) < fenetre // 2:
            continue
        moyenne = fenetre_vals.mean()
        ecart = fenetre_vals.std(ddof=1)
        if not np.isfinite(ecart) or ecart == 0:
            # régime constant : toute valeur différente est une rupture
            if s.iloc[i] != moyenne:
                ruptures.iloc[i] = True
            continue
        z = abs(s.iloc[i] - moyenne) / ecart
        if z > seuil:
            ruptures.iloc[i] = True
    return ruptures


def rapport_ruptures(dossier_raw, dossier_research=None):
    """Rapport de ruptures pour les séries pharma connues.

    Retourne un DataFrame (série, date, valeur, z, fenetre, seuil).
    """
    lignes = []

    def ajouter(nom, serie, fenetre, seuil):
        s = pd.Series(serie).astype(float).reset_index(drop=True)
        dates = pd.Series(serie).index
        r = detecter_ruptures(s, fenetre=fenetre, seuil=seuil)
        for i in np.flatnonzero(r.to_numpy()):
            i = int(i)
            fen = s.iloc[max(0, i - fenetre):i].dropna()
            z = abs(s.iloc[i] - fen.mean()) / (fen.std(ddof=1) or np.nan)
            lignes.append({
                "serie": nom, "date": str(dates[i]), "valeur": s.iloc[i],
                "z": None if not np.isfinite(z) else round(float(z), 3),
                "fenetre": fenetre, "seuil": seuil,
            })

    pharma = dossier_raw / "pharma"
    if (pharma / "sentinelles_ira.csv").exists():
        ira = pd.read_csv(pharma / "sentinelles_ira.csv", comment="#", usecols=["week", "inc100"])
        ira["semaine"] = pd.to_datetime([pd.Timestamp.fromisocalendar(int(w // 100), int(w % 100), 1)
                                         for w in ira.week])
        ajouter("sentinelles_ira", ira.set_index("semaine").inc100.dropna(), fenetre=26, seuil=3.0)
    if (pharma / "medicam_volumes.csv").exists():
        vol = pd.read_csv(pharma / "medicam_volumes.csv", parse_dates=["mois"])
        total = vol.groupby("mois").boites.sum()
        ajouter("medicam_total", total, fenetre=12, seuil=3.0)
    if (pharma / "ansm_couverture.csv").exists():
        cov = pd.read_csv(pharma / "ansm_couverture.csv", parse_dates=["observation"])
        hebdo = cov.groupby("observation").grossistes.apply(lambda x: (x == 0).mean())
        ajouter("ansm_part_insuffisante", hebdo, fenetre=8, seuil=3.0)
    if dossier_research is not None:
        od = dossier_research / "pharma" / "odisse_respiratoire_pediatrie.csv"
        if od.exists():
            df = pd.read_csv(od)
            col = [c for c in df.columns if c not in ("semaine", "week", "date")][0]
            s = df.set_index(pd.to_datetime(df.iloc[:, 0]))[col]
            ajouter("odisse_pediatrie", s, fenetre=26, seuil=3.0)
    return pd.DataFrame(lignes)


def main(racine):
    racine = __import__("pathlib").Path(racine)
    rapport = rapport_ruptures(racine / "data" / "raw", racine / "data" / "research")
    destination = racine / "outputs" / "ruptures_series.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    rapport.to_csv(destination, index=False)
    print(f"{len(rapport)} ruptures détectées -> {destination}")
    return rapport


if __name__ == "__main__":
    import sys
    from pathlib import Path
    racine = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    main(racine)
