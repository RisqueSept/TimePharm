"""Extraction des lits installés par discipline depuis les archives SAE (7z).

Pour chaque année, on lit les CSV des bordereaux :
  MCO  -> lits d'hospitalisation complète MCO        (colonne LIT_MCO)
  REA  -> lits de réanimation                         (colonne LIT)
  PSY  -> lits d'hospitalisation temps plein psy     (colonne CAP_HTP)
  SSR  -> lits de soins de suite et réadaptation     (colonne LIT)
  USLD -> lits de soins de longue durée              (colonne LIT)
puis on somme au niveau France entière -> data/processed/lits_sae_annuel.csv
"""

import os
import re
import glob
import shutil
import tempfile

import pandas as pd
import py7zr

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE, "data", "raw")
OUT_DIR = os.path.join(BASE, "data", "processed")

# prefix de bordereau -> colonne contenant les lits installés
BORDEREAUX = {
    "MCO": ["LIT_MCO", "LIT_MED", "LIT_CHI", "LIT_OBS"],
    "REA": ["LIT"],
    "PSY": ["CAP_HTP"],
    "SSR": ["LIT"],
    "SMR": ["LIT"],   # SMR = nouveau nom du SSR depuis la réforme SAE 2025
    "USLD": ["LIT"],
}
# renommage vers la discipline "officielle"
ALIAS = {"SMR": "ssr"}
DISCIPLINES_TOTAL = ["lits_mco", "lits_psy", "lits_ssr", "lits_usld"]
LIGNES_TOTAL = {"PSY": ("DIS", "TOT"), "SSR": ("GDE", "SSR_TOT"),
                "SMR": ("GDE", "SSR_TOT")}
FICHIER_RE = re.compile(r"^(MCO|REA|PSY|SSR|SMR|USLD)_(\d{4})([ar])?\.csv$", re.IGNORECASE)


def _read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep=";", encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep=";", encoding="latin-1", dtype=str, low_memory=False)


def _lits_pour_bordereau(df: pd.DataFrame, prefix: str) -> float:
    """Capacité du bordereau au niveau d'agrégation approprié."""
    if prefix in LIGNES_TOTAL:
        champ, code = LIGNES_TOTAL[prefix]
        df = df.loc[df[champ].eq(code)]
    if prefix == "REA":
        df = df.loc[df.UNI.isin(["REAADU", "REAENF"])]
    cols = BORDEREAUX[prefix]
    if prefix == "MCO" and "LIT_MCO" in df.columns:
        cols = ["LIT_MCO"]
    # fallback MCO = med + chi + obs ; sinon 1ère colonne dispo
    dispo = [c for c in cols if c in df.columns]
    if len(dispo) != len(cols):
        raise ValueError(f"{prefix} : colonnes de capacité incomplètes ({dispo})")
    total = df[dispo].apply(pd.to_numeric, errors="raise").sum(min_count=1).sum(min_count=1)
    if pd.isna(total) or total <= 0:
        raise ValueError(f"{prefix} : aucune capacité exploitable")
    return float(total)


def extract_annee(annee: int, archive: str) -> dict:
    """Extrait les CSV de lits d'une archive SAE et renvoie les totaux nationaux."""
    with py7zr.SevenZipFile(archive) as z:
        names = z.getnames()
        # dédoublonne par nom de fichier et préfère la version 'r' (révisée) à 'a'
        by_base = {}
        for n in names:
            b = os.path.basename(n)
            m = FICHIER_RE.match(b)
            if m:
                by_base.setdefault(b, n)
        # si MCO_2013a.csv et MCO_2013r.csv existent, garder le 'r'
        pref = {}
        for b, n in by_base.items():
            m = FICHIER_RE.match(b)
            cle = (m.group(1).upper(), m.group(2))
            if cle not in pref or b.lower().endswith("r.csv"):
                pref[cle] = n
        targets = [n for n in pref.values() if f"_{annee}" in os.path.basename(n)]
        if not targets:
            raise RuntimeError(f"Aucun CSV de lits trouvé dans {archive}")
        tmp = tempfile.mkdtemp(prefix=f"sae_{annee}_")
        try:
            z.extract(targets=targets, path=tmp)
            result = {"annee": annee}
            for n in targets:
                b = os.path.basename(n)
                m = FICHIER_RE.match(b)
                prefix = m.group(1).upper()
                chemin = None
                for root, _, files in os.walk(tmp):
                    if b in files:
                        chemin = os.path.join(root, b)
                        break
                df = _read_csv(chemin)
                cle = ALIAS.get(prefix, prefix.lower())
                if f"lits_{cle}" in result:
                    raise ValueError(f"SAE {annee} : deux bordereaux pour {cle}")
                result[f"lits_{cle}"] = _lits_pour_bordereau(df, prefix)
                if prefix == "MCO":
                    for champ in ("JOU_MCO", "JLI_MCO"):
                        total = pd.to_numeric(df[champ], errors="raise").sum(min_count=1)
                        if pd.isna(total) or total <= 0:
                            raise ValueError(f"SAE {annee} : {champ} inexploitable")
                        result[champ.lower()] = total
            if not set(DISCIPLINES_TOTAL).issubset(result):
                raise ValueError(f"SAE {annee} : bordereaux incomplets ({list(result)})")
            return result
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def archives_locales() -> dict:
    archives = {}
    for chemin in sorted(glob.glob(os.path.join(RAW_DIR, "sae_20*.7z"))):
        nom = os.path.basename(chemin)
        match = re.fullmatch(r"sae_(\d{4})(?:_.*)?\.7z", nom)
        if match and 2013 <= int(match[1]) <= 2025:
            annee = int(match[1])
            archives.setdefault(annee, chemin)
    return archives


def extract_mco_regional(annee, archive):
    """Rattache les capacités et journées MCO au FINESS géographique."""
    with py7zr.SevenZipFile(archive) as z, tempfile.TemporaryDirectory() as tmp:
        cibles = {}
        for prefix in ("ID", "MCO"):
            motif = re.compile(rf"^{prefix}_{annee}(r|a)?\.csv$", re.IGNORECASE)
            noms = [n for n in z.getnames() if motif.fullmatch(os.path.basename(n))]
            if not noms:
                raise ValueError(f"SAE {annee} : bordereau {prefix} introuvable")
            cibles[prefix] = min(noms, key=lambda n: not n.lower().endswith("r.csv"))
        z.extract(targets=list(cibles.values()), path=tmp)
        ident = _read_csv(os.path.join(tmp, cibles["ID"]))
        mco = _read_csv(os.path.join(tmp, cibles["MCO"]))

    ident.columns = ident.columns.str.upper()
    ident = ident[["FI", "REG"]].dropna().drop_duplicates()
    cols = ["LIT_MCO", "JOU_MCO", "JLI_MCO"]
    mco = mco[["FI", *cols]].merge(ident, on="FI", how="left", validate="one_to_one")
    if mco.REG.isna().any():
        raise ValueError(f"SAE {annee} : FINESS MCO sans territoire")
    mco[cols] = mco[cols].apply(pd.to_numeric, errors="raise")
    regional = mco.groupby("REG")[cols].sum(min_count=1).reset_index()
    regional.columns = regional.columns.str.lower()
    regional.insert(0, "annee", annee)
    return regional


def extract_all(sae_archives: dict) -> pd.DataFrame:
    if not sae_archives:
        raise ValueError("Aucune archive SAE exploitable dans data/raw")
    os.makedirs(OUT_DIR, exist_ok=True)
    lignes = []
    for annee in sorted(sae_archives):
        print(f"  [extract] SAE {annee} ...")
        lignes.append(extract_annee(annee, sae_archives[annee]))
    df = pd.DataFrame(lignes).sort_values("annee").reset_index(drop=True)
    df["lits_total"] = df[DISCIPLINES_TOTAL].sum(axis=1)
    derniere = max(sae_archives)
    mco = extract_mco_regional(derniere, sae_archives[derniere])
    cols_mco = ["lit_mco", "jou_mco", "jli_mco"]
    if not mco[cols_mco].sum().eq(df.iloc[-1][["lits_mco", "jou_mco", "jli_mco"]].to_numpy()).all():
        raise ValueError("Les totaux MCO par territoire diffèrent du total national")
    mco.to_csv(os.path.join(OUT_DIR, "mco_regional_reference.csv"), index=False)
    out = os.path.join(OUT_DIR, "lits_sae_annuel.csv")
    df.to_csv(out, index=False)
    print(f"[extract] -> {out}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    extract_all(archives_locales())
