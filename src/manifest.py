"""Manifeste d'exécution : hash des entrées, révision du modèle et du code, paramètres.

Chaque pipeline écrit `outputs/manifest.json` (dernier run) et une copie historisée
dans `outputs/manifests/`, pour pouvoir répondre à « pourquoi les résultats ont changé ? ».
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def hasher_fichier(chemin, taille_bloc=1 << 20):
    """Empreinte SHA-256 d'un fichier, lue par blocs."""
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(taille_bloc), b""):
            h.update(bloc)
    return h.hexdigest()


def entrees_dossier(dossier):
    """Fichiers d'un dossier (récursif), triés, sous forme de chemins absolus."""
    dossier = Path(dossier)
    if not dossier.exists():
        return []
    return sorted(str(p) for p in dossier.rglob("*") if p.is_file())


def revision_code(racine):
    """Commit git courant, ou None hors dépôt git."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(racine),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def ecrire_manifest(racine, run, entrees, parametres, destination=None):
    """Écrit le manifeste du run et renvoie son contenu.

    `entrees` : chemins des fichiers de données consommés (hashés).
    `parametres` : dictionnaire JSON-sérialisable (horizons, délais, options).
    """
    try:
        from .inference_timestral import MODELE, REVISION
    except ImportError:  # import « plat » (sys.path contient src/, ex. main.py)
        from inference_timestral import MODELE, REVISION

    racine = Path(racine)
    entrees = [str(Path(e)) for e in entrees]
    manifest = {
        "run": run,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modele": MODELE,
        "revision_modele": REVISION,
        "revision_code": revision_code(racine),
        "entrees": {},
        "parametres": parametres,
    }
    for chemin in entrees:
        p = Path(chemin)
        cle = str(p.relative_to(racine)) if p.is_relative_to(racine) else p.name
        manifest["entrees"][cle] = hasher_fichier(p) if p.is_file() else None

    destination = Path(destination) if destination else racine / "outputs" / "manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    historique = racine / "outputs" / "manifests"
    historique.mkdir(parents=True, exist_ok=True)
    tampon = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (historique / f"{run}_{tampon}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def lire_manifest(racine):
    """Dernier manifeste de run (`outputs/manifest.json`), ou None s'il est absent/illisible."""
    p = Path(racine) / "outputs" / "manifest.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
