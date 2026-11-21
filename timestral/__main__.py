"""Point d'entrée CLI : `python -m timestral <commande> ...`

Commandes :
  run lits [--skip-fetch]        pipeline capacité hospitalière (main.py)
  run mensuel [--skip-fetch]     pipeline mensuel SARS-CoV-2 / prévalences (main_mensuel.py)
  run pharma                     pipeline antibiotiques pédiatriques (main_pharma.py)
  dashboard lits|mensuel|pharma  régénère un seul tableau de bord
  test                           lance la suite de tests (unittest discover)
  recherche <nom> [args...]      expérience pharma : volumes, delai, signaux, calendrier,
                                 archives, familles (main_<nom>_pharma.py)
  publier                        régénère tout + tests + push + déploiement Vercel
"""

import argparse
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]

SCRIPTS = {"lits": "main.py", "mensuel": "main_mensuel.py", "pharma": "main_pharma.py"}
RECHERCHES = {
    "volumes": "main_recherche_pharma.py",
    "delai": "main_delai_pharma.py",
    "signaux": "main_signaux_pharma.py",
    "calendrier": "main_calendrier_pharma.py",
    "archives": "main_archives_pharma.py",
    "familles": "main_familles_pharma.py",
}


def _run_script(nom, args):
    """Exécute un main_*.py du dépôt avec les arguments donnés."""
    import runpy
    sys.argv = [nom] + list(args)
    runpy.run_path(str(RACINE / nom), run_name="__main__")


def _dashboard(cible):
    sys.path.insert(0, str(RACINE / "src"))
    if cible == "lits":
        import plots_interactifs
        plots_interactifs.generer_dashboard()
    elif cible == "mensuel":
        import plots_mensuel
        plots_mensuel.generer_dashboard_mensuel()
    elif cible == "pharma":
        _run_script("main_pharma.py", [])


def _test():
    subprocess.run([str(RACINE / ".venv-timestral" / "bin" / "python"), "-m", "unittest",
                    "discover", "-s", "tests"], cwd=RACINE, check=True)


def _publier():
    subprocess.run(["bash", str(RACINE / "scripts" / "publier.sh")], cwd=RACINE, check=True)


def construire_parser():
    p = argparse.ArgumentParser(prog="timestral", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sous = p.add_subparsers(dest="commande", required=True)

    run = sous.add_parser("run", help="lance un pipeline complet")
    run.add_argument("cible", choices=sorted(SCRIPTS))
    run.add_argument("args", nargs=argparse.REMAINDER, help="options transmises (ex. --skip-fetch)")

    dash = sous.add_parser("dashboard", help="régénère un tableau de bord")
    dash.add_argument("cible", choices=["lits", "mensuel", "pharma"])

    sous.add_parser("test", help="lance la suite de tests")

    rech = sous.add_parser("recherche", help="expérience pharma (main_*_pharma.py)")
    rech.add_argument("nom", choices=sorted(RECHERCHES))
    rech.add_argument("args", nargs=argparse.REMAINDER)

    sous.add_parser("publier", help="régénère + teste + push + déploie sur Vercel")
    return p


def main(argv=None):
    args = construire_parser().parse_args(argv)
    if args.commande == "run":
        _run_script(SCRIPTS[args.cible], args.args)
    elif args.commande == "dashboard":
        _dashboard(args.cible)
    elif args.commande == "test":
        _test()
    elif args.commande == "recherche":
        _run_script(RECHERCHES[args.nom], args.args)
    elif args.commande == "publier":
        _publier()


if __name__ == "__main__":
    main()
