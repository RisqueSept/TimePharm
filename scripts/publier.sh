#!/usr/bin/env bash
# Régénère, teste et publie les tableaux statiques sur le projet Vercel lié.
# Les commits et le push éventuels se font séparément.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-timestral/bin/python
VERCEL=~/.npm-global/bin/vercel
[ -x "$VERCEL" ] || VERCEL="$(command -v vercel)"
if [[ "${1:-}" != "--sorties-existantes" ]]; then
  "$PY" main.py --skip-fetch
  "$PY" main_mensuel.py --skip-fetch
  "$PY" main_pharma.py --refresh-sentinelles
  "$PY" -m unittest discover -s tests
fi

SITE=$(mktemp -d)
trap 'rm -rf "$SITE"' EXIT
mkdir "$SITE/.vercel"
cp .vercel/project.json "$SITE/.vercel/"
cp -r outputs vercel.json "$SITE/"
( cd "$SITE" && "$VERCEL" deploy --prod --yes --scope risced )
