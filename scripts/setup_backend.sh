#!/usr/bin/env bash
# One-time backend setup on a Mac: virtualenv, install, download pretrained weights.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
hushkey download
echo
echo "Backend ready. In the app's Settings ▸ Backend, set the command to:"
echo "  cd \"$ROOT/backend\" && .venv/bin/hushkey serve"
