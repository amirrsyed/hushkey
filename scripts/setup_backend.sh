#!/usr/bin/env bash
# DEVELOPER setup: editable install in backend/.venv (code changes take effect on restart) and
# point the app at it. Regular users: run ./install.sh instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# macOS ships python3 3.9 (too old) and no `pip` command; look for a newer Python.
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "✗ Hushkey needs Python 3.10 or newer, and none was found."
  echo "  Install one of these, then run this script again:"
  echo "    • Homebrew:   brew install python@3.12"
  echo "    • Installer:  https://www.python.org/downloads/macos/"
  exit 1
fi
echo "Using $PY ($("$PY" --version))"

cd "$ROOT/backend"
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/hushkey download

CMD="cd \"$ROOT/backend\" && .venv/bin/hushkey serve"
defaults write dev.hushkey.app backendCommand -string "$CMD"
echo
echo "✓ Backend ready. The Hushkey app is set to start it with:"
echo "    $CMD"
echo "  (Hushkey ▸ Settings ▸ Backend shows/changes this.)"
