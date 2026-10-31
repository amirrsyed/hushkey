#!/usr/bin/env bash
# Install (or update) the Hushkey model server into a fixed, per-user location:
#     ~/Library/Application Support/Hushkey/venv
# so the app works no matter where the source was unzipped, and the source folder can be deleted.
#
# Works from a source checkout (scripts/install_backend.sh) and from inside the app bundle
# (Hushkey.app/Contents/Resources/install_backend.sh, run by the in-app "Install" button).
#
# Python: uses Python 3.10–3.13 if one is installed; otherwise downloads `uv`, which fetches a
# private Python 3.12 just for Hushkey. Nothing is installed system-wide.
#
# Env: HUSHKEY_HOME (default ~/Library/Application Support/Hushkey), HUSHKEY_SKIP_DOWNLOAD=1 (skip weights).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/backend/pyproject.toml" ]; then SRC="$HERE/backend"        # inside Hushkey.app
elif [ -f "$HERE/../backend/pyproject.toml" ]; then SRC="$HERE/../backend" # source checkout
else echo "✗ can't find the backend source next to $0" >&2; exit 1; fi

if [ -z "${HUSHKEY_HOME:-}" ]; then
  if [ "$(uname)" = "Darwin" ]; then HUSHKEY_HOME="$HOME/Library/Application Support/Hushkey"; else HUSHKEY_HOME="$HOME/.hushkey"; fi
fi
VENV="$HUSHKEY_HOME/venv"
mkdir -p "$HUSHKEY_HOME"
echo "==> Installing the Hushkey model server into: $VENV"

# Build from a temporary copy: pip writes build files into the source tree, which must not
# happen inside a signed .app bundle.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp -R "$SRC" "$WORK/backend"
rm -rf "$WORK/backend/.venv" "$WORK/backend/build" "$WORK/backend"/*.egg-info "$WORK/backend/.pytest_cache"
find "$WORK/backend" -name __pycache__ -prune -exec rm -rf {} +

PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done

rm -rf "$VENV"
if [ -n "$PY" ]; then
  echo "==> Using $PY ($("$PY" --version 2>&1))"
  "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  echo "==> Installing PyTorch and dependencies (first time: a few minutes)…"
  "$VENV/bin/python" -m pip install --progress-bar off "$WORK/backend"
else
  echo "==> No Python 3.10–3.13 found; fetching uv to provide a private Python 3.12…"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$WORK/uv" sh >/dev/null
  export UV_PYTHON_INSTALL_DIR="$HUSHKEY_HOME/python"
  "$WORK/uv/uv" venv --python 3.12 "$VENV"
  echo "==> Installing PyTorch and dependencies (first time: a few minutes)…"
  "$WORK/uv/uv" pip install --python "$VENV/bin/python" "$WORK/backend"
fi

if [ "${HUSHKEY_SKIP_DOWNLOAD:-0}" != "1" ]; then
  echo "==> Downloading pretrained lip-reading model (~1 GB, skipped if already present)…"
  HUSHKEY_HOME="$HUSHKEY_HOME" "$VENV/bin/python" -m hushkey.cli download
fi

# Tell the app how to start the server (python -m avoids script-shebang issues with the space
# in "Application Support").
CMD="\"$VENV/bin/python\" -m hushkey.cli serve"
if command -v defaults >/dev/null 2>&1; then
  defaults write dev.hushkey.app backendCommand -string "$CMD"
fi
echo
echo "✓ Model server installed. Hushkey will start it with:"
echo "    $CMD"
