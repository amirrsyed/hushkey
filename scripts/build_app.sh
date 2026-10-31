#!/usr/bin/env bash
# Build Hushkey.app: swift build → bundle (Info.plist, model-server installer + source) → sign.
# Usage: scripts/build_app.sh [release|debug]
#   SIGN_ID="Developer ID Application: …"  sign with a real identity (default: ad-hoc "-")
set -euo pipefail
CONFIG="${1:-release}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/app"
if ! swift build -c "$CONFIG"; then
  echo
  echo "✗ Swift build failed — Hushkey.app was NOT created. Copy the error lines above (they start with the file path and \"error:\") and report them." >&2
  exit 1
fi
BIN="$(swift build -c "$CONFIG" --show-bin-path)/Hushkey"
APP="$ROOT/build/Hushkey.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/Hushkey"
cp Resources/Info.plist "$APP/Contents/Info.plist"

# Ship the model-server installer + Python source inside the app, so a downloaded Hushkey.app can
# install its own backend from the onboarding window (no Terminal, no source checkout).
RES="$APP/Contents/Resources"
cp "$ROOT/scripts/install_backend.sh" "$RES/install_backend.sh"
mkdir -p "$RES/backend"
cp -R "$ROOT/backend/hushkey" "$ROOT/backend/pyproject.toml" "$ROOT/backend/README.md" "$RES/backend/"
rm -rf "$RES/backend"/*.egg-info
find "$RES/backend" -name __pycache__ -prune -exec rm -rf {} +

# Hardened runtime + camera entitlement. With the default ad-hoc signature every build counts as a
# new app for macOS privacy permissions; install.sh resets them so you're asked again cleanly.
codesign --force --options runtime --entitlements Resources/Hushkey.entitlements --sign "${SIGN_ID:--}" "$APP"
echo "Built $APP"
