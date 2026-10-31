#!/usr/bin/env bash
# Build Hushkey.app without Xcode project files: swift build + bundle + ad-hoc sign.
# Usage: scripts/build_app.sh [release|debug]
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
# Ad-hoc signature with the camera entitlement + hardened runtime. TCC permissions are keyed
# to the signature, so re-granting may be needed after rebuilding; sign with a stable
# identity (SIGN_ID="Apple Development: …") to avoid that.
codesign --force --options runtime --entitlements Resources/Hushkey.entitlements --sign "${SIGN_ID:--}" "$APP"
echo "Built $APP"
echo "Run:  open \"$APP\""
