#!/usr/bin/env bash
# One-command install from the source folder:
#   1. installs the model server into ~/Library/Application Support/Hushkey (Python found or fetched)
#   2. builds Hushkey.app and copies it to /Applications (or ~/Applications)
#   3. clears stale camera/accessibility grants left by earlier builds, then opens the app
# Safe to re-run to update. Afterwards this folder is no longer needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
APP_ID="dev.hushkey.app"

command -v swift >/dev/null 2>&1 || { echo "✗ Apple's build tools are missing. Run:  xcode-select --install   then re-run ./install.sh"; exit 1; }

echo "==> Stopping any running Hushkey"
osascript -e "tell application id \"$APP_ID\" to quit" >/dev/null 2>&1 || true
lsof -ti tcp:8765 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1

echo "==> [1/3] Model server"
"$ROOT/scripts/install_backend.sh"

echo "==> [2/3] App"
"$ROOT/scripts/build_app.sh" release
DEST="/Applications"
[ -w "$DEST" ] || { DEST="$HOME/Applications"; mkdir -p "$DEST"; }
rm -rf "$DEST/Hushkey.app"
cp -R "$ROOT/build/Hushkey.app" "$DEST/"

echo "==> [3/3] Permissions"
# A rebuilt app has a new signature; old grants then look "on" in System Settings but don't apply.
tccutil reset Accessibility "$APP_ID" >/dev/null 2>&1 || true
tccutil reset Camera "$APP_ID" >/dev/null 2>&1 || true

open "$DEST/Hushkey.app"
echo
echo "✓ Hushkey installed in $DEST and opened (mouth icon in the menu bar)."
echo "  Grant Camera + Accessibility when asked. To update later: re-run ./install.sh. To remove: ./uninstall.sh"
