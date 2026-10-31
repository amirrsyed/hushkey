#!/usr/bin/env bash
# Build build/Hushkey.dmg: the app plus an Applications shortcut, ready to attach to a GitHub release.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/build_app.sh" release
STAGE="$ROOT/build/dmg"
rm -rf "$STAGE" "$ROOT/build/Hushkey.dmg"
mkdir -p "$STAGE"
cp -R "$ROOT/build/Hushkey.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cat > "$STAGE/READ ME FIRST.txt" <<'EOF'
1. Drag Hushkey into Applications.
2. First launch: right-click Hushkey in Applications ▸ Open ▸ Open
   (this build isn't notarized by Apple, so a plain double-click is blocked the first time).
   If macOS says the app is "damaged", run in Terminal:
       xattr -dr com.apple.quarantine /Applications/Hushkey.app
3. In the welcome window: allow Camera and Accessibility, then click "Install model server"
   (one-time, ≈1.5 GB, a few minutes). Then hold Right ⌥ and silently mouth.
EOF
hdiutil create -volname Hushkey -srcfolder "$STAGE" -ov -format UDZO "$ROOT/build/Hushkey.dmg" >/dev/null
echo "Built $ROOT/build/Hushkey.dmg"
