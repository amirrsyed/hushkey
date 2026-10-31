#!/usr/bin/env bash
# Remove Hushkey: the app, its settings and permissions; optionally models + your recordings.
set -uo pipefail
APP_ID="dev.hushkey.app"
DATA="$HOME/Library/Application Support/Hushkey"

osascript -e "tell application id \"$APP_ID\" to quit" >/dev/null 2>&1
lsof -ti tcp:8765 2>/dev/null | xargs kill 2>/dev/null
rm -rf "/Applications/Hushkey.app" "$HOME/Applications/Hushkey.app"
defaults delete "$APP_ID" >/dev/null 2>&1
tccutil reset All "$APP_ID" >/dev/null 2>&1
echo "✓ Removed the app, its settings and permissions."

if [ -d "$DATA" ]; then
  echo "Data folder: $DATA ($(du -sh "$DATA" 2>/dev/null | cut -f1))"
  echo "  contains the model server, downloaded models, personal adapters and enrollment clips."
  read -r -p "Delete it too? [y/N] " a
  if [[ "$a" =~ ^[Yy] ]]; then rm -rf "$DATA" && echo "✓ Deleted."; else echo "Kept."; fi
fi
