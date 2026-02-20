#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLIST_NAME="com.jarvis.crypto-bot.plist"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET_PLIST="${TARGET_DIR}/${PLIST_NAME}"
SERVICE_LABEL="com.jarvis.crypto-bot"
ENV_FILE="${PROJECT_DIR}/.env"

launchctl bootout "gui/$UID" "$TARGET_PLIST" >/dev/null 2>&1 || true

if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    if [[ "$line" != *=* ]]; then
      continue
    fi
    key="${line%%=*}"
    key="$(echo "$key" | xargs)"
    launchctl unsetenv "$key" || true
  done < "$ENV_FILE"
fi

if [[ -f "$TARGET_PLIST" ]]; then
  rm -f "$TARGET_PLIST"
fi

echo "Service uninstalled: ${SERVICE_LABEL}"
