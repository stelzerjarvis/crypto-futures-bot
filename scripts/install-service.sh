#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLIST_NAME="com.jarvis.crypto-bot.plist"
SOURCE_PLIST="${SCRIPT_DIR}/${PLIST_NAME}"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET_PLIST="${TARGET_DIR}/${PLIST_NAME}"
SERVICE_LABEL="com.jarvis.crypto-bot"
ENV_FILE="${PROJECT_DIR}/.env"

if [[ ! -f "$SOURCE_PLIST" ]]; then
  echo "Missing plist at $SOURCE_PLIST" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"
launchctl bootout "gui/$UID" "$TARGET_PLIST" >/dev/null 2>&1 || true
cp "$SOURCE_PLIST" "$TARGET_PLIST"

if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    if [[ "$line" != *=* ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    key="$(echo "$key" | xargs)"
    value="${value%$'\r'}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    launchctl setenv "$key" "$value"
  done < "$ENV_FILE"
fi

launchctl bootstrap "gui/$UID" "$TARGET_PLIST"
launchctl enable "gui/$UID/${SERVICE_LABEL}"
echo "Service installed: ${SERVICE_LABEL}"
