#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CAMERA="${1:-/dev/video0}"
TARGET="${2:-12.5}"
ARM="${3:-}"
if [[ "$ARM" != "--arm" ]]; then
  echo "Balance blocked. Pass --arm as the third argument after zero and gain verification." >&2
  exit 2
fi
exec .venv/bin/python board/steel_ball_tracker.py --config config/board.yaml --camera-device "$CAMERA" --control-mode balance --target-cm "$TARGET" --arm
