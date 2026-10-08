#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CAMERA="${1:-/dev/video0}"
ANGLE="${2:-0.15}"
ARM="${3:-}"
if [[ "$ARM" != "--arm" ]]; then
  echo "Motor movement blocked. Pass --arm as the third argument after completing all safety gates." >&2
  exit 2
fi
exec .venv/bin/python board/steel_ball_tracker.py --config config/board.yaml --camera-device "$CAMERA" --control-mode identify --identify-angle-deg "$ANGLE" --arm
