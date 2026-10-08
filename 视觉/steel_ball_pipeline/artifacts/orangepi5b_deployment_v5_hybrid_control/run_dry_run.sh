#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CAMERA="${1:-/dev/video0}"
TARGET="${2:-12.5}"
exec .venv/bin/python board/steel_ball_tracker.py --config config/board.yaml --camera-device "$CAMERA" --control-mode dry-run --target-cm "$TARGET"
