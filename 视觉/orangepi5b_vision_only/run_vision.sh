#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CAMERA="${1:-/dev/video0}"
exec .venv/bin/python board/steel_ball_vision.py --config config/vision.yaml --camera-device "$CAMERA" "${@:2}"
