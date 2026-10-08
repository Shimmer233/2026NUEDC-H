#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CAMERA="${1:-/dev/video0}"
MODEL="${2:-models/steel_ball_yolov5n_int8.rknn}"
exec .venv/bin/python board/steel_ball_tracker.py --config config/board.yaml --camera-device "$CAMERA" --model "$MODEL" --detection-only
