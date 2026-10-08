#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CAMERA="${1:-/dev/video0}"
exec .venv/bin/python examples/read_position_velocity.py --camera-device "$CAMERA" "${@:2}"
