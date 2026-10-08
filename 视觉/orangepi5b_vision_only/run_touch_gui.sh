#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CAMERA="${1:-${STEEL_BALL_CAMERA:-/dev/video0}}"
shift || true

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python. Run ./install.sh first." >&2
  exit 1
fi

exec .venv/bin/python examples/read_position_velocity.py \
  --camera-device "$CAMERA" \
  --gui \
  "$@"
