#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
APP_DIR="${1:-$HOME/orangepi5b_vision_only}"
SCENARIO="${2:-}"
DURATION="${3:-60}"
CAMERA="${4:-/dev/video0}"

case "$SCENARIO" in
  normal|fast|empty) ;;
  *)
    echo "Usage: $0 [APP_DIR] {normal|fast|empty} [DURATION_SECONDS] [CAMERA]" >&2
    exit 2
    ;;
esac
if [[ ! -d "$APP_DIR" ]]; then
  echo "Application directory is missing: $APP_DIR" >&2
  exit 2
fi
APP_DIR="$(readlink -f "$APP_DIR")"
PYTHON_BIN="$APP_DIR/.venv/bin/python"
STAGED_CONFIG="$APP_DIR/config/vision_v4_incremental_staged.yaml"
STAGE_MANIFEST="$APP_DIR/models/steel_ball_v4_incremental.stage.json"
if [[ ! -x "$PYTHON_BIN" || ! -f "$STAGED_CONFIG" || ! -f "$STAGE_MANIFEST" ]]; then
  echo "Run stage_model.sh before testing." >&2
  exit 2
fi
"$PYTHON_BIN" "$SCRIPT_DIR/verify_stage.py" "$APP_DIR" "$STAGE_MANIFEST"
"$PYTHON_BIN" - "$DURATION" <<'PY'
import sys

value = float(sys.argv[1])
if value <= 0:
    raise SystemExit("duration must be positive")
PY
ensure_camera_idle "$CAMERA"

RUN_NAME="$(date -u +%Y%m%dT%H%M%SZ)_${SCENARIO}"
RUN_DIR="$APP_DIR/runs/manual_v4/$RUN_NAME"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
RUN_CONFIG="$RUN_DIR/vision.yaml"
TRAJECTORY_RELATIVE="runs/manual_v4/$RUN_NAME/trajectory.csv"
RUN_LOG="$RUN_DIR/run.log"

"$PYTHON_BIN" - "$STAGE_MANIFEST" "$RUN_DIR/attempt.json" "$SCENARIO" "$CAMERA" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest, output = map(Path, sys.argv[1:3])
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "scenario": sys.argv[3],
    "camera": sys.argv[4],
    "stage_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
}
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY

"$PYTHON_BIN" - "$STAGED_CONFIG" "$RUN_CONFIG" "$TRAJECTORY_RELATIVE" <<'PY'
import sys
from pathlib import Path

import yaml

source, destination, trajectory = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config.setdefault("output", {})["csv"] = trajectory.as_posix()
destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

echo "Starting $SCENARIO test for $DURATION seconds with the staged model."
set +e
(
  cd "$APP_DIR"
  "$PYTHON_BIN" board/steel_ball_vision.py \
    --config "$RUN_CONFIG" \
    --camera-device "$CAMERA" \
    --headless \
    --duration "$DURATION"
) 2>&1 | tee "$RUN_LOG"
pipeline_status=("${PIPESTATUS[@]}")
set -e
if [[ "${pipeline_status[0]}" -ne 0 || "${pipeline_status[1]}" -ne 0 ]]; then
  echo "Staged test failed; inspect: $RUN_LOG" >&2
  exit 2
fi

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_run.py" \
  --csv "$RUN_DIR/trajectory.csv" \
  --log "$RUN_LOG" \
  --stage-manifest "$STAGE_MANIFEST" \
  --camera "$CAMERA" \
  --scenario "$SCENARIO" \
  --requested-duration "$DURATION"
