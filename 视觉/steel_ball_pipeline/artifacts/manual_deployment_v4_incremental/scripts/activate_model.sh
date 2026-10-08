#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

APP_DIR="${1:-$HOME/orangepi5b_vision_only}"
CONFIRM="${2:-}"
CAMERA="${3:-${STEEL_BALL_CAMERA:-/dev/video0}}"
if [[ "$CONFIRM" != "--confirm" ]]; then
  echo "Usage: $0 [APP_DIR] --confirm [CAMERA]" >&2
  exit 2
fi
if [[ ! -d "$APP_DIR" ]]; then
  echo "Application directory is missing: $APP_DIR" >&2
  exit 2
fi
APP_DIR="$(readlink -f "$APP_DIR")"
PYTHON_BIN="$APP_DIR/.venv/bin/python"
STAGE_MANIFEST="$APP_DIR/models/steel_ball_v4_incremental.stage.json"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Application Python is missing: $PYTHON_BIN" >&2
  exit 2
fi
ensure_camera_idle "$CAMERA"
"$PYTHON_BIN" "$SCRIPT_DIR/verify_stage.py" "$APP_DIR" "$STAGE_MANIFEST"
"$PYTHON_BIN" "$SCRIPT_DIR/check_activation_gate.py" \
  "$APP_DIR/runs/manual_v4" \
  --stage-manifest "$STAGE_MANIFEST" \
  --camera "$CAMERA"

declare -a ACTIVE_RELATIVES=(
  "config/vision.yaml"
  "models/steel_ball_yolov5n_int8.rknn"
  "models/steel_ball_yolov5n_int8.json"
  "models/steel_ball_yolov5n_fp16.rknn"
  "models/steel_ball_yolov5n_fp16.json"
  "models/SHA256SUMS.txt"
)
declare -a STAGED_RELATIVES=(
  "models/steel_ball_yolov5n_v4_incremental_int8.rknn"
  "models/steel_ball_yolov5n_v4_incremental_int8.json"
  "models/steel_ball_yolov5n_v4_incremental_fp16.rknn"
  "models/steel_ball_yolov5n_v4_incremental_fp16.json"
)
for relative in "${ACTIVE_RELATIVES[@]}" "${STAGED_RELATIVES[@]}"; do
  if [[ ! -f "$APP_DIR/$relative" ]]; then
    echo "Required activation file is missing: $APP_DIR/$relative" >&2
    exit 2
  fi
done

BACKUP_ROOT="$APP_DIR/backups"
BACKUP_DIR="$BACKUP_ROOT/pre_v4_$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -e "$BACKUP_DIR" ]]; then
  echo "Backup directory already exists: $BACKUP_DIR" >&2
  exit 2
fi
mkdir -p "$BACKUP_ROOT" "$BACKUP_DIR"
cp -a "$APP_DIR/models" "$APP_DIR/config" "$BACKUP_DIR/"
diff -qr "$APP_DIR/models" "$BACKUP_DIR/models"
diff -qr "$APP_DIR/config" "$BACKUP_DIR/config"

"$PYTHON_BIN" "$SCRIPT_DIR/make_backup_manifest.py" \
  "$BACKUP_DIR" \
  --purpose "complete models/config rollback point before v4 activation"
"$PYTHON_BIN" "$SCRIPT_DIR/verify_backup.py" "$BACKUP_DIR"

# Publish the verified rollback pointer before touching any active file. It remains
# usable even if power is lost during the following multi-file replacement.
pointer_tmp="$BACKUP_ROOT/.LAST_V4_BACKUP.tmp.$$"
printf '%s\n' "$BACKUP_DIR" >"$pointer_tmp"
mv -f "$pointer_tmp" "$BACKUP_ROOT/LAST_V4_BACKUP"
sync

CONFIG_TMP="$APP_DIR/config/.vision.yaml.v4.$$"
INT8_TMP="$APP_DIR/models/.steel_ball_yolov5n_int8.rknn.v4.$$"
INT8_JSON_TMP="$APP_DIR/models/.steel_ball_yolov5n_int8.json.v4.$$"
FP16_TMP="$APP_DIR/models/.steel_ball_yolov5n_fp16.rknn.v4.$$"
FP16_JSON_TMP="$APP_DIR/models/.steel_ball_yolov5n_fp16.json.v4.$$"
SUMS_TMP="$APP_DIR/models/.SHA256SUMS.txt.v4.$$"

install -m 0644 "$APP_DIR/models/steel_ball_yolov5n_v4_incremental_int8.rknn" "$INT8_TMP"
install -m 0644 "$APP_DIR/models/steel_ball_yolov5n_v4_incremental_int8.json" "$INT8_JSON_TMP"
install -m 0644 "$APP_DIR/models/steel_ball_yolov5n_v4_incremental_fp16.rknn" "$FP16_TMP"
install -m 0644 "$APP_DIR/models/steel_ball_yolov5n_v4_incremental_fp16.json" "$FP16_JSON_TMP"

"$PYTHON_BIN" - "$APP_DIR/config/vision.yaml" "$CONFIG_TMP" <<'PY'
import sys
from pathlib import Path

import yaml

source, destination = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text(encoding="utf-8"))
protected = {
    key: config.get(key)
    for key in ("calibration", "geometry", "camera", "detection", "coordinate")
}
config["model"] = "models/steel_ball_yolov5n_int8.rknn"
config.setdefault("roi", {}).update(
    {"x": 10, "y": 150, "width": 600, "height": 105, "output_width": 470, "output_height": 110}
)
if any(config.get(key) != value for key, value in protected.items()):
    raise SystemExit("activation attempted to modify calibration, geometry, camera, or detection settings")
destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

INT8_HASH="$(sha256sum "$INT8_TMP" | cut -d' ' -f1)"
FP16_HASH="$(sha256sum "$FP16_TMP" | cut -d' ' -f1)"
printf '%s  %s\n%s  %s\n' \
  "$FP16_HASH" "models/steel_ball_yolov5n_fp16.rknn" \
  "$INT8_HASH" "models/steel_ball_yolov5n_int8.rknn" >"$SUMS_TMP"

activation_started=0
restore_backup() {
  cp -a "$BACKUP_DIR/models/." "$APP_DIR/models/"
  cp -a "$BACKUP_DIR/config/." "$APP_DIR/config/"
  "$PYTHON_BIN" "$SCRIPT_DIR/verify_backup.py" \
    "$BACKUP_DIR" \
    --restored-root "$APP_DIR"
  (
    cd "$APP_DIR"
    sha256sum -c models/SHA256SUMS.txt
  )
}
on_failure() {
  local status=$?
  trap - ERR INT TERM
  if [[ "$activation_started" -eq 1 ]]; then
    echo "Activation failed; restoring the complete pre-v4 models/config backup." >&2
    restore_backup || echo "Automatic restore also failed; use rollback_model.sh with: $BACKUP_DIR" >&2
  fi
  rm -f "$CONFIG_TMP" "$INT8_TMP" "$INT8_JSON_TMP" "$FP16_TMP" "$FP16_JSON_TMP" "$SUMS_TMP"
  exit "$status"
}
trap on_failure ERR INT TERM

activation_started=1
mv -f "$INT8_JSON_TMP" "$APP_DIR/models/steel_ball_yolov5n_int8.json"
mv -f "$FP16_TMP" "$APP_DIR/models/steel_ball_yolov5n_fp16.rknn"
mv -f "$FP16_JSON_TMP" "$APP_DIR/models/steel_ball_yolov5n_fp16.json"
mv -f "$SUMS_TMP" "$APP_DIR/models/SHA256SUMS.txt"
mv -f "$CONFIG_TMP" "$APP_DIR/config/vision.yaml"
# The active INT8 is the final commit file. Before this move, a power loss leaves
# the old active model plus a valid LAST_V4_BACKUP pointer.
mv -f "$INT8_TMP" "$APP_DIR/models/steel_ball_yolov5n_int8.rknn"
sync

(
  cd "$APP_DIR"
  sha256sum -c models/SHA256SUMS.txt
)
"$PYTHON_BIN" - "$APP_DIR/config/vision.yaml" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert config["model"] == "models/steel_ball_yolov5n_int8.rknn"
expected_roi = {
    "x": 10,
    "y": 150,
    "width": 600,
    "height": 105,
    "output_width": 470,
    "output_height": 110,
}
assert all(config["roi"].get(key) == value for key, value in expected_roi.items())
print("Active model configuration verified")
PY

SMOKE_DIR="$APP_DIR/runs/manual_v4/activate_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
run_smoke_test "$APP_DIR" "$CAMERA" "$SMOKE_DIR/run.log"

activation_started=0
trap - ERR INT TERM

echo "V4 incremental model activated and smoke-tested."
echo "Rollback point: $BACKUP_DIR"
echo "Start normally with: cd '$APP_DIR' && ./run_vision.sh '$CAMERA'"
