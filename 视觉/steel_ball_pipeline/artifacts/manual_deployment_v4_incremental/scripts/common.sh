#!/usr/bin/env bash

VISION_PROCESS_PATTERN='board/steel_ball_vision.py|examples/read_position_velocity.py'
VISION_ERROR_PATTERN='Traceback|YOLO inference warning|RuntimeError'

ensure_camera_idle() {
  local camera="$1"
  local processes=""
  local fuser_output=""
  local fuser_status=0

  if [[ ! -e "$camera" ]]; then
    echo "Camera device is missing: $camera" >&2
    return 2
  fi

  processes="$(pgrep -af "$VISION_PROCESS_PATTERN" || true)"
  if [[ -n "$processes" ]]; then
    echo "A vision process is still running:" >&2
    printf '%s\n' "$processes" >&2
    echo "Exit that process normally, then retry." >&2
    return 2
  fi

  if ! command -v fuser >/dev/null 2>&1; then
    echo "The fuser command is required to verify camera ownership." >&2
    return 2
  fi

  set +e
  fuser_output="$(sudo fuser -v -- "$camera" 2>&1)"
  fuser_status=$?
  set -e
  if [[ "$fuser_status" -eq 0 ]]; then
    echo "The camera is still in use:" >&2
    printf '%s\n' "$fuser_output" >&2
    return 2
  fi
  if [[ "$fuser_status" -ne 1 ]] || grep -qiE 'sudo:|permission denied|not found' <<<"$fuser_output"; then
    echo "Unable to verify that the camera is idle:" >&2
    printf '%s\n' "$fuser_output" >&2
    return 2
  fi
  echo "Camera is idle: $camera"
}

require_clean_log() {
  local log_path="$1"
  if grep -nE "$VISION_ERROR_PATTERN" "$log_path"; then
    echo "Runtime errors were found in: $log_path" >&2
    return 2
  fi
}

run_smoke_test() {
  local app_dir="$1"
  local camera="$2"
  local log_path="$3"
  local python_bin="$app_dir/.venv/bin/python"
  local run_dir=""
  local smoke_config=""
  local trajectory_path=""
  local performance_path=""

  run_dir="$(dirname -- "$log_path")"
  smoke_config="$run_dir/vision.yaml"
  trajectory_path="$run_dir/trajectory.csv"
  performance_path="$run_dir/trajectory.performance.json"
  mkdir -p "$run_dir"
  "$python_bin" - \
    "$app_dir/config/vision.yaml" \
    "$smoke_config" \
    "$trajectory_path" <<'PY'
import sys
from pathlib import Path

import yaml

source, destination, trajectory = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config.setdefault("output", {})["csv"] = str(trajectory)
destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
  if ! (
    cd "$app_dir"
    "$python_bin" board/steel_ball_vision.py \
      --config "$smoke_config" \
      --camera-device "$camera" \
      --headless \
      --duration 10
  ) >"$log_path" 2>&1; then
    echo "The 10-second smoke test failed. Last log lines:" >&2
    tail -n 40 "$log_path" >&2 || true
    return 2
  fi
  if ! require_clean_log "$log_path"; then
    tail -n 40 "$log_path" >&2 || true
    return 2
  fi
  "$python_bin" - "$trajectory_path" "$performance_path" <<'PY'
import csv
import json
import sys
from pathlib import Path

trajectory, performance_path = map(Path, sys.argv[1:])
if not trajectory.is_file() or not performance_path.is_file():
    raise SystemExit("smoke test did not create trajectory and performance files")
with trajectory.open("r", newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
performance = json.loads(performance_path.read_text(encoding="utf-8"))
processed = int(performance.get("processed_frames", 0))
duration = float(performance.get("duration_seconds", 0.0))
fusion_fps = float(performance.get("fusion_fps_median", 0.0))
average_fps = processed / duration if duration > 0.0 else 0.0
if processed <= 0 or len(rows) != processed:
    raise SystemExit("smoke test frame count is invalid")
if duration < 9.0:
    raise SystemExit(f"smoke test is too short: {duration:.3f} seconds")
if fusion_fps < 30.0 or average_fps < 30.0:
    raise SystemExit(
        "smoke test FPS is too low: "
        f"median={fusion_fps:.3f}, average={average_fps:.3f}"
    )
print(
    f"Smoke output verified: {processed} frames in {duration:.3f} seconds, "
    f"median FPS {fusion_fps:.3f}, average FPS {average_fps:.3f}"
)
PY
  echo "10-second smoke test passed: $log_path"
}
