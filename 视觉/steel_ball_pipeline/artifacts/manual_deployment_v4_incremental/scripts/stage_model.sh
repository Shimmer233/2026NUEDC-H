#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${1:-$HOME/orangepi5b_vision_only}"
BUNDLE_DIR="${2:-$HOME/rknn_conversion_bundle_v4_incremental}"

if [[ ! -d "$APP_DIR" || ! -d "$BUNDLE_DIR" ]]; then
  echo "Usage: $0 [APP_DIR] [CONVERSION_BUNDLE_DIR]" >&2
  exit 2
fi
APP_DIR="$(readlink -f "$APP_DIR")"
BUNDLE_DIR="$(readlink -f "$BUNDLE_DIR")"
PYTHON_BIN="$APP_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Application Python is missing: $PYTHON_BIN" >&2
  exit 2
fi
for path in \
  "$APP_DIR/board/steel_ball_vision.py" \
  "$APP_DIR/config/vision.yaml" \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_int8.rknn" \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_int8.json" \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_fp16.rknn" \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_fp16.json" \
  "$BUNDLE_DIR/output/SHA256SUMS.txt" \
  "$BUNDLE_DIR/bundle_manifest.json" \
  "$BUNDLE_DIR/verify_bundle.py"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file is missing: $path" >&2
    exit 2
  fi
done

"$PYTHON_BIN" "$BUNDLE_DIR/verify_bundle.py"
"$PYTHON_BIN" - \
  "$BUNDLE_DIR/bundle_manifest.json" \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_int8.json" \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_int8.rknn" \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_fp16.json" \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_fp16.rknn" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

bundle = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if bundle.get("target_platform") != "rk3588":
    raise SystemExit("bundle target_platform is not rk3588")
expected_onnx = bundle["model"]["sha256"]
expected_dataset = bundle["calibration"]["dataset_txt_sha256"]
for mode, manifest_name, model_name in (
    ("int8", sys.argv[2], sys.argv[3]),
    ("fp16", sys.argv[4], sys.argv[5]),
):
    manifest = json.loads(Path(manifest_name).read_text(encoding="utf-8"))
    model = Path(model_name)
    actual = hashlib.sha256(model.read_bytes()).hexdigest()
    if manifest.get("toolkit_version") != "2.3.2":
        raise SystemExit(f"{mode}: expected Toolkit2 2.3.2")
    if manifest.get("target_platform") != "rk3588" or manifest.get("mode") != mode:
        raise SystemExit(f"{mode}: conversion manifest contract mismatch")
    if manifest.get("onnx_sha256") != expected_onnx:
        raise SystemExit(f"{mode}: converted from an unexpected ONNX model")
    if manifest.get("mean_values") != [[0, 0, 0]]:
        raise SystemExit(f"{mode}: unexpected input mean normalization")
    if manifest.get("std_values") != [[255, 255, 255]]:
        raise SystemExit(f"{mode}: unexpected input scale normalization")
    if mode == "int8" and manifest.get("dataset_list_sha256") != expected_dataset:
        raise SystemExit("int8: converted with an unexpected quantization dataset")
    if mode == "fp16" and manifest.get("dataset_list_sha256") is not None:
        raise SystemExit("fp16: unexpected quantization dataset")
    if manifest.get("output_sha256") != actual:
        raise SystemExit(f"{mode}: RKNN hash does not match its manifest")
print("RKNN conversion manifests verified")
PY

(
  cd "$BUNDLE_DIR"
  sha256sum -c output/SHA256SUMS.txt
)

atomic_install() {
  local source="$1"
  local destination="$2"
  local temporary="${destination}.tmp.$$"
  install -m 0644 "$source" "$temporary"
  mv -f "$temporary" "$destination"
}

mkdir -p "$APP_DIR/models" "$APP_DIR/config"
atomic_install \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_int8.rknn" \
  "$APP_DIR/models/steel_ball_yolov5n_v4_incremental_int8.rknn"
atomic_install \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_int8.json" \
  "$APP_DIR/models/steel_ball_yolov5n_v4_incremental_int8.json"
atomic_install \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_fp16.rknn" \
  "$APP_DIR/models/steel_ball_yolov5n_v4_incremental_fp16.rknn"
atomic_install \
  "$BUNDLE_DIR/output/steel_ball_yolov5n_fp16.json" \
  "$APP_DIR/models/steel_ball_yolov5n_v4_incremental_fp16.json"

"$PYTHON_BIN" - "$APP_DIR" "$BUNDLE_DIR" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

app = Path(sys.argv[1])
bundle = Path(sys.argv[2])
source_config = app / "config" / "vision.yaml"
staged_config = app / "config" / "vision_v4_incremental_staged.yaml"
config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
protected = {
    key: config.get(key)
    for key in ("calibration", "geometry", "camera", "detection", "coordinate")
}
config["model"] = "models/steel_ball_yolov5n_v4_incremental_int8.rknn"
config.setdefault("roi", {}).update(
    {"x": 10, "y": 150, "width": 600, "height": 105, "output_width": 470, "output_height": 110}
)
config.setdefault("output", {})["csv"] = "runs/manual_v4/staged/trajectory.csv"
if any(config.get(key) != value for key, value in protected.items()):
    raise SystemExit("staging attempted to modify a protected calibration or camera setting")
temporary = staged_config.with_name(f".{staged_config.name}.tmp")
temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
os.replace(temporary, staged_config)

files = [
    "models/steel_ball_yolov5n_v4_incremental_int8.rknn",
    "models/steel_ball_yolov5n_v4_incremental_int8.json",
    "models/steel_ball_yolov5n_v4_incremental_fp16.rknn",
    "models/steel_ball_yolov5n_v4_incremental_fp16.json",
    "config/vision_v4_incremental_staged.yaml",
]
runtime_dependencies = {}
for key in ("calibration", "geometry"):
    value = config.get(key)
    if not value:
        raise SystemExit(f"active config is missing the {key} path")
    dependency = Path(value)
    if not dependency.is_absolute():
        dependency = app / dependency
    dependency = dependency.resolve()
    try:
        relative = dependency.relative_to(app).as_posix()
    except ValueError as error:
        raise SystemExit(f"{key} file is outside the application directory") from error
    if not dependency.is_file():
        raise SystemExit(f"{key} file is missing: {dependency}")
    runtime_dependencies[relative] = {
        "size": dependency.stat().st_size,
        "sha256": hashlib.sha256(dependency.read_bytes()).hexdigest(),
    }
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "bundle": str(bundle),
    "bundle_manifest_sha256": hashlib.sha256(
        (bundle / "bundle_manifest.json").read_bytes()
    ).hexdigest(),
    "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
    "runtime_dependencies": runtime_dependencies,
    "active_model_changed": False,
    "files": {
        relative: {
            "size": (app / relative).stat().st_size,
            "sha256": hashlib.sha256((app / relative).read_bytes()).hexdigest(),
        }
        for relative in files
    },
}
manifest = app / "models" / "steel_ball_v4_incremental.stage.json"
temporary = manifest.with_name(f".{manifest.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, manifest)
print(f"Staged config: {staged_config}")
print(f"Stage manifest: {manifest}")
PY

echo "Staging complete. The active model has not been replaced."
echo "Next: $SCRIPT_DIR/run_staged_test.sh '$APP_DIR' normal 60 /dev/video0"
