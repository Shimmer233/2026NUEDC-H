#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${RKNN_PYTHON:-python3.10}"
"$PYTHON_BIN" verify_bundle.py
"$PYTHON_BIN" - <<'PY'
from importlib import metadata

version = metadata.version("rknn-toolkit2")
if version != "2.3.2":
    raise SystemExit(f"expected rknn-toolkit2 2.3.2, found {version}")
from rknn.api import RKNN  # noqa: F401
print(f"using rknn-toolkit2 {version}")
PY

"$PYTHON_BIN" convert_rknn.py --onnx model/best.onnx --mode fp16 --output output/steel_ball_yolov5n_fp16.rknn
"$PYTHON_BIN" convert_rknn.py --onnx model/best.onnx --mode int8 --dataset-list dataset.txt --output output/steel_ball_yolov5n_int8.rknn
sha256sum output/*.rknn | tee output/SHA256SUMS.txt
echo "Conversion complete with the existing RKNN-Toolkit2 environment."
