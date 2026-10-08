#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This script is for the Orange Pi / aarch64; current architecture: $(uname -m)" >&2
  exit 2
fi
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "python3.10 is required (Ubuntu 22.04 default)." >&2
  exit 2
fi

python3.10 verify_bundle.py
python3.10 -m venv .venv-convert
source .venv-convert/bin/activate

PIP_MIRROR="${RKNN_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TIMEOUT="${RKNN_PIP_TIMEOUT:-120}"
PIP_RETRIES="${RKNN_PIP_RETRIES:-10}"
PIP_ARGS=(
  --index-url "$PIP_MIRROR"
  --timeout "$PIP_TIMEOUT"
  --retries "$PIP_RETRIES"
)
echo "Using Python package index: $PIP_MIRROR"
python -m pip install "${PIP_ARGS[@]}" --upgrade pip
python -m pip install "${PIP_ARGS[@]}" -r toolkit_arm64/arm64_requirements_cp310.txt
python -m pip install "${PIP_ARGS[@]}" toolkit_arm64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

python convert_rknn.py --onnx model/best.onnx --mode fp16 --output output/steel_ball_yolov5n_fp16.rknn
python convert_rknn.py --onnx model/best.onnx --mode int8 --dataset-list dataset.txt --output output/steel_ball_yolov5n_int8.rknn
sha256sum output/*.rknn | tee output/SHA256SUMS.txt
echo "Conversion complete on Orange Pi."
