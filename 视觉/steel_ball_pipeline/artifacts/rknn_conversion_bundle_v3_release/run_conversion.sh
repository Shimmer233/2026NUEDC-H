#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "Conversion requires x86_64 Linux; current architecture: $(uname -m)" >&2
  exit 2
fi
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "python3.10 is required (Ubuntu 22.04 default)." >&2
  exit 2
fi

python3.10 verify_bundle.py
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r toolkit/requirements_cp310-2.3.2.txt
python -m pip install toolkit/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

python convert_rknn.py --onnx model/best.onnx --mode fp16 --output output/steel_ball_yolov5n_fp16.rknn
python convert_rknn.py --onnx model/best.onnx --mode int8 --dataset-list dataset.txt --output output/steel_ball_yolov5n_int8.rknn
sha256sum output/*.rknn | tee output/SHA256SUMS.txt
echo "Conversion complete. Copy output/*.rknn to the Orange Pi deployment directory."
