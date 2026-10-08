#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sudo apt-get update
sudo apt-get install -y python3-venv python3-opencv python3-yaml python3-psutil python3-ruamel.yaml
if ! command -v v4l2-ctl >/dev/null 2>&1; then
  sudo apt-get install -y --no-upgrade v4l-utils
fi
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps board_wheel/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
.venv/bin/python - <<'PY'
import cv2, numpy, yaml
from rknnlite.api import RKNNLite
print("OpenCV", cv2.__version__)
print("NumPy", numpy.__version__)
print("RKNNLite2 import OK", RKNNLite)
PY
