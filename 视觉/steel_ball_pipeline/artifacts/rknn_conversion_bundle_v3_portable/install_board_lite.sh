#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "RKNNLite2 installation must run on the Orange Pi (aarch64)." >&2
  exit 2
fi
python3.10 -m pip install board_wheel/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
