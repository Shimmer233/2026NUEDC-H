#!/usr/bin/env bash
set -euo pipefail
cd /home/orangepi/orangepi5b_vision_only
exec ./run_touch_gui.sh /dev/video0 --uart-device /dev/ttyS1 --uart-baud 115200 --fullscreen
