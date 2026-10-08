#!/usr/bin/env bash
set -euo pipefail
v4l2-ctl --list-devices
echo
echo "Stable USB camera paths:"
find /dev/v4l/by-id -maxdepth 1 -type l -print 2>/dev/null || true
echo
ls -l /dev/ttyS0 /dev/ttyS1 2>/dev/null || true
echo
for device in /dev/video*; do
  [[ -e "$device" ]] || continue
  echo "===== $device ====="
  v4l2-ctl --device="$device" --list-formats-ext 2>/dev/null || true
done
