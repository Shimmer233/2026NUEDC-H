#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CAMERA="/dev/video0"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CAMERA="$1"
  shift
fi
UART_DEVICE="${STEEL_BALL_UART_DEVICE:-/dev/ttyS1}"
UART_BAUD="${STEEL_BALL_UART_BAUD:-115200}"
ENABLE_UART=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --uart-device)
      UART_DEVICE="${2:?missing value for --uart-device}"
      ENABLE_UART=1
      shift 2
      ;;
    --uart-baud)
      UART_BAUD="${2:?missing value for --uart-baud}"
      shift 2
      ;;
    --no-uart)
      ENABLE_UART=0
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done
PROJECT_DIR="$(pwd)"
DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
DESKTOP_FILE="$DESKTOP_DIR/SteelBallVision.desktop"
LAUNCH_SCRIPT="$PROJECT_DIR/.touch_launcher_start.sh"

if [[ ! -f run_touch_gui.sh ]]; then
  echo "run_touch_gui.sh not found in $PROJECT_DIR" >&2
  exit 1
fi

mkdir -p "$DESKTOP_DIR"
chmod +x run_touch_gui.sh

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "$PROJECT_DIR"
  printf 'exec ./run_touch_gui.sh %q' "$CAMERA"
  if [[ "$ENABLE_UART" -eq 1 ]]; then
    printf ' --uart-device %q --uart-baud %q' "$UART_DEVICE" "$UART_BAUD"
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    printf ' %q' "${EXTRA_ARGS[@]}"
  fi
  printf '\n'
} > "$LAUNCH_SCRIPT"
chmod +x "$LAUNCH_SCRIPT"

escape_desktop_arg() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

LAUNCH_ARG="$(escape_desktop_arg "$LAUNCH_SCRIPT")"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Steel Ball Vision
Comment=Start steel ball touchscreen API
Exec=sh -c "exec \"\$1\"" sh $LAUNCH_ARG
Path=$PROJECT_DIR
Terminal=false
StartupNotify=true
Categories=Utility;
EOF

chmod +x "$DESKTOP_FILE"
gio set "$DESKTOP_FILE" metadata::trusted true >/dev/null 2>&1 || true

echo "Installed launcher: $DESKTOP_FILE"
if [[ "$ENABLE_UART" -eq 1 ]]; then
  echo "UART output: $UART_DEVICE @ $UART_BAUD"
else
  echo "UART output: disabled"
fi
echo "Double-click or double-tap 'Steel Ball Vision' on the desktop to start."
