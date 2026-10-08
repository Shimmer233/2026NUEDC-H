#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

APP_DIR="${1:-$HOME/orangepi5b_vision_only}"
CONFIRM="${2:-}"
REQUESTED_BACKUP="${3:-}"
CAMERA="${4:-${STEEL_BALL_CAMERA:-/dev/video0}}"
if [[ "$CONFIRM" != "--confirm" ]]; then
  echo "Usage: $0 [APP_DIR] --confirm [BACKUP_DIR] [CAMERA]" >&2
  exit 2
fi
if [[ ! -d "$APP_DIR" ]]; then
  echo "Application directory is missing: $APP_DIR" >&2
  exit 2
fi
APP_DIR="$(readlink -f "$APP_DIR")"
PYTHON_BIN="$APP_DIR/.venv/bin/python"
BACKUP_ROOT="$APP_DIR/backups"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Application Python is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ -z "$REQUESTED_BACKUP" ]]; then
  if [[ ! -f "$BACKUP_ROOT/LAST_V4_BACKUP" ]]; then
    echo "No LAST_V4_BACKUP pointer exists." >&2
    exit 2
  fi
  REQUESTED_BACKUP="$(<"$BACKUP_ROOT/LAST_V4_BACKUP")"
fi
if [[ ! -d "$REQUESTED_BACKUP" ]]; then
  echo "Backup directory is missing: $REQUESTED_BACKUP" >&2
  exit 2
fi
BACKUP_DIR="$(readlink -f "$REQUESTED_BACKUP")"
BACKUP_ROOT="$(readlink -f "$BACKUP_ROOT")"
case "$BACKUP_DIR" in
  "$BACKUP_ROOT"/*) ;;
  *)
    echo "Refusing to restore from outside the application backup directory: $BACKUP_DIR" >&2
    exit 2
    ;;
esac
for required in "$BACKUP_DIR/models" "$BACKUP_DIR/config" "$BACKUP_DIR/manifest.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Incomplete rollback backup: $required is missing" >&2
    exit 2
  fi
done

ensure_camera_idle "$CAMERA"

"$PYTHON_BIN" "$SCRIPT_DIR/verify_backup.py" "$BACKUP_DIR"

POST_BACKUP="$BACKUP_ROOT/post_v4_before_rollback_$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -e "$POST_BACKUP" ]]; then
  echo "Post-v4 backup directory already exists: $POST_BACKUP" >&2
  exit 2
fi
mkdir -p "$POST_BACKUP"
cp -a "$APP_DIR/models" "$APP_DIR/config" "$POST_BACKUP/"
"$PYTHON_BIN" "$SCRIPT_DIR/make_backup_manifest.py" \
  "$POST_BACKUP" \
  --purpose "complete models/config snapshot before manual rollback"
"$PYTHON_BIN" "$SCRIPT_DIR/verify_backup.py" "$POST_BACKUP"

rollback_started=0
restore_post_v4() {
  cp -a "$POST_BACKUP/models/." "$APP_DIR/models/"
  cp -a "$POST_BACKUP/config/." "$APP_DIR/config/"
  "$PYTHON_BIN" "$SCRIPT_DIR/verify_backup.py" \
    "$POST_BACKUP" \
    --restored-root "$APP_DIR"
  (
    cd "$APP_DIR"
    sha256sum -c models/SHA256SUMS.txt
  )
}
on_failure() {
  local status=$?
  trap - ERR INT TERM
  if [[ "$rollback_started" -eq 1 ]]; then
    echo "Rollback failed; restoring the complete pre-rollback models/config snapshot." >&2
    restore_post_v4 || echo "Automatic recovery failed; preserved files are at: $POST_BACKUP" >&2
  fi
  exit "$status"
}
trap on_failure ERR INT TERM

rollback_started=1
cp -a "$BACKUP_DIR/models/." "$APP_DIR/models/"
cp -a "$BACKUP_DIR/config/." "$APP_DIR/config/"
sync

"$PYTHON_BIN" "$SCRIPT_DIR/verify_backup.py" \
  "$BACKUP_DIR" \
  --restored-root "$APP_DIR"
(
  cd "$APP_DIR"
  sha256sum -c models/SHA256SUMS.txt
)

SMOKE_DIR="$APP_DIR/runs/manual_v4/rollback_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
run_smoke_test "$APP_DIR" "$CAMERA" "$SMOKE_DIR/run.log"
rollback_started=0
trap - ERR INT TERM

echo "Rollback and smoke test completed from: $BACKUP_DIR"
echo "Pre-rollback v4 files were preserved at: $POST_BACKUP"
