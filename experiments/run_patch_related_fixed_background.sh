#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-icassp-mbrs-patch-fixed}"
LOG_PATH="${LOG_PATH:-/mnt/wmcontent/GLX/icassp/MBRS/logs/patch-related-fixed.log}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
	echo "already running: tmux_session=$SESSION_NAME" >&2
	exit 1
fi

mkdir -p "$(dirname -- "$LOG_PATH")"
tmux new-session -d -s "$SESSION_NAME" \
	"cd $(printf '%q' "$PROJECT_ROOT") && exec $(printf '%q' "$SCRIPT_DIR/run_patch_related_fixed.sh") > $(printf '%q' "$LOG_PATH") 2>&1"
echo "started tmux_session=$SESSION_NAME"
echo "log=$LOG_PATH"
echo "attach: tmux attach -t $SESSION_NAME"
