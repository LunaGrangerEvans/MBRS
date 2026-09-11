#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-icassp-mbrs-matrix}"
LOG_PATH="${LOG_PATH:-/mnt/wmcontent/GLX/icassp/MBRS/logs/matrix.stdout.log}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
	echo "already running: tmux_session=$SESSION_NAME" >&2
	exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
	"cd $(printf '%q' "$PROJECT_ROOT") && exec $(printf '%q' "$SCRIPT_DIR/run_matrix.sh") > $(printf '%q' "$LOG_PATH") 2>&1"
echo "started tmux_session=$SESSION_NAME"
echo "log=$LOG_PATH"
echo "attach: tmux attach -t $SESSION_NAME"
