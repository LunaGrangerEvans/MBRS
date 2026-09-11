#!/usr/bin/env bash
set -euo pipefail

LEGACY_PARENT_PID="${1:?legacy parent PID required}"
SEED29_PID="${2:?seed29 PID required}"
PROJECT_ROOT="/root/workspace/GLX/icassp/MBRS"
CONTENT_ROOT="/mnt/wmcontent/GLX/icassp/MBRS"
PYTHON_BIN="/root/miniforge/bin/python"
LOG="$CONTENT_ROOT/logs/seed29-finalize-then-controlled.log"

cd "$PROJECT_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "waiting for legacy Patch16/Weight25 seed29 pid=$SEED29_PID"
while kill -0 "$SEED29_PID" 2>/dev/null; do
	sleep 60
done

echo "seed29 finished; waiting for legacy launcher to skip seed41 and finish evaluation"
while kill -0 "$LEGACY_PARENT_PID" 2>/dev/null; do
	sleep 30
done

echo "starting single-seed controlled pipeline"
exec env GPU=0 CONTENT_ROOT="$CONTENT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
	./experiments/run_controlled_hard_vs_soft_pipeline.sh
