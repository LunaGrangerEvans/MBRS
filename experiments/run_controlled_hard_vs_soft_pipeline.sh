#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONTENT_ROOT="${CONTENT_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge/bin/python}"
GPU="${GPU:-0}"
PIPELINE_LOG="$CONTENT_ROOT/logs/controlled-hard-vs-soft-pipeline.log"
LEGACY_PATTERN='^bash /root/workspace/GLX/icassp/MBRS/experiments/run_patch16_weight25_suite.sh$'

cd "$PROJECT_ROOT"
mkdir -p "$(dirname -- "$PIPELINE_LOG")"

exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "controlled pipeline queued gpu=$GPU"
while pgrep -f "$LEGACY_PATTERN" >/dev/null; do
	echo "waiting for legacy P16/T25/L25 suite to release MBRS GPUs"
	sleep 60
done

echo "legacy suite finished; starting controlled seed17 gate"
GPU="$GPU" CONTENT_ROOT="$CONTENT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
	"$SCRIPT_DIR/run_controlled_preflight.sh"

GPU="$GPU" CONTENT_ROOT="$CONTENT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
	"$SCRIPT_DIR/run_controlled_seed17.sh"

seed17_go="$($PYTHON_BIN - "$CONTENT_ROOT/reports/controlled_seed17_metrics.json" <<'PY'
import json, sys
print("true" if json.load(open(sys.argv[1]))["soft_temperature_screening_go"] else "false")
PY
)"
if [[ "$seed17_go" != "true" ]]; then
	echo "Soft T=0.5 no-go; pipeline stopped before temperature and additional seeds"
	exit 0
fi

GPU="$GPU" CONTENT_ROOT="$CONTENT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
	"$SCRIPT_DIR/run_soft_temperature_seed17.sh"

selected="$($PYTHON_BIN - "$CONTENT_ROOT/reports/controlled_soft_temperature_seed17.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))["selected_temperature"]
print("none" if value is None else value)
PY
)"
if [[ "$selected" == "none" ]]; then
	echo "no Soft temperature passed; single-seed pipeline complete"
	exit 0
fi

echo "single-seed policy active; seed29/41 controlled branches are not launched"
echo "controlled hard-vs-soft seed17 pipeline complete selected_temperature=$selected"
