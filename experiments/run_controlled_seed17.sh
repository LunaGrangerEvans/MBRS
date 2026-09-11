#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONTENT_ROOT="${CONTENT_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge/bin/python}"
GPU="${GPU:-0}"
SEED=17
SOURCE_CHECKPOINT="$CONTENT_ROOT/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth"
LOG_DIR="$CONTENT_ROOT/logs/controlled-seed17"

cd "$PROJECT_ROOT"

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
	echo "missing source checkpoint: $SOURCE_CHECKPOINT" >&2
	exit 1
fi

if pgrep -af 'experiments/train_local_patch.py' | grep -v 'controlled_seed17_' >/dev/null; then
	echo "another MBRS training process is active; controlled seed17 run not started" >&2
	exit 3
fi

mkdir -p "$LOG_DIR"

run_branch() {
	local label="$1"
	local config="$2"
	local output="$CONTENT_ROOT/experiments/runs/controlled_seed17_${label}"
	local log="$LOG_DIR/${label}.log"
	if [[ -e "$output" ]]; then
		echo "refusing to reuse controlled output directory: $output" >&2
		exit 2
	fi
	echo "starting controlled branch=$label seed=$SEED gpu=$GPU output=$output"
	CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
		experiments/train_local_patch.py \
		--config "$config" \
		--output-dir "$output" \
		--device cuda \
		--seed "$SEED" \
		--init-checkpoint "$SOURCE_CHECKPOINT" \
		2>&1 | tee "$log"
}

run_branch \
	global_continuation \
	"$SCRIPT_DIR/config_finetune_global_control_128_m64.json"
run_branch \
	hard_p16_t25_l50 \
	"$SCRIPT_DIR/config_finetune_hard_patch16_weight50_128_m64.json"
run_branch \
	soft_p16_t05_l50 \
	"$SCRIPT_DIR/config_finetune_soft_tail_patch16_weight50_t05_128_m64.json"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
	experiments/evaluate_controlled_seed17.py \
	--device cuda \
	--output-json "$CONTENT_ROOT/reports/controlled_seed17_metrics.json"

echo "controlled_seed17_complete"
