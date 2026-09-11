#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONTENT_ROOT="${CONTENT_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge/bin/python}"
GPU="${GPU:-0}"
SOURCE_CHECKPOINT="$CONTENT_ROOT/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth"
SEED17_METRICS="$CONTENT_ROOT/reports/controlled_seed17_metrics.json"
LOG_DIR="$CONTENT_ROOT/logs/controlled-seed17"

cd "$PROJECT_ROOT"

go="$($PYTHON_BIN - "$SEED17_METRICS" <<'PY'
import json, sys
print("true" if json.load(open(sys.argv[1]))["soft_temperature_screening_go"] else "false")
PY
)"
if [[ "$go" != "true" ]]; then
	echo "Soft T=0.5 did not pass seed17 gate; temperature screen not started"
	exit 0
fi

run_temperature() {
	local suffix="$1"
	local config="$2"
	local output="$CONTENT_ROOT/experiments/runs/controlled_seed17_soft_p16_${suffix}_l50"
	local log="$LOG_DIR/soft_p16_${suffix}_l50.log"
	if [[ -e "$output" ]]; then
		echo "refusing to reuse controlled output directory: $output" >&2
		exit 2
	fi
	echo "starting controlled soft temperature=$suffix seed=17 gpu=$GPU"
	CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
		experiments/train_local_patch.py \
		--config "$config" \
		--output-dir "$output" \
		--device cuda \
		--seed 17 \
		--init-checkpoint "$SOURCE_CHECKPOINT" \
		2>&1 | tee "$log"
}

run_temperature \
	t025 \
	"$SCRIPT_DIR/config_finetune_soft_tail_patch16_weight50_t025_128_m64.json"
run_temperature \
	t10 \
	"$SCRIPT_DIR/config_finetune_soft_tail_patch16_weight50_t10_128_m64.json"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
	experiments/evaluate_soft_temperature_seed17.py --device cuda

echo "controlled_soft_temperature_seed17_complete"
