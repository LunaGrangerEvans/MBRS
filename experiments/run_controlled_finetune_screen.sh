#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge/bin/python}"
GPU="${GPU:-0}"

cd "$PROJECT_ROOT"

SEEDS="${SEEDS:-17}"
for seed in $SEEDS; do
	global_checkpoint="$RESULTS_ROOT/optimization_global_seed${seed}_128_m64_crop/checkpoint_0100.pth"
	if [[ ! -f "$global_checkpoint" ]]; then
		echo "missing Global checkpoint: $global_checkpoint" >&2
		exit 1
	fi
	for variant in global_control patch16_weight25; do
		config="$SCRIPT_DIR/config_finetune_${variant}_128_m64.json"
		output="$RESULTS_ROOT/controlled_finetune_${variant}_seed${seed}_128_m64_crop"
		echo "starting variant=$variant seed=$seed gpu=$GPU output=$output"
		CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
			experiments/train_local_patch.py \
			--config "$config" \
			--output-dir "$output" \
			--device cuda \
			--seed "$seed" \
			--init-checkpoint "$global_checkpoint"
	done
done

echo "controlled_finetune_screen_complete"
