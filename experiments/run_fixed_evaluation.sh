#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge/bin/python}"

cd "$PROJECT_ROOT"
for variant in global patch_mean worst ablation_patch16 ablation_patch64 ablation_topk10 ablation_topk50; do
	config="$SCRIPT_DIR/config_${variant}_128_m64.json"
	output_dir="$RESULTS_ROOT/fixed_${variant}_128_m64_crop"
	checkpoint="$output_dir/checkpoint_0100.pth"
	metrics="$output_dir/crop_metrics.json"
	if [[ ! -f "$checkpoint" ]]; then
		echo "missing checkpoint: $checkpoint" >&2
		exit 1
	fi
	if [[ -f "$metrics" ]]; then
		echo "skip existing evaluation: $variant"
		continue
	fi
	echo "starting evaluation: $variant"
	CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
		experiments/evaluate_crop.py \
		--config "$config" \
		--checkpoint "$checkpoint" \
		--output "$metrics" \
		--device cuda --repeats 5 \
		--save-examples "$output_dir/examples.pt"
done
