#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs}"
REPORTS_ROOT="${REPORTS_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS/reports}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge/bin/python}"
OMP_THREADS="${OMP_NUM_THREADS:-8}"
MKL_THREADS="${MKL_NUM_THREADS:-$OMP_THREADS}"
CONFIG="$SCRIPT_DIR/config_ablation_patch16_128_m64.json"

cd "$PROJECT_ROOT"

# This legacy exploratory suite is intentionally capped after seed29.
for seed in 17 29; do
	output_dir="$RESULTS_ROOT/candidate_worst_patch16_weight25_seed${seed}_128_m64_crop"
	mkdir -p "$output_dir"
	latest_checkpoint="$(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint_*.pth' -printf '%f\n' | sort -V | tail -n 1)"
	if [[ "$latest_checkpoint" == "checkpoint_0100.pth" ]]; then
		echo "skip_complete seed=$seed output=$output_dir"
		continue
	fi
	resume_args=()
	if [[ -n "$latest_checkpoint" ]]; then
		resume_args=(--resume "$output_dir/$latest_checkpoint")
	fi
	echo "starting patch=16 topk=0.25 global_weight=0.75 local_weight=0.25 seed=$seed resume=${latest_checkpoint:-none}"
	CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
		PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$PYTHON_BIN" \
			experiments/train_local_patch.py \
			--config "$CONFIG" \
			--output-dir "$output_dir" \
			--device cuda \
			--seed "$seed" \
			--global-loss-weight 0.75 \
			--local-loss-weight 0.25 \
			"${resume_args[@]}"
done

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
	PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$PYTHON_BIN" \
		experiments/evaluate_all_checkpoints.py \
		--runs-root "$RESULTS_ROOT" \
		--output "$REPORTS_ROOT/uniform_eval_all.jsonl" \
		--manifest "$REPORTS_ROOT/uniform_eval_manifest.pt" \
		--device cuda \
		--repeats 5

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" experiments/summarize_uniform_eval.py
echo "patch16_weight25_suite_complete"
