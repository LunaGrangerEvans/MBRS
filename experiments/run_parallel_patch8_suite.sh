#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs}"
REPORTS_ROOT="${REPORTS_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS/reports}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge/bin/python}"
OMP_THREADS="${OMP_NUM_THREADS:-8}"
MKL_THREADS="${MKL_NUM_THREADS:-$OMP_THREADS}"

cd "$PROJECT_ROOT"

run_family() {
	local gpu="$1"
	local label="$2"
	local config="$3"
	local global_weight="$4"
	local local_weight="$5"
	for seed in 17 29 41; do
		local output_dir="$RESULTS_ROOT/parallel_patch8_${label}_seed${seed}_128_m64_crop"
		local latest_checkpoint
		local resume_args=()
		mkdir -p "$output_dir"
		latest_checkpoint="$(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint_*.pth' -printf '%f\n' | sort -V | tail -n 1)"
		if [[ "$latest_checkpoint" == "checkpoint_0100.pth" ]]; then
			echo "skip_complete label=$label seed=$seed"
			continue
		fi
		if [[ -n "$latest_checkpoint" ]]; then
			resume_args=(--resume "$output_dir/$latest_checkpoint")
		fi
		echo "starting gpu=$gpu label=$label seed=$seed output=$output_dir resume=${latest_checkpoint:-none}"
		CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
			PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$PYTHON_BIN" \
				experiments/train_local_patch.py \
				--config "$config" \
				--output-dir "$output_dir" \
				--device cuda \
				--seed "$seed" \
				--global-loss-weight "$global_weight" \
				--local-loss-weight "$local_weight" \
				"${resume_args[@]}"
	done
}

run_family 0 worst_weight75 "$SCRIPT_DIR/config_candidate_patch8_top25_128_m64.json" 0.25 0.75 &
worker0=$!
run_family 0 worst_weight50 "$SCRIPT_DIR/config_candidate_patch8_top25_128_m64.json" 0.5 0.5 &
worker1=$!
run_family 1 worst_weight25 "$SCRIPT_DIR/config_candidate_patch8_top25_128_m64.json" 0.75 0.25 &
worker2=$!
run_family 1 topk10_weight50 "$SCRIPT_DIR/config_candidate_patch8_top10_128_m64.json" 0.5 0.5 &
worker3=$!
run_family 1 topk50_weight50 "$SCRIPT_DIR/config_candidate_patch8_top50_128_m64.json" 0.5 0.5 &
worker4=$!

wait "$worker0"
wait "$worker1"
wait "$worker2"
wait "$worker3"
wait "$worker4"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
	PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$PYTHON_BIN" \
		experiments/evaluate_all_checkpoints.py \
		--runs-root "$RESULTS_ROOT" \
		--output "$REPORTS_ROOT/uniform_eval_all.jsonl" \
		--manifest "$REPORTS_ROOT/uniform_eval_manifest.pt" \
		--device cuda \
		--repeats 5

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" experiments/summarize_uniform_eval.py
echo "patch8_suite_complete"
