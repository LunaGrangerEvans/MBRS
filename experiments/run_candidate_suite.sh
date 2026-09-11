#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs}"
REPORTS_ROOT="${REPORTS_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS/reports}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda}"
OMP_THREADS="${OMP_NUM_THREADS:-8}"
MKL_THREADS="${MKL_NUM_THREADS:-$OMP_THREADS}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Python environment not found: $PYTHON_BIN" >&2
	exit 1
fi
if [[ "$DEVICE" == "cuda" ]]; then
	if ! CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" -c \
		'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
		echo "CUDA is unavailable" >&2
		exit 2
	fi
elif [[ "$DEVICE" != "cpu" ]]; then
	echo "DEVICE must be either cuda or cpu" >&2
	exit 2
fi

cd "$PROJECT_ROOT"

run_one() {
	local label="$1"
	local seed="$2"
	local config="$3"
	local global_weight="$4"
	local local_weight="$5"
	local output_dir="$RESULTS_ROOT/candidate_${label}_seed${seed}_128_m64_crop"
	local resume_args=()
	local latest_checkpoint
	mkdir -p "$output_dir"
	latest_checkpoint="$(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint_*.pth' -printf '%f\n' | sort -V | tail -n 1)"
	if [[ "$latest_checkpoint" == "checkpoint_0100.pth" ]]; then
		echo "skip_complete label=$label seed=$seed output=$output_dir"
		return
	fi
	if [[ -n "$latest_checkpoint" ]]; then
		resume_args=(--resume "$output_dir/$latest_checkpoint")
	fi
	echo "starting label=$label seed=$seed output=$output_dir resume=${latest_checkpoint:-none}"
	CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
		PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$PYTHON_BIN" \
			experiments/train_local_patch.py \
			--config "$config" \
			--output-dir "$output_dir" \
			--device "$DEVICE" \
			--seed "$seed" \
			--global-loss-weight "$global_weight" \
			--local-loss-weight "$local_weight" \
			"${resume_args[@]}"
}

patch16_config="$SCRIPT_DIR/config_candidate_patch16_128_m64.json"
patch32_config="$SCRIPT_DIR/config_worst_128_m64.json"
topk50_config="$SCRIPT_DIR/config_ablation_topk50_128_m64.json"

for seed in 17 29 41; do
	run_one worst_patch16_weight75 "$seed" "$patch16_config" 0.25 0.75
done
for seed in 29 41; do
	run_one worst_patch16_weight50 "$seed" "$patch16_config" 0.5 0.5
	run_one worst_patch32_weight25 "$seed" "$patch32_config" 0.75 0.25
	run_one worst_patch32_topk50 "$seed" "$topk50_config" 0.5 0.5
done

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
	PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$PYTHON_BIN" \
		experiments/evaluate_all_checkpoints.py \
		--runs-root "$RESULTS_ROOT" \
		--output "$REPORTS_ROOT/uniform_eval_all.jsonl" \
		--manifest "$REPORTS_ROOT/uniform_eval_manifest.pt" \
		--device "$DEVICE" \
		--repeats 5

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
	PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$PYTHON_BIN" \
		experiments/summarize_uniform_eval.py

echo "candidate_suite_complete"
