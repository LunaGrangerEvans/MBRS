#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda}"
OMP_THREADS="${OMP_NUM_THREADS:-8}"
MKL_THREADS="${MKL_NUM_THREADS:-$OMP_THREADS}"
PROJECT_PYTHON="$PROJECT_ROOT/../.conda-envs/mbrs/bin/python"
CUDA_PYTHON_BIN="${CUDA_PYTHON_BIN:-/root/miniforge/bin/python}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
	if [[ "$DEVICE" == "cuda" && -x "$CUDA_PYTHON_BIN" ]]; then
		PYTHON_BIN="$CUDA_PYTHON_BIN"
	else
		PYTHON_BIN="$PROJECT_PYTHON"
	fi
fi
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

mkdir -p "$RESULTS_ROOT"
cd "$PROJECT_ROOT"

run_variant() {
	local variant="$1"
	local seed="$2"
	local output_dir="$3"
	shift 3
	local config="$SCRIPT_DIR/config_${variant}_128_m64.json"
	local resume_args=()
	local latest_checkpoint
	mkdir -p "$output_dir"
	latest_checkpoint="$(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint_*.pth' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
	if [[ -n "$latest_checkpoint" ]]; then
		resume_args=(--resume "$output_dir/$latest_checkpoint")
	fi
	echo "starting variant=$variant seed=$seed output=$output_dir device=$DEVICE threads=$OMP_THREADS resume=${latest_checkpoint:-none} $*"
	CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
		PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
			experiments/train_local_patch.py \
			--config "$config" \
			--output-dir "$output_dir" \
			--device "$DEVICE" \
			--seed "$seed" \
			"$@" \
			"${resume_args[@]}"
}

for seed in 17 29 41; do
	for variant in global patch_mean worst; do
		run_variant "$variant" "$seed" "$RESULTS_ROOT/optimization_${variant}_seed${seed}_128_m64_crop"
	done
done

run_variant worst 17 "$RESULTS_ROOT/optimization_worst_weight25_seed17_128_m64_crop" \
	--global-loss-weight 0.75 --local-loss-weight 0.25
run_variant worst 17 "$RESULTS_ROOT/optimization_worst_weight75_seed17_128_m64_crop" \
	--global-loss-weight 0.25 --local-loss-weight 0.75
