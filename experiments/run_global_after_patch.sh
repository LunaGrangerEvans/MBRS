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
PATCH_SESSION="${PATCH_SESSION:-icassp-mbrs-patch-fixed}"

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

cd "$PROJECT_ROOT"
patch_variants=(patch_mean worst ablation_patch16 ablation_patch64 ablation_topk10 ablation_topk50)
while tmux has-session -t "$PATCH_SESSION" 2>/dev/null; do
	echo "waiting_for_patch_session=$PATCH_SESSION"
	sleep 30
done

for variant in "${patch_variants[@]}"; do
	checkpoint="$RESULTS_ROOT/fixed_${variant}_128_m64_crop/checkpoint_0100.pth"
	if [[ ! -f "$checkpoint" ]]; then
		echo "patch_variant_incomplete=$variant missing=$checkpoint" >&2
		exit 3
	fi
done

config="$SCRIPT_DIR/config_global_128_m64.json"
output_dir="$RESULTS_ROOT/fixed_global_128_m64_crop"
mkdir -p "$output_dir"
resume_args=()
latest_checkpoint="$(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint_*.pth' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
if [[ -n "$latest_checkpoint" ]]; then
	resume_args=(--resume "$output_dir/$latest_checkpoint")
fi
echo "starting variant=global output=$output_dir device=$DEVICE threads=$OMP_THREADS resume=${latest_checkpoint:-none}"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
	PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
		experiments/train_local_patch.py \
		--config "$config" \
		--output-dir "$output_dir" \
		--device "$DEVICE" \
		"${resume_args[@]}"
