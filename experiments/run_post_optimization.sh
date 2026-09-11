#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs}"
REPORTS_ROOT="${REPORTS_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS/reports}"
OPT_SESSION="${OPT_SESSION:-icassp-mbrs-optimization}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
OMP_THREADS="${OMP_NUM_THREADS:-8}"
MKL_THREADS="${MKL_NUM_THREADS:-$OMP_THREADS}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge/bin/python}"
MANIFEST="$REPORTS_ROOT/uniform_eval_manifest.pt"
SUMMARY="$REPORTS_ROOT/uniform_eval_all.jsonl"
NOCROP_DIR="$RESULTS_ROOT/nocrop_global_128_m64"

mkdir -p "$REPORTS_ROOT" "$NOCROP_DIR"
cd "$PROJECT_ROOT"

expected_runs=()
for seed in 17 29 41; do
	for variant in global patch_mean worst; do
		expected_runs+=("$RESULTS_ROOT/optimization_${variant}_seed${seed}_128_m64_crop/checkpoint_0100.pth")
	done
done
expected_runs+=(
	"$RESULTS_ROOT/optimization_worst_weight25_seed17_128_m64_crop/checkpoint_0100.pth"
	"$RESULTS_ROOT/optimization_worst_weight75_seed17_128_m64_crop/checkpoint_0100.pth"
)

while :; do
	missing=0
	for checkpoint in "${expected_runs[@]}"; do
		if [[ ! -f "$checkpoint" ]]; then
			missing=1
			break
		fi
	done
	if [[ "$missing" == 0 ]] && ! tmux has-session -t "$OPT_SESSION" 2>/dev/null; then
		break
	fi
	echo "waiting_for_optimization_completion session=$OPT_SESSION missing_final_checkpoint=$missing"
	sleep 60
done

echo "optimization_session_finished=$OPT_SESSION"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
	PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$PYTHON_BIN" \
		experiments/evaluate_all_checkpoints.py \
		--runs-root "$RESULTS_ROOT" \
		--output "$SUMMARY" \
		--manifest "$MANIFEST" \
		--device "$DEVICE" \
		--repeats 5

checkpoint="$NOCROP_DIR/checkpoint_0100.pth"
if [[ ! -f "$checkpoint" ]]; then
	resume_args=()
	latest_checkpoint="$(find "$NOCROP_DIR" -maxdepth 1 -type f -name 'checkpoint_*.pth' -printf '%f\n' | sort -V | tail -n 1)"
	if [[ -n "$latest_checkpoint" ]]; then
		resume_args=(--resume "$NOCROP_DIR/$latest_checkpoint")
	fi
	echo "starting_nocrop_baseline output=$NOCROP_DIR resume=${latest_checkpoint:-none}"
	CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
		PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$PYTHON_BIN" \
			experiments/train_local_patch.py \
			--config "$SCRIPT_DIR/config_nocrop_global_128_m64.json" \
			--output-dir "$NOCROP_DIR" \
			--device "$DEVICE" \
			"${resume_args[@]}"
else
	echo "skip_existing_nocrop_baseline checkpoint=$checkpoint"
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
	PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$PYTHON_BIN" \
		experiments/evaluate_all_checkpoints.py \
		--runs-root "$RESULTS_ROOT" \
		--output "$SUMMARY" \
		--manifest "$MANIFEST" \
		--device "$DEVICE" \
		--repeats 5

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" \
	PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$PYTHON_BIN" \
		experiments/summarize_uniform_eval.py

echo "post_optimization_complete summary=$SUMMARY"
