#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_MULTI_SEED:-0}" != "1" ]]; then
	echo "multi-seed expansion disabled by current single-seed experiment policy" >&2
	exit 4
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONTENT_ROOT="${CONTENT_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge/bin/python}"
GPU="${GPU:-0}"
SELECTION="$CONTENT_ROOT/reports/controlled_soft_temperature_seed17.json"

cd "$PROJECT_ROOT"

temperature="$($PYTHON_BIN - "$SELECTION" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))["selected_temperature"]
print("none" if value is None else value)
PY
)"

case "$temperature" in
	0.25)
		soft_suffix=t025
		soft_config="$SCRIPT_DIR/config_finetune_soft_tail_patch16_weight50_t025_128_m64.json"
		;;
	0.5)
		soft_suffix=t05
		soft_config="$SCRIPT_DIR/config_finetune_soft_tail_patch16_weight50_t05_128_m64.json"
		;;
	1.0|1)
		soft_suffix=t10
		soft_config="$SCRIPT_DIR/config_finetune_soft_tail_patch16_weight50_t10_128_m64.json"
		;;
	none)
		echo "no Soft temperature passed seed17; final seeds not started"
		exit 0
		;;
	*)
		echo "unsupported selected temperature: $temperature" >&2
		exit 2
		;;
esac

run_branch() {
	local seed="$1"
	local label="$2"
	local config="$3"
	local source="$CONTENT_ROOT/experiments/runs/optimization_global_seed${seed}_128_m64_crop/checkpoint_0100.pth"
	local output="$CONTENT_ROOT/experiments/runs/controlled_seed${seed}_${label}"
	local log_dir="$CONTENT_ROOT/logs/controlled-seed${seed}"
	local log="$log_dir/${label}.log"
	if [[ ! -f "$source" ]]; then
		echo "missing source checkpoint: $source" >&2
		exit 1
	fi
	if [[ -e "$output" ]]; then
		echo "refusing to reuse controlled output directory: $output" >&2
		exit 3
	fi
	mkdir -p "$log_dir"
	echo "starting controlled branch=$label seed=$seed gpu=$GPU"
	CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
		experiments/train_local_patch.py \
		--config "$config" \
		--output-dir "$output" \
		--device cuda \
		--seed "$seed" \
		--init-checkpoint "$source" \
		2>&1 | tee "$log"
}

for seed in 29 41; do
	run_branch "$seed" global_continuation \
		"$SCRIPT_DIR/config_finetune_global_control_128_m64.json"
	run_branch "$seed" hard_p16_t25_l50 \
		"$SCRIPT_DIR/config_finetune_hard_patch16_weight50_128_m64.json"
	run_branch "$seed" "soft_p16_${soft_suffix}_l50" "$soft_config"
done

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
	experiments/evaluate_controlled_multiseed.py \
	--device cuda \
	--temperature "$temperature"

echo "controlled_final_seeds_complete temperature=$temperature"
