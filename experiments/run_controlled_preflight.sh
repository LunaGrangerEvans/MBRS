#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONTENT_ROOT="${CONTENT_ROOT:-/mnt/wmcontent/GLX/icassp/MBRS}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge/bin/python}"
GPU="${GPU:-0}"
SOURCE="$CONTENT_ROOT/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth"
PREFLIGHT_ROOT="$CONTENT_ROOT/experiments/controlled_preflight"
REPORT="$PROJECT_ROOT/reports/controlled_reproducibility_preflight.md"

cd "$PROJECT_ROOT"
mkdir -p "$PREFLIGHT_ROOT"
run_a="$(mktemp -d "$PREFLIGHT_ROOT/repeat-a.XXXXXX")"
run_b="$(mktemp -d "$PREFLIGHT_ROOT/repeat-b.XXXXXX")"

for output in "$run_a" "$run_b"; do
	CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
		experiments/train_local_patch.py \
		--config "$SCRIPT_DIR/config_finetune_global_control_128_m64.json" \
		--output-dir "$output" \
		--device cuda \
		--seed 17 \
		--epochs 1 \
		--max-train-batches 1 \
		--max-val-batches 1 \
		--init-checkpoint "$SOURCE"
done

"$PYTHON_BIN" - "$run_a" "$run_b" "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

import torch

left = Path(sys.argv[1])
right = Path(sys.argv[2])
report = Path(sys.argv[3])

def row(path, name):
    value = json.loads((path / name).read_text().splitlines()[0])
    value.pop("elapsed_s", None)
    return value

max_metric_difference = 0.0
for name in ("train.jsonl", "val.jsonl"):
    a = row(left, name)
    b = row(right, name)
    for key in a:
        if isinstance(a[key], (int, float)):
            max_metric_difference = max(max_metric_difference, abs(a[key] - b[key]))
        elif a[key] != b[key]:
            raise SystemExit("non-numeric preflight mismatch: {} {}".format(name, key))

def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)

a_checkpoint = load(left / "checkpoint_0001.pth")
b_checkpoint = load(right / "checkpoint_0001.pth")
max_tensor_difference = max(
    float((a_checkpoint["model"][key] - b_checkpoint["model"][key]).abs().max())
    for key in a_checkpoint["model"]
)
passed = max_metric_difference == 0.0 and max_tensor_difference == 0.0
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(
    "# Controlled reproducibility preflight\n\n"
    "Two independent single-GPU deterministic one-batch continuations were started "
    "from the same seed17 Global checkpoint.\n\n"
    "- Maximum logged-metric difference (excluding elapsed time): `{:.3e}`\n"
    "- Maximum model-tensor difference after one update: `{:.3e}`\n"
    "- Result: **{}**\n"
    "- Repeat A: `{}`\n"
    "- Repeat B: `{}`\n".format(
        max_metric_difference,
        max_tensor_difference,
        "PASS" if passed else "FAIL",
        left,
        right,
    )
)
if not passed:
    raise SystemExit("controlled reproducibility preflight failed")
print("controlled_reproducibility_preflight=PASS")
PY
