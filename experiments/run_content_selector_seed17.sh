#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/GLX/icassp/MBRS
CONTENT_ROOT=/mnt/wmcontent/GLX/icassp/MBRS
RUN_NAME=seed17_contentaware_gradient_patch16_stride8_top10_alpha1_global0.5_local0.5
RUN_OUTPUT="$CONTENT_ROOT/experiments/runs/$RUN_NAME"
SOURCE_CHECKPOINT="$CONTENT_ROOT/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth"
if [[ -e "$RUN_OUTPUT" ]]; then
    echo "refusing to reuse output: $RUN_OUTPUT" >&2
    exit 2
fi
if pgrep -f '[e]xperiments/train_local_patch.py' >/dev/null; then
    echo 'another MBRS training process is active' >&2
    exit 3
fi
python - <<'PY'
import hashlib
import json
from pathlib import Path
root=Path('/mnt/wmcontent/GLX/icassp/MBRS')
d=json.loads((root/'reports/content_selector/decision.json').read_text())
assert d['training_justified'] and d['selected']['feature']=='gradient' and d['selected']['alpha']==1
source=root/'experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth'
with source.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()=='1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907'
print('validated fixed formulation and source checkpoint', flush=True)
PY
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
exec /root/miniforge/bin/python experiments/train_local_patch.py \
    --config "experiments/config_${RUN_NAME}.json" \
    --output-dir "$RUN_OUTPUT" --device cuda --seed 17 \
    --init-checkpoint "$SOURCE_CHECKPOINT"
