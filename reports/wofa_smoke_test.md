# WOFA smoke test

日期：2026-09-13。该文件记录失败边界和已通过的组件检查；它不是端到端
message-accuracy 结果。

## Commands

The official entry was run unmodified from the pinned author checkout:

```bash
cd /mnt/wmcontent/GLX/icassp/MBRS/external_repos/wofa/inference
CUDA_VISIBLE_DEVICES=0 \
  /mnt/wmcontent/GLX/icassp/MBRS/envs/wofa/bin/python inf4all.py
```

Exit status: **1**.

Observed failure on PyTorch 2.7.0+cu126:

```text
_pickle.UnpicklingError: Weights only load failed
Unsupported global: GLOBAL model.Embedder
```

The script's intended paths then reveal a second independent blocker:

```text
inference/models/encoder.pth   missing
inference/models/decoder.pth   missing
```

The author README calls the release temporary and provides only a Baidu-share link
for oversized files. Since that share did not expose an auditable file/hash manifest,
it was not downloaded or used.

## Component-only smoke

The reproducible runner is
[external_baselines/wofa_audit.py](../external_baselines/wofa_audit.py). It loads only
the two author-repository Git-tracked modules with the pinned commit and hashes, uses
the supplied `I_ori.png`, `I_bg.png`, and `I_mask.png`, and injects a synthetic all-zero
`1×200×200` pattern. It does **not** claim that this pattern corresponds to a valid
30-bit message.

```bash
cd /root/workspace/GLX/icassp/MBRS
CUDA_VISIBLE_DEVICES=0 \
  /mnt/wmcontent/GLX/icassp/MBRS/envs/wofa/bin/python \
  external_baselines/wofa_audit.py --device cuda
```

Result: `component_smoke_pass_end_to_end_blocked`.

| Tensor | Result |
|---|---|
| Input image | RGB `1×3×200×200`, float `[0,1]` |
| Embedder residual | finite `1×3×200×200`, approximately `[-0.02,0.02]` |
| Clamped watermarked image | finite `1×3×200×200`, `[0,1]` |
| Mask | finite `1×1×200×200`, supplied anti-aliased silhouette |
| Fused image | finite `1×3×200×200` |
| Extractor pattern | finite `1×1×200×200` |
| Message output | **not run**; decoder checkpoint absent |

Raw JSON artifact:

```text
/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/wofa_smoke/component_smoke.json
```

No `reports/wofa_results.csv` was produced because no valid message-level or frozen-quality evaluation completed.

