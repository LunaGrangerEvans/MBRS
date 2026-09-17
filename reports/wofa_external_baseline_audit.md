# WOFA external-baseline feasibility and reproducibility audit

审计日期：2026-09-13。范围是 CVPR 2025 方法 **Watermarking One for All:
A Robust Watermarking Scheme Against Partial Image Theft**（WOFA）的官方资源、
论文/supplement 协议和在 MBRS 中做统一评测的可行性。**本轮没有启动长训练，
没有修改 WOFA 模型或 checkpoint。**

## Executive verdict

WOFA 与 MBRS 的研究问题高度相关，但当前官方 release 还不能完成端到端推理。
作者仓库只有 temporary inference bundle：可验证的 Git-tracked
`embedder.pth` 和 `extractor.pth` 已提供，但 message-to-pattern 的
`encoder.pth` 与 pattern-to-message 的 `decoder.pth` 缺失。官方入口还硬编码
引用这两个缺失文件，并在 PyTorch 2.6+ 默认 `weights_only=True` 下先触发安全
反序列化错误。因此没有生成 WOFA BER、PSNR、SSIM、LPIPS 或 crop 数值结果，
也没有伪造一个 64-bit adapter。

| 最终问题 | 结论 | 证据/边界 |
|---|---|---|
| 1. official code available? | **Yes, partial** | 作者账号的公开仓库有 `inference/`；README 明确称其为 temporary version，完整代码尚未发布。 |
| 2. official pretrained weights available? | **Partial, not sufficient** | Git-tracked `embedder.pth`/`extractor.pth` 有 hash；端到端必需的 `encoder.pth`/`decoder.pth` 不在仓库。 |
| 3. can inference run? | **Component smoke yes; end-to-end no** | 官方 `inf4all.py` 退出 1；已提供组件可在独立环境完成 200×200 I/O 和 finite smoke。 |
| 4. can training/fine-tuning run? | **No official training pipeline in release** | 没有 training code、requirements/environment、dataset generation 或 reproducible command。 |
| 5. payload length? | **30 bits** | 论文 implementation details 的 `l=30`；Stage-1 decoder 最后一层也是 30。 |
| 6. image resolution? | **200×200 for WOFA experiments/release** | OPA block size 和 Table 3 均为 200×200；官方入口把输入 fit 到 200×200。 |
| 7. crop / partial-theft definition? | **Binary mask of stolen content, 1–95% area; irregular dataset masks** | 面向 partial masking → geometry → new-background fusion 的 full-canvas process。比例是 stolen/mask area，不是 MBRS retained-area。 |
| 8. unified 50-image evaluation feasible? | **Not with current release** | 缺两个 Stage-1 weights，且 200/30-bit、background fusion、mask semantics 与 frozen MBRS 不同。资源补齐后可做明确标注的 external reference run。 |
| 9. comparison fairness with MBRS? | **Not fair for strict BER or raw quality ranking** | payload、resolution、mask distribution、background、training data、metric contract 都不同。 |
| 10. recommended paper role? | **External reference; numeric values paper-reported only for now** | 不进入 `STRICT` 或统一 BER 主表；论文中的 WOFA 数值标为 `PAPER-REPORTED REFERENCE`。 |

## 1. Source and provenance audit

### Official first-party sources

| Source | Classification | What was verified |
|---|---|---|
| [CVPR Open Access paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Watermarking_One_for_All_A_Robust_Watermarking_Scheme_Against_Partial_CVPR_2025_paper.html) and [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Watermarking_One_for_All_A_Robust_Watermarking_Scheme_Against_Partial_CVPR_2025_paper.pdf) | Official publication | Authors, method definition, protocol, datasets, tables 1–6. |
| [CVPR supplementary PDF](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Liu_Watermarking_One_for_CVPR_2025_supplemental.pdf) | Official supplementary | Tables 7–10 and extra partial-mask/geometry results. |
| [Author-account GitHub repository](https://github.com/gzlIU-nOthOrsE/Watermarking-One-for-All_cvpr.25) | **Official-author release, strong attribution** | Repository description is the exact paper title; author profile lists this repository as its primary paper code. The audit pins commit `af883000f2dac570059051e629c873ae4386fc13`. |
| [Author repository at pinned commit](https://github.com/gzlIU-nOthOrsE/Watermarking-One-for-All_cvpr.25/tree/af883000f2dac570059051e629c873ae4386fc13) | Official code snapshot | `inference/`, model source, sample images, and two Git-tracked serialized modules. |

The account/identity link is strong but not institutionally verified from a profile
page: the paper lists first-author contact `gzliu24@m.fudan.edu.cn`, while the GitHub
handle is `gzlIU-nOthOrsE`. I therefore call it **official-author release** rather
than claiming an independently verified institutional GitHub identity.
Searches for a Fudan personal/lab page tied to `gzliu24` or the exact WOFA title did
not return a stronger first-party code/weight location.

### External and untrusted sources

- The repository's [`inference/models/Download_here.md`](https://github.com/gzlIU-nOthOrsE/Watermarking-One-for-All_cvpr.25/blob/af883000f2dac570059051e629c873ae4386fc13/inference/models/Download_here.md)
  points to a Baidu share for oversized `.pth` files. The share has no filename
  manifest, SHA-256, checkpoint metadata, or model-to-paper binding in the GitHub
  release. The request returned Baidu `errno=9019 need verify`; no file was downloaded
  or used. This is an author-referenced external link, but its artifact identity is
  **not auditable enough for a checkpoint result**.
- Exact-title GitHub/Gitee searches found no separately verifiable third-party
  WOFA reimplementation. This is a bounded search result, not proof that no private
  or unindexed fork exists.
- Secondary pages/mirrors such as CSDN, Bytez, ResearchGate, and stale
  Papers with Code indexing were not used for code, weights, or numeric results.

### Local storage and hashes

The clone and all runtime/generated artifacts are outside the workspace repository:

```text
repo:    /mnt/wmcontent/GLX/icassp/MBRS/external_repos/wofa
env:     /mnt/wmcontent/GLX/icassp/MBRS/envs/wofa
outputs: /mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/wofa_smoke
workspace view: external_baselines/repos -> /mnt/wmcontent/GLX/icassp/MBRS/external_repos
workspace view: external_baselines/envs  -> /mnt/wmcontent/GLX/icassp/MBRS/envs
workspace view: external_baselines/outputs -> /mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs
```

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `inference/saved_models/embedder.pth` | 4,822,639 | `3235c60bd99c133c4a3416188ab574ba7b547a4569df72dde1e350927b5879e7` |
| `inference/saved_models/extractor.pth` | 4,820,109 | `09a483823c29d43858863d3e5aec29a887a0cbed5724d2c9ba86eb6220184c6c` |
| repository HEAD | — | `af883000f2dac570059051e629c873ae4386fc13` |

The two `.pth` files are full serialized PyTorch modules, not plain state dicts.
They were loaded only with `weights_only=False` inside the pinned author checkout;
the audit code documents this trust boundary and does not apply it to unknown files.

## 2. Official-code feasibility check

The official repository currently contains:

```text
inference/I_ori.png
inference/I_bg.png
inference/I_mask.png
inference/model.py
inference/Stage1_Model.py
inference/inf4all.py
inference/inf4stage1.py
inference/saved_models/embedder.pth
inference/saved_models/extractor.pth
inference/models/Download_here.md
```

It does not contain:

```text
inference/models/encoder.pth
inference/models/decoder.pth
```

The official [`inf4all.py`](https://github.com/gzlIU-nOthOrsE/Watermarking-One-for-All_cvpr.25/blob/af883000f2dac570059051e629c873ae4386fc13/inference/inf4all.py)
uses the absent paths at lines 18 and 37. It also calls `torch.load` without an
explicit `weights_only=False`; on the audit runtime (`torch 2.7.0+cu126`) this
fails on the serialized `model.Embedder` before the missing-path error can be
reached. These are release/runtime blockers, not changes made by this project.

The dedicated environment is a separate `/mnt` venv with system-site packages,
not the MBRS workspace environment:

```text
Python 3.12.8
PyTorch 2.7.0+cu126
torchvision 0.22.0+cu126
NumPy 1.26.4, Pillow 11.2.1, einops 0.8.1, kornia 0.8.3, OpenCV 4.10.0
CUDA available; test restricted to CUDA_VISIBLE_DEVICES=0
```

### Smoke result

Command:

```bash
cd /root/workspace/GLX/icassp/MBRS
CUDA_VISIBLE_DEVICES=0 \
  /mnt/wmcontent/GLX/icassp/MBRS/envs/wofa/bin/python \
  external_baselines/wofa_audit.py --device cuda
```

Result: `component_smoke_pass_end_to_end_blocked`.

- Official embedder output: finite `1×3×200×200`, range approximately `[-0.02,0.02]`.
- Watermarked tensor: finite `1×3×200×200`, clamped to `[0,1]`.
- Supplied mask/background composition: finite `1×3×200×200`.
- Official extractor output: finite `1×1×200×200`.
- The injected pattern was an all-zero synthetic pattern, not an encoded 30-bit
  message; no accuracy claim follows.

Full details are in [wofa_smoke_test.md](wofa_smoke_test.md) and the generated
JSON under `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/wofa_smoke/`.

## 3. Protocol extracted from paper and supplementary

### 3.1 Input, payload, and architecture

| Item | WOFA protocol |
|---|---|
| Input resolution | `200×200` in the main experiments: 61,990 OPA-derived blocks are cropped to 200×200; Table 3 reports WOFA at 200×200. |
| Payload | `l=30` binary bits; no ECC or message-detection contract is reported. |
| Stage I encoder | Maps the 30-bit message to a grayscale `1×200×200` noise pattern. The paper describes two FC layers; the released `MessageProcessor` uses `Linear(900,3000)` then `Linear(3000,10000)`, reshapes to `1×100×100`, and upsamples ×2. |
| Stage I decoder | Four Conv-BN-ReLU blocks with channels `1→32→64→128→256`, max-pooling `2,2,2,5`, flatten 6400, then FC `6400→1024→30` and sigmoid. |
| Stage II embedder | Receives one noise-pattern channel plus RGB image (`4` channels), predicts a 3-channel residual, and adds it to the image. The released source scales the tanh residual by `0.02` and clamps the watermarked image to `[0,1]`. |
| Stage II extractor | Receives RGB and predicts a one-channel `200×200` pattern. The paper says both embedder and extractor use depth-5 U-Net structures. The temporary source defines `conv5/up6/conv6` but comments the bottleneck path out in `forward`; its effective executed graph is therefore not identical to the prose description. |
| Final decoder | Reuses the frozen Stage-I decoder to map extracted pattern to 30 sigmoid outputs; the corresponding checkpoint is absent. |

The released code is valuable for auditing the tensor contract, but it is not enough
to reconstruct the trained method: Stage-I weights and the complete training graph are
not released in this checkout.

### 3.2 Partial-image-theft process

The paper defines three ordered operations:

1. **Partial masking:** obtain stolen content from the original image with a binary
   mask `m∈{0,1}^{1×H×W}`.
2. **Geometric transformation:** transform the stolen part before fitting it to the
   new background.
3. **Background fusing:** cover the transformed stolen part on a new background
   `I_b∈R^{3×H×W}`. The paper assumes the new background has the same spatial size as
   the original, so the fused output remains on the original canvas.

The paper's area ratio is the area of the **mask/stolen part**, not the retained part.
The main partial-mask result uses bins `0–10, 10–20, ..., 90–100%`; the detailed
partial setting is described as a mask-area range from 1% to 95%. The OPA dataset
provides original image, mask, and background examples. The paper says the dataset
masks are random and irregular; it does not publish the mask sampler, seed, shape
distribution, or exact per-bin sampling procedure.

The supplied official inference sample is consistent with full-canvas compositing:
`I_compo = mask * I_w + (1-mask) * I_bg`. Its `I_mask.png` is a non-rectangular
anti-aliased silhouette (223 grayscale values, about 12.1% nonzero), even though the
paper's mathematical definition calls `m` binary. This is a sample-code observation,
not a claim about all training masks.

### 3.3 Crop, resize, background, and geometry details

| Question | Finding |
|---|---|
| crop-and-resize? | **Not specified as a standalone crop-and-resize-back attack.** The paper models stolen and fused images on the same `H×W` canvas. Internal resampling during rotation/scaling is not specified. |
| mask-on-canvas? | **Yes.** Same-size `I_b` and `I_f`, plus the official sample's pixelwise mask composition. |
| random shape? | **Yes in the dataset/late Stage-I description:** random irregular shapes, 1–95% area. |
| rectangle? | **Yes only as an early Stage-I curriculum component:** first 1% uses random squares covering at least 50%; next 10% uses random rectangles covering at least 25%. The final dataset protocol is not rectangle-only. |
| new background? | **Yes.** The fused image covers the transformed stolen part onto an unknown background. |
| translation | Maximum 50% of image side length/width; progressive range reaches maximum by 50% of training. |
| rotation | Maximum 45°; progressive range reaches maximum by 50% of training. |
| scaling | Maximum ±25%; progressive range reaches maximum by 50% of training. |
| Gaussian noise | Stage I pattern noise, standard deviation ramps from 0 to 0.1 within first 20% of training. |
| JPEG | Stage II JPEG QF=95. |
| resize | Stage II resizing range 80–125%. The paper does not give the interpolation/kernel or exact sampling distribution. |
| combined distortions | Stage II applies partial theft plus geometry at max intensity rather than the gradual Stage-I curriculum, then adds JPEG and resizing. |

### 3.4 Training and datasets

- **Two stages:** Stage I trains encoder+decoder only. Stage II freezes encoder and
  decoder and trains embedder+extractor through the full pipeline.
- Stage I loss is BCE between the original 30-bit message and decoder prediction.
- Stage II uses equal-form loss notation
  `L2 = MSE(Io, Iw) + MSE(wn', wpn) + L1`. No additional coefficients are
  published in the paper.
- Optimizer is Adam with learning rate `5e-5`; the reported hardware is one RTX
  3090 Ti. Epoch counts, batch size, random seed, exact schedule length, and checkpoint
  selection rule are not reported in the paper.
- Main data: 61,990 images sampled from OPA, cropped as 200×200 blocks and split 4:1.
  The wording does not expose a file-level manifest or exact train/test identities.
- Generalization: SOIM and self-constructed `matteImageNet`; the latter uses
  MatteAnything on 977 images from 19 selected ImageNet classes with corresponding
  masks.
- No official dataset download, OPA preprocessing script, partial-theft generator,
  training script, requirements file, or environment lockfile is present in the
  author checkout.

### 3.5 Metrics, reported values, and baselines

The paper reports **BAR (bit accuracy rate)** for robustness and does not report raw
BER, exact-message success, detection/false-positive rate, or an ECC contract. Quality
uses PSNR, SSIM, and LPIPS. The paper does not specify the exact image-range,
aggregation, LPIPS version/network, or crop-level quality metric contract used for
Table 3. It does not report MS-SSIM, Top25 local PSNR, P95 MSE, or Gini.

The MBRS side of the comparison is the existing
[external crop protocol](external_crop_protocol.md),
[controlled continuation protocol](controlled_continuation_protocol.md), and fixed
manifest at `/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt` with
SHA-256 `36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339`.

**PAPER-REPORTED REFERENCE, not re-run MBRS results:**

| WOFA paper setting | Value |
|---|---:|
| Visual input | 200×200, 30 bits |
| PSNR | 39.76 dB |
| SSIM | 0.9945 |
| LPIPS | 0.0188 |
| Clean BAR | 98.39% |
| Partial-mask BAR, 0–10 / 10–20 / 20–30 / 30–40 / 40–50% stolen area | 89.33 / 94.90 / 96.42 / 97.61 / 98.23% |
| Partial-mask BAR, 50–60 / 60–70 / 70–80 / 80–90 / 90–100% stolen area | 98.55 / 98.61 / 98.66 / 98.62 / 98.67% |
| Partial + translation 10 / 25 / 50% | 91.97 / 93.25 / 87.93% |
| Partial + rotation 15 / 30 / 45° | 95.26 / 94.24 / 90.63% |
| Partial + scaling ±10 / ±20 / ±25% | 95.72 / 95.50 / 95.02% |
| Partial + mixed 2: TS+RT / TS+SC / RT+SC | 87.20 / 91.34 / 90.74% |
| Partial + mixed 3 | 87.06% |

The paper's comparison rows are **HiDDeN, DWSF, MBRS, PIMoG, StegaStamp, and
MuST**. It says all methods used in the experiments were retrained with open-source
code and paper configurations; this is not a released WOFA training artifact and
does not make the rows comparable to the current MBRS frozen evaluator.

## 4. Reproduction decision

### What can be done now

- Keep the pinned official repo, two verified Git-tracked weights, dedicated venv,
  source hashes, and component smoke artifact under `/mnt/wmcontent`.
- Cite WOFA's paper-reported BAR/quality as a **paper-only numeric reference** with
  its native OPA/200×200/30-bit/irregular-mask protocol explicitly stated.
- Use the protocol comparison in the next paper revision to explain why WOFA is
  highly relevant but not a strict MBRS baseline.

### What is blocked

- No valid 30-bit message extraction can run without the missing Stage-I encoder and
  decoder checkpoints.
- No official training/fine-tuning run can be reproduced from this checkout: the
  training code, mask generator, OPA split, hyperparameters beyond a few values, and
  full checkpoint set are missing.
- No `wofa_results.csv` is generated. Creating one from a synthetic pattern or random
  Stage-I weights would be a fabricated reproduction.

### Conditional future run

If the authors publish both Stage-I weights with verifiable hashes and a complete
environment/inference contract, the next run can use the existing 50 host views only
as a separately labeled reference:

1. Preserve WOFA's native 30-bit message contract and 200×200 input; do not pad or
   force it to 64 bits.
2. Define a new 200×200 view of the fixed 50-image manifest and record its hash; do
   not silently call it the frozen 128×128 MBRS view.
3. Run WOFA's native irregular-mask/background protocol separately.
4. Optionally run a clearly named `WOFA-on-MBRS-rectangle-canvas` probe using a
   200×200 rectangle mask and an explicit background. This must not be merged with
   MBRS zero-outside retained-mask BER.
5. Recompute PSNR/SSIM/LPIPS/Top25 local PSNR/P95 MSE/Gini only after fixing the
   resampling, RGB range, background, output quantization, and aggregation contract.

## 5. Final recommendation for the MBRS paper

WOFA is **more relevant than HiDDeN to the paper's partial-retention research
question**, because it explicitly targets partial theft, unknown background, and
geometry. It is **not currently a better primary executable external baseline**:
HiDDeN already has a documented 64-bit unified reimplementation on the current
manifest, while WOFA cannot decode a real message with the released bundle.

Recommended wording/role:

> WOFA is an official-author external reference and paper-reported reference for
> partial-image-theft robustness. Its reported values are not an apples-to-apples
> comparison with MBRS because WOFA uses 200×200 images, 30-bit payloads, OPA-derived
> irregular stolen masks, fused backgrounds, and BAR rather than the MBRS 128×128,
> 64-bit, rectangle retained-mask, raw-BER contract.

Do **not** start “WOFA + our local-tail loss” now. The release lacks the full trained
pipeline and Stage-I weights; a self-designed 30-bit/200×200 recipe would be a new
method, not a faithful WOFA fine-tune. Revisit only after the official training
pipeline, complete checkpoints, dataset/mask code, and a declared 30-bit controlled
fine-tuning protocol are available.
