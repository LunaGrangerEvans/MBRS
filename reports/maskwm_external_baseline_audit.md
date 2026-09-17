# MaskWM external-baseline audit

审计日期：2026-09-13。目标方法是 NeurIPS 2025 **Mask Image Watermarking
(MaskWM)**。本轮只做官方资源、协议、端到端 smoke、fixed-manifest reference
评估和 qualitative visualization；**没有启动长训练、没有修改 MaskWM
upstream、没有微调 checkpoint**。

## Executive verdict

MaskWM 是目前最适合作为论文第二个可视化 external baseline 的候选：它有作者
官方 repo、作者 Hugging Face 权重、完整的 encoder/decoder inference 路径，且
原生支持 64 bits；相比 TrustMark，它没有 BCH/ECC payload 变换，视觉 operating
point 也更接近当前论文的 qualitative 需求。

官方 `D_64bits` 和 `ED_64bits` 均已成功端到端运行。fixed 50-image reference
也已完成；但它仍不是 strict MBRS baseline，因为 MaskWM 的训练/推理分辨率是
256/512，当前 reference 输出再下采样为 128，且 MBRS crop probe 与 MaskWM
native CropResize 不是同一攻击协议。

| 最终问题 | 结论 |
|---|---|
| MaskWM 是否成功端到端运行？ | **Yes**。官方 D_64bits、ED_64bits 均完成 encode → masked fusion → decoder message/localization。 |
| 64-bit official checkpoint 是否可用？ | **Yes**。D_64bits 和 ED_64bits 均可从作者 Hugging Face 直接下载并 strict-load。 |
| 是否适合作为第二个 qualitative external baseline？ | **Yes，推荐使用 D_64bits**；它是 global 64-bit model，且已有同 host/message 的 fixed visualization。 |
| 是否比 TrustMark 更适合主 qualitative figure？ | **Yes，作为 visual reference 更合适**：64 raw bits、无 ECC，视觉 fidelity 没有 TrustMark 那么悬殊；仍需注明 256/512 native path 和 common 512 display canvas。 |
| 是否能做 matched-PSNR comparison？ | **Yes**。已用预定义官方 JND/μ grid 做 strength sweep；不改 decoder 或训练。native/default 仍保持 `μ=1.3`。 |
| 若不能，WAM 是否应作为 fallback？ | **不需要 fallback**。MaskWM 已成功运行；WAM 本轮未 clone/download。 |

## 1. Official resources and provenance

### First-party sources

| Source | Classification | Verified content |
|---|---|---|
| [NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d74fb06615baaf6349007da7521223aa-Abstract-Conference.html) | Official paper record | Authors, D/ED design, global/local goals, 256/512 evaluation, training cost. |
| [Author GitHub repo](https://github.com/hurunyi/MaskWM) | **Official author repo** | Repository title says official implementation; contains `inference.py`, `train.py`, model/config/noise/mask source. |
| [Author GitHub profile](https://github.com/hurunyi) | First-party author profile | Profile identifies Runyi Hu and pins MaskWM as “[NeurIPS 2025] Mask Image Watermarking (Official Implementation)”. |
| [Author Hugging Face model repo](https://huggingface.co/Runyi-Hu/MaskWM) | **Official author weights** | Model card links the GitHub repo and lists the MaskWM paper; files are in the author namespace. HF revision at audit: `a4aacf5c9769a25482dc087b39ccd57354c3f030`. |
| [Official HF file tree](https://huggingface.co/Runyi-Hu/MaskWM/tree/main) | Official checkpoint inventory | D/ED 32/64/128 variants and official fine-tuned files. |

The repository clone is pinned at:

```text
https://github.com/hurunyi/MaskWM
commit: 00864624446a57d2cf5d56dab73095ef64c6ade6
branch: master
```

### Checkpoint inventory

The following files were found in the official HF author namespace. The `oid` values
below are HF file/object identifiers; the local SHA-256 values are computed after
download and are the hashes used by this audit.

| Official filename | Variant | HF bytes | HF oid | Local status |
|---|---|---:|---|---|
| `D_32bits.pth` | global | 253,735,941 | `f709ba1276a4a29c2e123339932d8f90dfa16b3f` | not downloaded |
| `D_64bits.pth` | global | 255,934,341 | `c503cab2f0d500447b96d3a38eb7683cfb5f10c8` | **downloaded; tested** |
| `D_128bits.pth` | global | 270,663,813 | `10089347388c3873b37c12dee27b548311e2ad6a` | not downloaded |
| `ED_32bits.pth` | adaptive/local | 253,738,245 | `91b6bb5484e1aa10299c66030195b16b0cc838b0` | not downloaded |
| `ED_64bits.pth` | adaptive/local | 255,936,645 | `0ba2d90ac04c6fa44e3de65d1b20236b1ca7f217` | **downloaded; tested** |
| `ED_128bits.pth` | adaptive/local | 270,666,117 | `823794dba5cd15c1f8785c9f8ae4df22fec9a0c2` | not downloaded |
| `D_32bits_crop&resize_ft.pth` | D crop-resize fine-tune | 253,735,941 | `24e2eebab8500cf29a5ead96ffa980e2cef8395b` | official inventory only |
| `D_32bits_move&resize_ft.pth` | D move-resize fine-tune | 253,737,203 | `b31b1efa6700522504429751b3780d0f4aa19927` | official inventory only |
| `D_32bits_vae_ft.pth` | D VAE fine-tune | 253,735,941 | `4476d9f56293c2ac55d73a7f16f8cf6f2ce6e646` | official inventory only |
| `ED_32bits_move&resize_ft.pth` | ED move-resize fine-tune | 253,739,507 | `b41c5c51847c52b949d02ad6888aa198795e428f` | official inventory only |
| `ED_32bits_vae_ft.pth` | ED VAE fine-tune | 253,738,245 | `9d86b31282e7cdd83f1a99100c9e83fe4538f1c1` | official inventory only |

Important limitation: the current official HF tree contains **D_32 crop&resize
fine-tuning**, but no `D_64bits_crop&resize_ft.pth` or
`ED_64bits_crop&resize_ft.pth`. Thus “official crop-resize fine-tuned variant” is
true, but not for a downloaded 64-bit checkpoint in the current tree.

| Downloaded checkpoint | Local path | Bytes | SHA-256 |
|---|---|---:|---|
| `D_64bits.pth` | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/checkpoints/maskwm/D_64bits.pth` | 255,934,341 | `0eb1b2bfe37e0479a74483a2d07cd4a28fd113c4d83c4e4396c12fad9e5393b2` |
| `ED_64bits.pth` | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/checkpoints/maskwm/ED_64bits.pth` | 255,936,645 | `8a0230442d7d20422cb59c00f6a2823acd7794a14dde8a66245673cfe0d19f9d` |

No third-party checkpoint or mirror was used. The HF model card is directly linked
from the official GitHub README, and both state dicts loaded with `strict=True`:

```text
state keys: 629
parameters: D 63,916,487; ED 63,917,063
message length: 64
encoder image_size: 256
missing keys: none
unexpected keys: none
```

## 2. Architecture and official variant semantics

### MaskWM-D: global embedding, local extraction

`D_64bits.yaml` sets `message_length=64`, `mask_channel=0` in the encoder, and
`mask_channel=1` in the decoder. The encoder receives the image and message feature
map but no user mask, so watermark embedding is global. At decoding, the model has a
U2-Net localization branch and a U-Net/CNN message branch; it can localize and extract
from a masked local region.

### MaskWM-ED: adaptive/local embedding and extraction

`ED_64bits.yaml` sets `mask_channel=1` in the encoder and decoder. The input mask is
concatenated into the encoder feature path, so the encoder is explicitly conditioned
to allocate watermark signal to the selected region. This is not merely D with a
different decoder mask: its intended operating point is local embedding.

For the fixed 50-image reference, D_64bits is the main qualitative baseline. ED_64bits
was also evaluated, but with a full encoder mask because the current MBRS manifest has
no registered semantic local masks. That ED row is therefore a full-mask adaptive-model
reference, not a local-mask claim.

### Encoder, decoder, and strength

The paper and released source define:

```text
message bits -> Linear(l, l*l) -> bilinear map -> ConvNormReLU feature map
image + message feature (+ mask for ED) -> U-Net -> intermediate encoded image
I_wm = I_orig + μ * JND(I_orig) * (I_enc - I_orig)
```

The official inference uses:

```text
D_64bits:  μ = 1.3, blue=True, no encoder mask
ED_64bits: μ = 1.75, blue=True, full encoder mask in this audit reference
```

The source normalizes RGB with `(image - 0.5) * 2` to `[-1,1]`, denormalizes with
`image/2+0.5`, and clamps outputs. Its JND module computes fixed luminance/contrast
heatmaps; `blue=True` scales the channel heatmap factors to approximately
`[0.75, 0.50, 1.00]`. The official inference first transforms a PIL input to 512×512,
then runs the learned model at 256×256 and upsamples the residual back to 512×512.

The source implementation is complete enough to load both encoder and decoder from a
single official state dict. There is no separate missing encoder/decoder issue like
the WOFA release.

## 3. Protocol extracted from paper and code

### Resolution and arbitrary-size behavior

| Item | MaskWM |
|---|---|
| Training image | Resize + center crop to `256×256`. |
| Official example/evaluation | `512×512` output canvas; model internally runs at `256×256`. |
| Arbitrary resolution | Supported by the paper's resolution-scaling algorithm: normalize input, interpolate to training `(m,n)`, compute residual, interpolate residual back to original `(H,W)`, clamp, convert to uint8. |
| Fixed-manifest adapter | Manifest `128×128` → official-style tensor resize to `512` → model `256` → `512` output → antialiased bilinear `128` RGB PNG for frozen MBRS quality metrics. |

The 128×128 exported view is a declared adapter contract, not the paper's native
512×512 evaluation. To avoid attributing a 128→512→128 resampling floor to the
watermark, frozen-evaluator quality uses the same round-tripped host as reference;
this is still `REFERENCE`, not strict MBRS quality.

### Masking, crop, and CropResize

MaskWM's native four-stage training pipeline is:

1. randomly generate a candidate mask;
2. embed the message globally (D) or using the mask (ED);
3. fuse watermarked and original image,
   `I_fuse = I_wm ⊙ M + I_orig ⊙ (1-M)`;
4. apply a distortion, then obtain `I_mask = I'_fuse ⊙ M` for message extraction.

The candidate-mask pool is LaMa-like and contains **Full, Rectangle, Irregular, and
Segment** masks. The released `MixedMaskEmbedder` uses default probabilities of
`1/4` for irregular, rectangle, full, and COCO segmentation masks, with
`invert_proba=0.5`. Rectangles are not the only shape.

The official `CropResize` noise layer is a real crop-and-resize-back operation:

```text
h_ratio, w_ratio ~ Uniform(min_ratio, max_ratio)
random top/left crop of the image and mask
resize both back to the original H×W
mask resize uses nearest interpolation
```

The official crop-resize fine-tuning config uses `CropResize(0.1, 1.0)`,
`num_training_steps=50000`, `ED_path=<pretrained model>`, and `full_mask_ft=true`.
The current official tree does not provide the corresponding 64-bit crop-resize
fine-tuned checkpoint.

### Distortions

Training value-metric distortion pool: JPEG QF 50, Gaussian filter kernel 1/sigma 5,
Gaussian noise σ 0.1, median filter kernel 5, salt-and-pepper ratio 0.1, resize,
brightness/contrast/hue/saturation. Training geometric pool: rotation sampled from
[-90°,90°], perspective scale [0.1,0.5], and horizontal flip.

Paper evaluation value-metric settings: JPEG QF 60, Gaussian filter kernel 1/sigma 3,
Gaussian noise σ 0.05, median kernel 3, salt-and-pepper ratio 0.05, resize factor 0.5,
brightness/contrast [0.7,1.3], hue [-0.1,0.1], saturation [0.7,1.3]. Geometric
evaluation uses rotation [-30°,30°], perspective [0.1,0.3], and horizontal flip.

### Dataset and evaluation protocol

- Training uses 83k images from MS-COCO 2014 training data.
- Training is 100k steps, batch size 16, one A6000, AdamW learning rate 1e-4,
  cosine scheduler with 2k warm-up steps; the paper reports about 20 hours.
- Easy-to-hard schedule: full mask/no distortion for first 0.5k steps; all masks
  from 0.5k–1k; distortions after 1k; decoder weight 20→0.2 over first 5k;
  mask loss weight 0.5; JND introduced/tuned from step 5k.
- Global evaluation uses 1,000 MS-COCO 2014 validation images at 512×512.
- Local evaluation uses 41k validation images divided into watermarked-area bins
  `1–5, 5–10, 10–20, ..., 90–95, 95–99%`; 400 samples per bin plus inverted masks,
  for 9,600 image-mask pairs.
- The area ratio is **watermarked/masked area**, not MBRS retained-area by default.
  Inverted masks are explicitly used to simulate both inpainting and outpainting.

## 4. Metrics and payload semantics

The official inference computes:

```text
message = random binary vector of length l
decoded_message.gt(0.5)
bit accuracy = mean(decoded_bits == target_bits)
```

The paper calls the robustness metric **Bit Accuracy**. It does not specify an ECC,
and the released training/inference path has no BCH or other message code. There is no
paper-reported exact-message success, detection rate, false-positive rate, or raw BER
table. For a fixed trial, raw BER can be described as `1 - bit_accuracy`, but this is
not an ECC-aware exact-message metric.

The paper's main global table uses PSNR, SSIM, and Bit Accuracy. It does not use the
current MBRS Top25 local PSNR, P95 MSE, Gini, or MS-SSIM. LPIPS appears in a visual
quality ablation/training discussion, but is not the main Table 1 evaluation column.

**Paper-reported 32-bit reference:**

| Method | Bits | PSNR | SSIM | No distortion BA | Valuemetric BA | Geometric BA |
|---|---:|---:|---:|---:|---:|---:|
| MaskWM-D | 32 | 39.55 | 0.9814 | 1.0000 | 1.0000 | 0.9998 |
| MaskWM-ED | 32 | 39.52 | 0.9828 | 1.0000 | 1.0000 | 1.0000 |

These are **PAPER-REPORTED REFERENCE** values from the paper's 512×512/COCO protocol,
not the fixed-manifest numbers below.

## 5. Smoke and fixed 50-image reference

### Official single-image smoke

Commands were run in the separate `/mnt/wmcontent/GLX/icassp/MBRS/envs/wofa` venv,
from the unmodified official checkout. A local `checkpoints` symlink points to the
external `/mnt` checkpoint directory only to satisfy the official relative path.

```bash
cd /mnt/wmcontent/GLX/icassp/MBRS/external_repos/maskwm
CUDA_VISIBLE_DEVICES=0 \
  /mnt/wmcontent/GLX/icassp/MBRS/envs/wofa/bin/python \
  inference.py --device cuda:0 --model_name D_64bits --image_name 00
CUDA_VISIBLE_DEVICES=0 \
  /mnt/wmcontent/GLX/icassp/MBRS/envs/wofa/bin/python \
  inference.py --device cuda:0 --model_name ED_64bits --image_name 00
```

Both exited 0. The official sample outputs were:

| Variant | PSNR | SSIM | Bit accuracy | Watermarked IoU | Unwatermarked IoU |
|---|---:|---:|---:|---:|---:|
| D_64bits | 37.33 dB | 0.9668 | 1.0000 | 0.9816 | 0.9867 |
| ED_64bits | 36.26 dB | 0.9669 | 1.0000 | 0.9888 | 0.9920 |

These are the official sample's random 64-bit messages and predefined example mask;
they are smoke evidence, not a statistical result.

The separate environment required `diffusers==0.32.2`, `huggingface-hub==0.36.2`,
and `compressai==1.2.8` because MaskWM's VAE module is imported unconditionally even
for the default inference path. The MBRS environment was not changed.

### Fixed 50-image results

The reference adapter uses the existing test manifest:

```text
manifest: /mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt
sha256: 36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339
images: 50 × 3 × 128 × 128
messages: 50 × 64
```

The 64-bit messages are passed unchanged to native D_64bits/ED_64bits. Quality is
computed by the current frozen evaluator after loading the saved 128×128 RGB PNGs.
Crop decoding uses the exact MBRS normalized zero-fill rectangle mask on the 512 canvas,
then the official decoder at 256; it is explicitly not MaskWM native CropResize.

| Method | PSNR | SSIM | MS-SSIM | LPIPS | Top25 local PSNR | P95 MSE | Gini | Clean BA | BA 100/70/50/40/30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MaskWM-D_64bits | 40.950378 | 0.983441 | 0.993848 | 0.00087915 | 40.105495 | 0.000107735 | 0.135103 | 1.0000 | 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 |
| MaskWM-ED_64bits* | 42.156009 | 0.987481 | 0.995649 | 0.00055338 | 41.099024 | 0.000088530 | 0.152835 | 1.0000 | 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 |

`*` ED is run with a full encoder mask in this fixed reference because no registered
per-image local mask exists in the MBRS manifest. All crop BA values are raw 64-bit
thresholded-bit accuracy over five fixed repeats; corresponding BER is zero in this
reference run. These unexpectedly strong crop numbers must not be generalized to
MaskWM's native 1–99% local-mask curves or treated as a strict MBRS superiority claim.

Machine-readable output: [maskwm_results.csv](maskwm_results.csv). Saved RGB outputs,
per-variant provenance, and all generated artifacts are under:

```text
/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm/
```

## 6. Qualitative-baseline decision

The generated [external qualitative protocol](external_qualitative_protocol.md) uses
five pre-registered validation indices `[0, 10, 20, 30, 40]` and the same host image
and fixed 64-bit message for every method:

```text
Original
HiDDeN-64 reimplementation
MaskWM-D_64 official global model
MBRS Global
Hard Top10
Hard Top10 + OKLab
```

The main/default figure has no residual amplification. Every method is rendered as
actual RGB, and zooms use the same fixed native 32×32 ROI chosen by the existing
incumbent Hard Top10 CIEDE2000-tail rule. The matched-PSNR figure re-runs the official
MaskWM encoder over a predeclared JND/μ grid and selects the closest per-sample
operating point to Hard Top10 PSNR; it does not change the decoder, checkpoint, or
training.

This makes MaskWM the recommended second qualitative external baseline. The roles are:

| Method | Visual role |
|---|---|
| HiDDeN-64 | Unified 64-bit reimplementation; lower quality operating point. |
| MaskWM-D_64 | **Primary second qualitative external baseline**; official global 64-bit model. |
| TrustMark-Q/P | Strong official external reference, but different 100-bit/BCH/ECC/preprocessing and much higher fidelity. |
| WOFA | Highly relevant paper-reported/external reference, but current official release cannot decode end-to-end. |

The main figure should label MaskWM as `official D_64bits reference` and mention the
256/512 native path plus common 512 display canvas in the caption. It should not claim
strict BER superiority over MBRS.

## 7. Storage and changed artifacts

Large files remain outside the workspace Git tree:

```text
repo:        /mnt/wmcontent/GLX/icassp/MBRS/external_repos/maskwm
checkpoints: /mnt/wmcontent/GLX/icassp/MBRS/external_baselines/checkpoints/maskwm
env:         /mnt/wmcontent/GLX/icassp/MBRS/envs/wofa
outputs:     /mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm*
visuals:     /mnt/wmcontent/GLX/icassp/MBRS/visualizations/external_qualitative
```

Workspace additions are limited to audit/adapter/renderer code, reports, and the
lightweight summary CSV. No checkpoint, environment, clone, or generated image is
tracked in Git.
