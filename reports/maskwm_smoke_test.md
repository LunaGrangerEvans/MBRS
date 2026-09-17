# MaskWM smoke test

日期：2026-09-13。MaskWM 官方 D_64bits 和 ED_64bits 均完成真实端到端 smoke；
本文件同时记录固定 50-image reference 的执行边界。没有训练或微调。

## Official single-image run

The official repository was cloned at commit
`00864624446a57d2cf5d56dab73095ef64c6ade6`. The official relative
`checkpoints/` directory is a symlink inside the external clone to
`/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/checkpoints/maskwm`; the source
checkout itself was not edited.

```bash
cd /mnt/wmcontent/GLX/icassp/MBRS/external_repos/maskwm
CUDA_VISIBLE_DEVICES=0 \
  /mnt/wmcontent/GLX/icassp/MBRS/envs/wofa/bin/python \
  inference.py --device cuda:0 --model_name D_64bits --image_name 00
CUDA_VISIBLE_DEVICES=0 \
  /mnt/wmcontent/GLX/icassp/MBRS/envs/wofa/bin/python \
  inference.py --device cuda:0 --model_name ED_64bits --image_name 00
```

| Variant | Exit | PSNR | SSIM | Bit accuracy | Watermarked IoU | Unwatermarked IoU |
|---|---:|---:|---:|---:|---:|---:|
| D_64bits | 0 | 37.33 dB | 0.9668 | 1.0000 | 0.9816 | 0.9867 |
| ED_64bits | 0 | 36.26 dB | 0.9669 | 1.0000 | 0.9888 | 0.9920 |

The official entry uses 512×512 image tensors, internally calls the model at 256×256,
and writes `orig`, `wm`, `res`, and predicted-mask PNGs under the external clone's
`results/{model_name}` directory. Its random message is 64 bits for these variants.

### Environment note

MaskWM imports `diffusers` and `compressai` through the VAE noise module even when the
default inference does not select VAE noise. The separate venv under
`/mnt/wmcontent/GLX/icassp/MBRS/envs/wofa` uses:

```text
Python 3.12.8
PyTorch 2.7.0+cu126
torchvision 0.22.0+cu126
diffusers 0.32.2
huggingface-hub 0.36.2
compressai 1.2.8
omegaconf 2.3.1, kornia 0.8.3, Pillow 11.2.1
```

The MBRS environment was not changed. A later `torch.load(..., strict=True)` check
confirmed both official 64-bit state dicts have 629 keys and no missing/unexpected
keys.

## Fixed 50-image reference

The reproducible adapter is
[external_baselines/maskwm_adapter.py](../external_baselines/maskwm_adapter.py).
It preserves the native 64-bit message contract and does not edit the official
source. Quality outputs are saved as RGB PNGs at 128×128 after the declared
512→256→512 native path; the frozen MBRS evaluator then computes PSNR, SSIM, LPIPS,
Top25 local PSNR, P95 MSE, and Gini.

Because a direct 128×128 comparison would include the unavoidable 128→512→128
resampling floor, these quality metrics use the same official-style round-tripped
host view as the evaluator reference. The metric formulas remain frozen, but the
view/protocol is explicitly `REFERENCE`, not strict MBRS quality.

Crop decoding is a separate exact MBRS rectangle-retained-mask probe:

```text
fused = watermarked_tensor * fixed_rectangle_mask
```

The exterior is normalized zero, matching MBRS's frozen crop tensor semantics. This is
not MaskWM's native `I_wm*M + I_orig*(1-M)` fused-background inference and not the
official CropResize fine-tuning attack.

| Variant | Clean BA / BER | BA 100% | BA 70% | BA 50% | BA 40% | BA 30% |
|---|---:|---:|---:|---:|---:|---:|
| D_64bits | 1.0000 / 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ED_64bits (full encoder mask) | 1.0000 / 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The fixed-manifest quality/reference summary is
[maskwm_results.csv](maskwm_results.csv); raw outputs and variant provenance are
under `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm/`.

## Status

```text
official D_64bits checkpoint: PASS
official ED_64bits checkpoint: PASS
official encoder load: PASS
official decoder load: PASS
single-image encode/decode: PASS
fixed 50-image 64-bit reference: PASS
MBRS rectangle crop probe: PASS, separate reference protocol
MaskWM native CropResize fine-tuned 64-bit run: NOT AVAILABLE (no official 64-bit FT file)
long training/fine-tuning: NOT STARTED
```
