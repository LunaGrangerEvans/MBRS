# Frozen final-method configuration

This is the authoritative definition of the reader-facing method `Ours`. It was frozen before the formal project-test evaluation; no project-test result was used to tune any field.

## Reader-facing definition

`Ours = Hard Local-Tail + global OKLab regularization`.

```text
L = w_msg L_msg + w_g L_RGB + w_t L_tail + lambda_OK L_OKLab
L_tail = mean of hard Top-10% highest-error P16/S8 patch scores
L_OKLab = mean per-pixel OKLab distance in linear-sRGB OKLab
```
## Exact frozen values

| Field | Frozen value |
|---|---|
| Run | `seed17_crop_hard16_stride8_top10_global_oklab_g25` |
| Config | `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/config.resolved.json` |
| Config SHA-256 | `bfe4ccf7efa339e0eacf1bc19dc77aa3f88f8d1dce3dcd105bc62c6e3f314fa6` |
| Checkpoint | `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/checkpoint_0020.pth` |
| Checkpoint SHA-256 | `4a6bfedff9d79769a364ed189fc56a947e1c1115f9c247d4a86b70c6262e1ae4` |
| Source checkpoint | `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth` |
| Source checkpoint SHA-256 | `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907` |
| OKLab lambda | `0.059149764` |
| RGB/global weight | `0.5` |
| Hard local-tail weight | `0.5` |
| Message weight | `10.0` |
| Local loss | `topk` |
| Patch / stride | `16 / 8` |
| Top-k fraction | `0.1` (`ceil(225×0.1)=23` training patches) |
| Noise | `RandomCrop(0.3, 1.0)` |
| Continuation epochs | `20` |
| Learning rate | `0.0001` |
| Seed | `17` |
| Batch / workers | `16 / 0` |

## Optimizer and state restoration

The run uses `init_checkpoint`, not `resume`: the source model state (including BatchNorm buffers) and saved Adam optimizer state are loaded, the optimizer learning rate is reset to the frozen config value, and the new 20-epoch schedule starts at epoch 1. The resolved config has `resume=null`; the epoch-20 checkpoint contains both `model` and `optimizer` state.

- Optimizer-state present: `True`.
- Source checkpoint epoch: `100`.
- No additional inference module is introduced; local-tail and OKLab are training losses only.

## Fixed validation verification

- Manifest: `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt`; SHA-256 `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`; 50 validation images.
- Validation provenance: `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/provenance.json`; SHA-256 `3075d566c65cae862d50c38a7c37c5890546bf09e7b9161568e2f23e03fa78db`.
- Validation raw output: `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/seed17_crop_hard16_stride8_top10_global_oklab_g25_outputs.pt`; SHA-256 `3e36c80c181c999a1856350ec94991e50a4161bfe9259672eed23e926c68079f`.

| Metric | Saved value | Expected check | Status |
|---|---:|---:|:---:|
| `global_psnr` | 36.8181499344 | 36.8181499344 | **PASS** |
| `global_ssim` | 0.959187165499 | 0.959187165499 | **PASS** |
| `global_ms_ssim` | 0.98622705698 | 0.98622705698 | **PASS** |
| `full_lpips` | 0.00185862496262 | 0.00185862496262 | **PASS** |
| `top25_local_psnr` | 36.0322517215 | 36.0322517215 | **PASS** |
| `patch_mse_p95` | 0.000272149378434 | 0.000272149378434 | **PASS** |
| `gini` | 0.100017246497 | 0.100017246497 | **PASS** |
| `ciede2000_global` | 3.365011549 | 3.365011549 | **PASS** |
| `ciede2000_top10` | 4.92061436653 | 4.92061436653 | **PASS** |
| `ber30` | 0.1108125 | 0.1108125 | **PASS** |

The g25 values above are validation evidence only. The formal project-test decision is made once below using the pre-frozen acceptance rule; no hyperparameter, checkpoint, or sample selection follows that result.
