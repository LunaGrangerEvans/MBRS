# External matched-PSNR feasibility and results

## Feasibility

A common target **does exist** without extrapolation. Using the common 512×512 Figure 2 display canvas, the target is **40.72 dB**, defined as the maximum frozen α=1 endpoint plus 0.05 dB, ceiled to 0.01 dB.

| Method | Frozen α=1 PSNR | Selected α | Matched validation PSNR | In [0,1] |
|---|---:|---:|---:|:---:|
| HiDDeN-64 | 30.140571 | 0.295820698 | 40.720000 | PASS |
| MaskWM-D_64 | 38.988422 | 0.819258720 | 40.720000 | PASS |
| MBRS crop-trained global | 40.160196 | 0.937583178 | 40.720000 | PASS |
| Ours = Hard Local-Tail + global OKLab | 40.666397 | 0.993847698 | 40.720000 | PASS |

The strongest frozen endpoint is Ours at 40.666 dB on the common display canvas; it therefore determines the common target. MaskWM-D_64 remains an external native-output reference with a lower α=1 endpoint here. No method required α>1. No model was retrained, and α was not tuned per image.

## Unified validation results

Metrics are recomputed on the same 512×512 RGB canvas for all four methods. This common canvas is required because the frozen external assets have different native resolutions; it is also the Figure 2 display/evaluation space. P95/P99 are means of per-image percentiles over non-overlapping 32×32 patches. CIEDE2000 Top10 uses 5×5 stride-1 patches. External BER is N/A because decoder semantics are not verified as strictly identical to MBRS.

| Method | PSNR | BER100 | BER70 | BER50 | BER40 | BER30 | Top-25 local PSNR | P95 MSE | P99 MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HiDDeN-64 | 40.719999 | N/A | N/A | N/A | N/A | N/A | 38.817830 | 1.648454551e-04 | 2.215514471e-04 | 0.01786180 | 1.533073 | 2.986658 | 0.255025 |
| MaskWM-D_64 | 40.720000 | N/A | N/A | N/A | N/A | N/A | 38.985120 | 1.485163762e-04 | 1.832364385e-04 | 0.06123562 | 2.339700 | 5.043237 | 0.219450 |
| MBRS crop-trained global | 40.720000 | 0.0000000 | 0.0000000 | 0.0159375 | 0.0436250 | 0.1110625 | 39.063971 | 1.595281763e-04 | 2.232467965e-04 | 0.06092356 | 2.122905 | 4.398734 | 0.225634 |
| Ours = Hard Local-Tail + global OKLab | 40.720000 | 0.0000000 | 0.0000000 | 0.0158125 | 0.0438125 | 0.1106875 | 39.053217 | 1.612139965e-04 | 2.306717905e-04 | 0.06100676 | 2.093658 | 4.366610 | 0.229671 |

## BER comparability boundary

- MBRS Global and Ours: BER is reported using the same MBRS decoder, the fixed validation messages, the same five-repeat crop masks, and the raw-bit threshold definition.
- HiDDeN-64 and MaskWM-D_64: BER is **N/A** for the strict comparison. Their external decoder/preprocessing semantics were not verified to be identical to the MBRS raw-bit path, so no external BER claim is made.

## Figure 2

Because the common target is feasible on the common 512×512 canvas, Figure 2 has been regenerated with all four methods at their frozen-checkpoint, validation-calibrated matched-PSNR operating points. The external methods are no longer shown at unmatched native strength; only their residual strength is adjusted globally.
- Figure outputs: `/root/workspace/GLX/icassp/MBRS/paper/figures/figure2_matched_psnr.png`, `/root/workspace/GLX/icassp/MBRS/paper/figures/figure2_matched_psnr.pdf`, `/root/workspace/GLX/icassp/MBRS/paper/figures/figure2_matched_psnr.pptx`.
- Figure selection report: `/root/workspace/GLX/icassp/MBRS/reports/figure2_matched_psnr_selection.md`.
- Figure provenance: `/root/workspace/GLX/icassp/MBRS/reports/figure2_matched_psnr_manifest.json`.

## Provenance and safeguards

- Validation manifest: `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt` (SHA-256 `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`).
- HiDDeN cache: `/mnt/wmcontent/GLX/icassp/MBRS/results/fig2_stress/external_baselines/hidden_64bit_validation.pt`; checkpoint `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt`.
- MaskWM native output directory: `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits/native_512`; released checkpoint `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/checkpoints/maskwm/D_64bits.pth`.
- Internal checkpoints: `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/controlled_seed17_global_continuation/checkpoint_0020.pth` and `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/checkpoint_0020.pth`.
- Raw matched validation outputs: `/mnt/wmcontent/GLX/icassp/MBRS/reports/external_matched_psnr/matched_validation_outputs_512.pt`.
- Project-test data was not opened or used for target/alpha calibration, metric evaluation, or visual selection.
