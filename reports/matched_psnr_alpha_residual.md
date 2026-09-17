# Matched-PSNR residual-strength comparison

## Result

Validation selected a shared target of **36.87 dB**. The target is the higher α=1 validation PSNR plus 0.05 dB, ceiled to 0.01 dB; both selected scales are inside [0, 1], so no extrapolation was used.

The frozen project-test PSNR mismatch is **0.004014 dB** (PASS for the requested ≤0.05 dB aim).

## Validation calibration

| Method | α=1 PSNR | Selected α | Selected PSNR | Target | Validation gap |
|---|---:|---:|---:|---:|---:|
| MBRS crop-trained global | 36.222526 | 0.927892238 | 36.870000 | 36.87 | 0.00000018 |
| Ours = Hard Local-Tail + global OKLab | 36.818150 | 0.994025141 | 36.870001 | 36.87 | 0.00000111 |

## Frozen project-test comparison

| Method | α | PSNR | BER100 | BER70 | BER50 | BER40 | BER30 | Top-25 local PSNR | P95 patch MSE | P99 patch MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MBRS crop-trained global | 0.927892238 | 36.910772 | 0.0000000 | 0.0000000 | 0.0014375 | 0.0417500 | 0.1131250 | 36.169771 | 2.598485021e-04 | 2.756783463e-04 | 0.00201959 | 3.289427 | 4.809968 | 0.094982 |
| Ours = Hard Local-Tail + global OKLab | 0.994025141 | 36.906757 | 0.0000000 | 0.0000000 | 0.0015000 | 0.0416875 | 0.1131250 | 36.193775 | 2.588815489e-04 | 2.749319419e-04 | 0.00189444 | 3.235305 | 4.745388 | 0.092125 |

## Ours minus MBRS crop-trained global

- `global_psnr`: -4.014355850e-03 (MBRS global is better under the metric direction).
- `ber100`: +0.000000000e+00 (tie).
- `ber70`: +0.000000000e+00 (tie).
- `ber50`: +6.250000000e-05 (MBRS global is better under the metric direction).
- `ber40`: -6.250000000e-05 (Ours is better under the metric direction).
- `ber30`: -1.387778781e-17 (tie).
- `top25_local_psnr`: +2.400344503e-02 (Ours is better under the metric direction).
- `patch_mse_p95`: -9.669532301e-07 (Ours is better under the metric direction).
- `patch_mse_p99`: -7.464044029e-07 (Ours is better under the metric direction).
- `lpips`: -1.251497224e-04 (Ours is better under the metric direction).
- `ciede2000_global`: -5.412223816e-02 (Ours is better under the metric direction).
- `ciede2000_top10`: -6.457977295e-02 (Ours is better under the metric direction).
- `gini`: -2.856616057e-03 (Ours is better under the metric direction).

## Answer to the key question

At matched global PSNR, Ours strictly wins 1/5 requested crop-BER levels, ties 3, and loses 1; it wins 3/3 requested local-tail metrics (Top-25 local PSNR, P95, P99). Gini is treated only as a secondary diagnostic.

Calibration used only the fixed validation manifest. The project-test manifest, messages, and crop masks were opened only after α values were frozen; neither project-test PSNR nor any other project-test metric was used to choose the target or scales.

## Frozen protocol and provenance

- Validation manifest: `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt` (SHA-256 `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`).
- Validation crop masks: `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_crop_masks.pt` (SHA-256 `e592df12eecacd2ea81bf98c55b78f7011bbb6a8eb38898adb712969fa2efe7b`).
- Project-test manifest: `/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt` (SHA-256 `36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339`).
- Project-test crop-40 extension: `/mnt/wmcontent/GLX/icassp/MBRS/reports/controlled_crop35_40_manifest.pt` (SHA-256 `ffa19caecb5d97e717208d3aad7755e67d4b027d79c7c405e83b82bc6a995dde`).
- Evaluator: `/root/workspace/GLX/icassp/MBRS/experiments/run_matched_psnr_comparison.py` (SHA-256 `160be51d5aa54f104ccd785a2ffae0a4c8beac2eca9b32dac2545a4f10493bb6`).
- Both methods use the same fixed messages within each split, the same five-repeat masks, affine normalized-tensor residual scaling `x + α(x̂ − x)`, clipped-RGB quality evaluation, native 32×32 non-overlap patch metrics, mean per-image P95/P99, full-image LPIPS, standard CIEDE2000 with 5×5 stride-1 Top-10 patches, and BER on the unquantized normalized tensor.
