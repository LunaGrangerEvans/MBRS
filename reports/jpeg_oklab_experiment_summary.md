# JPEG + OKLab-tail experiment summary

> Subsequent implementation audit: the legacy color transform in `experiments/color_losses.py` applied an XYZ→LMS matrix directly to linear RGB. Consequently, these checkpoints are **not a verified standard-OKLab experiment**. Their measured fidelity/color/BER values describe the saved models, but their outcome cannot establish that standard OKLab supervision fails. See [crop_global_oklab_protocol.md](crop_global_oklab_protocol.md) for the separately tested correction; historical checkpoints and implementation are preserved.

Two seed17 controlled continuations were trained for 20 epochs from the same crop-trained Global epoch100 checkpoint. No new seed, full-scratch run, Q30 training, or parameter grid was used.

Training attack: `Combined([Jpeg(50), Identity()])`, using the existing differentiable/simulated JPEG implementation and the existing Identity layer. Evaluation uses clean Identity, real PIL JPEG quality50, and real PIL JPEG quality30.

## Clean quality and color tail

| Method | PSNR | SSIM | MS-SSIM | LPIPS | Global CIEDE2000 | Patch mean CIEDE2000 | P95 CIEDE2000 | Top10 CIEDE2000 | Max CIEDE2000 | Mean Δa | Mean Δb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JPEG Global | 35.460434 | 0.941263 | 0.979203 | 0.00278555 | 4.190982 | 4.240510 | 5.970382 | 6.097279 | 8.402176 | -0.332036 | 0.315069 |
| JPEG OKLab-tail | 38.116315 | 0.969421 | 0.989523 | 0.00261648 | 2.489505 | 2.502145 | 3.467641 | 3.539543 | 4.859564 | 0.163380 | -0.444308 |

## Robustness

| Method | Identity Acc / BER | RealJPEG50 Acc / BER | RealJPEG30 Acc / BER |
|---|---:|---:|---:|
| JPEG Global | 1.000000 / 0.000000 | 0.720625 / 0.279375 | 0.677500 / 0.322500 |
| JPEG OKLab-tail | 0.999687 / 0.000313 | 0.550625 / 0.449375 | 0.550313 / 0.449687 |

## Per-image color-tail improvement

- patch_ciede2000_mean: 100.0% of formal images improved under OKLab-tail.
- global_ciede2000_mean: 100.0% of formal images improved under OKLab-tail.
- max_patch_ciede2000: 100.0% of formal images improved under OKLab-tail.
- top10_patch_ciede2000: 100.0% of formal images improved under OKLab-tail.
- patch_ciede2000_p95: 100.0% of formal images improved under OKLab-tail.

## Decision

**Result: TRADE-OFF / NOT ACCEPTED as the next paper main method.**

- Clean quality improved strongly: PSNR `+2.655880 dB`, SSIM `+0.028159`, MS-SSIM `+0.010321`, and full LPIPS `-0.00016907`.
- Color-tail quality improved on **100% of the 50 formal images**: global CIEDE2000 `-1.701478`, patch P95 `-2.502741`, Top10 patch CIEDE2000 `-2.557736`, and max patch CIEDE2000 `-3.542612`.
- Identity accuracy was essentially preserved: `1.000000 → 0.999687`.
- JPEG50 accuracy deteriorated from `0.720625` to `0.550625` and BER increased from `0.279375` to `0.449375`.
- JPEG30 accuracy deteriorated from `0.677500` to `0.550313` and BER increased from `0.322500` to `0.449687`.

The OKLab-tail formulation therefore clearly controls the measured color-distortion tail, but it does not satisfy the required JPEG robustness criterion. The direction should be **stopped for this deadline**; do not launch alpha, top-ratio, patch-size, color-space, or robustness-weight searches. The result is suitable as a negative/trade-off ablation, not as the paper's primary method.

The signed chroma shift also changes direction: JPEG Global has mean `Δa=-0.332036, Δb=+0.315069`, while OKLab-tail has `Δa=+0.163380, Δb=-0.444308`. This indicates the loss reduces magnitude of local color error but does not preserve the same color-bias direction.

Qualitative figures and CIEDE2000 distributions are under `visualizations/jpeg_oklab/`. Machine-readable per-image data are in [jpeg_oklab_per_image.csv](jpeg_oklab_per_image.csv), the summary table is [jpeg_oklab_main_table.csv](jpeg_oklab_main_table.csv), and robustness details are [jpeg_oklab_robustness.csv](jpeg_oklab_robustness.csv).
