# External and MBRS metric-consistency audit

Audit date: 2026-09-10. No model was trained. The purpose of this file is to freeze one full-reference image-quality definition before mixing MBRS and external-baseline tables.

## Frozen paper definition

All image-quality rows in `external_baseline_main_table` and the extended-quality reports use:

- fixed 50-image formal manifest, converted from the stored `[-1,1]` tensors to RGB `[0,1]`;
- output and reference clipped to `[0,1]` before pixel metrics;
- PSNR from the dataset-wide mean RGB MSE, `10 log10(1 / MSE)`;
- full-image SSIM from `pytorch_msssim.ssim`, `data_range=1`, Gaussian `win_size=7`, valid Gaussian filtering (no zero padding);
- 3-scale MS-SSIM from `pytorch_msssim.ms_ssim`, `win_size=7`, weights `[0.3, 0.3, 0.4]`;
- full and native-32×32 patch LPIPS using AlexNet LPIPS v0.1; native 32×32 non-overlapping patches are the local perceptual grid;
- local MSE/SSIM/LPIPS and concentration metrics on the native 32×32 non-overlapping grid; Top10 uses `ceil(16×0.10)=2` patches;
- DISTS and GMSD are `N/A` because no verified implementation was available in the environment.

The external crop table is separate from image quality: TrustMark reports ECC-corrected exact-message success/detection, while MBRS reports raw 64-bit BER. Because the payload/ECC and decoder semantics differ, TrustMark is `REFERENCE`, not `STRICT`.

## Why old audit values differed

| Quantity | Legacy controlled/audit path | Frozen extended path | Consequence |
|---|---|---|---|
| Pixel range | direct model tensors in `[-1,1]`; PSNR uses range 2 | RGB tensors clipped to `[0,1]`; PSNR uses range 1 | out-of-range encoded pixels lower the legacy-vs-extended PSNR by about 0.036 dB for Global |
| Full/local SSIM | project `ssim_index`: Gaussian 5×5, zero padding, `max_value=2`, `[-1,1]` | `pytorch_msssim`: valid Gaussian window; full 7×7 and local 5×5, `data_range=1`, `[0,1]` | boundary handling, constants, range, and window differ; absolute SSIM is not interchangeable |
| Patch grid | legacy local audit used its own `ssim_per_sample` on normalized tensors | native 32×32 patches with the frozen `pytorch_msssim` definition | Bottom10 SSIM absolute values change even on identical outputs |
| Aggregation | legacy reports include batch/legacy evaluator aggregation | one dataset-wide mean MSE for PSNR; arithmetic mean of per-image SSIM/MS-SSIM/LPIPS | tiny residual discrepancies can remain if a legacy JSON is copied directly |

## Direct same-checkpoint comparison

| Method | Legacy PSNR | Frozen PSNR | Δ | Legacy Bottom10 SSIM | Frozen Bottom10 SSIM | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Global continuation | 36.227406 | 36.263172 | +0.035767 | 0.917602 | 0.902273 | -0.015329 |
| Hard Patch16 / stride8 / Top10 / MSE selector | 36.407159 | 36.442300 | +0.035142 | 0.920070 | 0.905504 | -0.014566 |

The changes are implementation-definition changes, not evidence that either model changed. The PSNR difference is expected because roughly 0.77% of encoded tensor values are outside `[-1,1]` before clipping. The larger Bottom10 SSIM shift is primarily the SSIM implementation/range/padding/grid difference.

## Formal-table action

- New external baseline tables use only the frozen definition above.
- The old `current_controlled_master_table` remains the source of the controlled-training delta/BER lineage and is explicitly legacy-protocol data; its old absolute SSIM/PSNR values must not be placed beside the frozen external rows without relabeling.
- New extended reports contain the frozen absolute image-quality values for the four controlled checkpoints. No old value is silently relabeled as frozen.

## Verification

- SSIM identity sanity: `1.00000000`.
- 3-scale MS-SSIM identity sanity: `1.00000000`.
- Noise sanity: SSIM `0.68048477`, MS-SSIM `0.69430465`; both below identity.
- Manifest: fixed 50 images; SHA-256 `36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339`.
