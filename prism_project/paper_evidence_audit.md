# Paper Evidence Audit

This audit uses `paper_bundle/` as the only authoritative evidence source. The bundled reports may contain paths and hashes for provenance, but no file outside `paper_bundle/` was used to define, select, or interpret the paper evidence. The paper itself has not been edited.

Primary reading set: [`paper_bundle/README.md`](paper_bundle/README.md), [`notes/paper_story.md`](paper_bundle/notes/paper_story.md), [`notes/claims_and_limits.md`](paper_bundle/notes/claims_and_limits.md), the bundled evidence reports, the bundled table fragments, and the bundled Figure 1/Figure 2 renders.

## 1. Final method definition

The reader-facing method is:

`Ours = Hard Local-Tail + global OKLab regularization`.

The frozen run is `seed17_crop_hard16_stride8_top10_global_oklab_g25`. Its objective is:

```text
L = 10 L_msg + 0.5 L_RGB + 0.5 L_tail + 0.059149764 L_OKLab
```

The terms are defined as follows:

- `L_msg` is the message loss with weight `10`.
- `L_RGB` is global RGB reconstruction loss with weight `0.5`.
- `L_tail` is the mean of the hard Top-10% highest-error patch scores with weight `0.5`.
- `L_OKLab` is mean per-pixel OKLab distance in linear-sRGB OKLab with `lambda_OK = 0.059149764`.

Hard Local-Tail uses patch size `16`, stride `8`, `225` overlapping candidates, and `23` selected patches (`ceil(225 × 0.1)`). It is a training loss, as is OKLab. No additional inference module is introduced; inference retains the adopted MBRS encoder → crop channel → decoder path.

## 2. Internal baseline definition

The controlled internal baseline is **MBRS crop-trained global**. It is the global-RGB reconstruction branch of the adopted MBRS backbone, with no local-tail term. It is the comparison row named `MBRS crop-trained global` in both bundled table fragments.

The ablation chain is:

```text
MBRS crop-trained global → Hard Local-Tail → Ours = Hard Local-Tail + global OKLab
```

The baseline and Ours use the same MBRS inference architecture. External HiDDeN-64 and MaskWM-D_64 are not internal baselines; they are contextual external references for Figure 2 only.

## 3. Controlled training protocol

The frozen Ours protocol is:

- input: `128x128` RGB;
- payload: `64` bits;
- crop augmentation: `RandomCrop(0.3, 1.0)`;
- seed: `17`;
- source: the same epoch-100 crop-trained Global checkpoint;
- state restoration: model state including BatchNorm buffers and Adam optimizer state;
- continuation: `20` epochs at `lr = 1e-4`;
- batch size: `16`, workers: `0`;
- one GPU with deterministic algorithm/cudNN settings;
- no warm-up, full-scratch retraining, DataParallel, or hyperparameter search.

The controlled-continuation report also records a lineage isolation study in which all branches start from the same seed17 epoch-100 source and differ only in image-loss objective. That branch table contains `Global continuation`, `Hard P16-T25-L50` (hard Top-25%), and `Soft P16-T0.5-L50` (detached normalized softmax, temperature `0.5`), each for 20 epochs at `1e-4`. This is protocol context and must not be relabeled as the frozen final Ours configuration: the reader-facing final method is P16/S8, hard Top-10%, 23 patches, plus global OKLab.

## 4. Formal project-test results

These are the natural frozen source-resolution project-test results on the fixed 50-image project-test manifest. No project-test result was used to retune or select the frozen configuration. CIEDE2000 and Gini are diagnostics/evaluation metrics; Gini is not a primary objective or acceptance gate.

| Method | PSNR | SSIM | LPIPS | Top-25 local PSNR | P95 MSE | P99 MSE | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MBRS crop-trained global | 36.263172 | 0.950759 | 0.00234960 | 35.522213 | 3.016531971e-04 | 3.199585360e-04 | 3.535857 | 5.159104 | 0.095007 | 0.1131250 |
| Hard Local-Tail | 36.442300 | 0.952541 | 0.00221518 | 35.754376 | 2.844551472e-04 | 3.010726967e-04 | 3.484790 | 5.075666 | 0.086897 | 0.1129375 |
| Ours | 36.854909 | 0.957018 | 0.00191711 | 36.141895 | 2.619925590e-04 | 2.782322409e-04 | 3.254091 | 4.772229 | 0.092130 | 0.1131250 |

The bundled formal report classifies the frozen final method as **FINAL METHOD ACCEPTED** under a pre-frozen rule requiring local PSNR at least as high as Hard Local-Tail, P95 no higher, global PSNR no lower or without meaningful degradation, LPIPS no higher or without meaningful degradation, and absolute BER30 degradation no more than `0.002`. All listed checks pass. The formal report explicitly keeps the Ours Gini increase visible and treats Gini as diagnostic only.

## 5. Internal matched-PSNR results

This is a separate source-resolution project-test analysis. Validation was used to freeze the residual-strength target/scales; the project-test data was evaluated only afterward. The validation target for this analysis is `36.87 dB`.

| Method | α | PSNR | BER100 | BER70 | BER50 | BER40 | BER30 | Top-25 local PSNR | P95 patch MSE | P99 patch MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MBRS crop-trained global | 0.927892238 | 36.910772 | 0.0000000 | 0.0000000 | 0.0014375 | 0.0417500 | 0.1131250 | 36.169771 | 2.598485021e-04 | 2.756783463e-04 | 0.00201959 | 3.289427 | 4.809968 | 0.094982 |
| Ours = Hard Local-Tail + global OKLab | 0.994025141 | 36.906757 | 0.0000000 | 0.0000000 | 0.0015000 | 0.0416875 | 0.1131250 | 36.193775 | 2.588815489e-04 | 2.749319419e-04 | 0.00189444 | 3.235305 | 4.745388 | 0.092125 |

The source-resolution PSNR mismatch is `0.004014 dB` in magnitude, and BER30 is tied at `0.1131250`. Ours retains the local-tail and perceptual/color improvements in this matched analysis, while its global PSNR is lower by approximately `0.004014 dB`. Crop-BER outcomes are mixed across the five reported crop levels; therefore no broad crop-BER superiority claim is supported.

## 6. External matched-PSNR Figure 2 protocol

Figure 2 is an external qualitative/display-space comparison, not a formal project-test benchmark.

- Scope: fixed 50-image validation manifest only.
- Canvas: common `512x512` display canvas.
- Target: mean display PSNR `40.72 dB`.
- Calibration: one global residual-strength scale `α` per method; no per-image tuning, retraining, or project-test data.
- Residual scaling: `x_α = x + α(x_hat - x)` on the common RGB canvas, followed by the documented display path.
- Methods: HiDDeN-64, MaskWM-D_64, MBRS crop-trained global, and Ours; Original is the host/reference column.

The bundled calibration values are:

| Method | Frozen α=1 endpoint | Selected α | Matched validation PSNR |
|---|---:|---:|---:|
| HiDDeN-64 | 30.140571 | 0.295820698 | 40.720000 |
| MaskWM-D_64 | 38.988422 | 0.819258720 | 40.720000 |
| MBRS crop-trained global | 40.160196 | 0.937583178 | 40.720000 |
| Ours = Hard Local-Tail + global OKLab | 40.666397 | 0.993847698 | 40.720000 |

The common target is feasible without extrapolation. External BER is not strictly comparable because external decoder/preprocessing/raw-bit semantics were not verified to be identical to the MBRS path. It must not be ranked or used for an external BER superiority claim. The bundled external validation table is contextual evidence only.

## 7. Figure 1 definition

The bundled `figures/figure1_framework.{pdf,png}` defines the frozen final method as a shared MBRS encoder/decoder forward path with a training-time-only crop channel and three training-loss branches:

1. Global RGB average-distortion control.
2. Hard Local-Tail upper-tail local-distortion control using P16/S8, 225 candidates, and 23 hard Top-10% patches.
3. Global linear-sRGB OKLab color-aware fidelity control with `lambda_OK = 0.059149764`.

The displayed final objective is written with symbolic weights, while the frozen paper objective is the exact numeric expression recorded in Section 1. The figure states that local-tail and OKLab are training losses only, that the crop channel is training-time only, and that no additional inference module is introduced. It contains no validation-only OKLab interpretation and does not present Gini as a key effect.

## 8. Figure 2 definition

The bundled `figures/figure2_matched_psnr.{pdf,png}` is the five-column all-method qualitative comparison:

```text
Original | HiDDeN-64 | MaskWM-D_64 | MBRS crop-trained global | Ours = Hard Local-Tail + global OKLab
```

It contains two fixed validation samples, each with a full image, zoom, and residual ×10 row. The shared ROIs are:

- Sample 1: ID `0846`, validation index `45`, ROI `(48,28,32,32)`.
- Sample 2: ID `0821`, validation index `20`, ROI `(84,32,32,32)`.

Red ROI boxes appear only on full-image rows. Zoom and residual tiles have no borders. Residual maps use one shared per-sample range across all five columns and are mean absolute displayed-RGB residuals amplified ×10. The authoritative bundled caption identifies the panel as a matched-PSNR qualitative comparison at mean display PSNR `40.72 dB` and says that external methods use frozen checkpoints with only global residual-strength adjustment.

## 9. Supported claims

The bundle supports these claims:

- Ours improves absolute local-tail fidelity under the controlled internal protocol.
- At the natural frozen operating point versus MBRS crop-trained global, Ours improves global PSNR by `+0.592 dB` and Top-25 local PSNR by `+0.620 dB` after rounding, lowers P95 patch MSE by about `13.1%`, lowers LPIPS by about `18.4%`, lowers CIEDE2000, and retains identical BER30 `0.113125`.
- Crop robustness is approximately preserved; the evidence does not establish significantly stronger crop robustness.
- Internal matched-PSNR analysis shows that the improvements are not explained only by higher global PSNR: the source-resolution PSNR mismatch is only `0.004014 dB`, BER30 is tied, and local/perceptual metrics remain favorable for Ours.
- External methods are qualitative/contextual references at matched display PSNR on the common 512x512 canvas.
- Hard Local-Tail and OKLab are training-only losses and inference architecture is unchanged.

## 10. Prohibited or unsupported claims

Do not claim:

- state of the art;
- universal superiority over HiDDeN or MaskWM;
- significantly stronger crop robustness;
- human perceptual superiority;
- uniformly distributed residuals;
- Gini as the primary objective or a decisive primary effect;
- external BER superiority or an external BER ranking;
- validation results as formal project-test results;
- that the external `40.72 dB` display result is a `128x128` project-test PSNR;
- that historical `w_msg=80` stress experiments, old Ours-base naming, old six-column Figure 2 variants, or outdated tables define the paper method.

The validation-only OKLab values in the bundled frozen-method report are provenance/context for the frozen candidate and must not replace the formal project-test values in Section 4. The formal paper identity is Ours = Hard Local-Tail + global OKLab, not an old Ours-base or validation-only variant.

## 11. Exact metric-space separation

| Space | Definition and purpose | Authoritative values/handling |
|---|---|---|
| **A. Natural frozen 128x128 project-test** | Fixed 50-image project-test evaluation of the frozen method and internal ablation chain; formal acceptance evidence. | Global PSNR `36.263172`; Hard Local-Tail `36.442300`; Ours `36.854909`. Ours local PSNR `36.141895`, P95 `2.619925590e-04`, LPIPS `0.00191711`, CIEDE2000 `3.254091`, BER30 `0.1131250`. |
| **B. Internal matched-PSNR 128x128** | Source-resolution project-test comparison after validation-only residual-strength calibration to target `36.87 dB`. | Global PSNR `36.910772`; Ours `36.906757`; mismatch `0.004014 dB`. Ours local PSNR `36.193775`, P95 `2.588815489e-04`, LPIPS `0.00189444`, CIEDE2000 `3.235305`, BER30 `0.1131250`. |
| **C. External/Figure-2 matched display-PSNR** | Fixed 50-image validation-only qualitative comparison on a common `512x512` display canvas; four methods calibrated to mean display PSNR `40.72 dB`. | External HiDDeN-64 and MaskWM-D_64 are contextual references. No project-test data is used for target or selection. External BER is N/A/not strictly comparable and is not ranked. |

Never substitute C for A or B, and never report C's `40.72 dB` as a source-resolution project-test result. Likewise, A's natural operating-point values and B's residual-scaled values must not be blended into a single table or claim.
