# Figure 2 strong visual search

This is an exploratory, validation-only stress search for a scientifically fair qualitative Figure 2 for MBRS.
Formal Table I results and formal checkpoints are not modified. Exploratory model checkpoints are isolated under `/mnt/wmcontent/GLX/icassp/MBRS/checkpoints/fig2_stress/`; result tensors and diagnostics are under `/mnt/wmcontent/GLX/icassp/MBRS/results/fig2_stress/`.

## Fixed protocol

- Seed: `17`; fixed validation manifest: `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt`; 50 images, 64-bit messages.
- Source: `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth`; every screen run starts from this same crop-trained checkpoint and loads its compatible Adam state plus model BatchNorm buffers through `train_local_patch.py`.
- Retraining was used only for these exploratory continuations: 20 five-epoch screen arms, four tail/top-k extension arms, and eight independently restarted 20-epoch promoted arms. Formal checkpoints and Table I were not retrained or overwritten.
- No project-test data was opened by this pipeline. No attack realization was used to select the main figure.
- Native RGB displays use the saved output tensors. Residual amplification and error maps are separate diagnostics, labelled as visualization-only.

## Attempted strong-embedding screen

The screen used 5-epoch continuations for matched Ours-base/Ours pairs. Ours-base is `w_msg L_msg + λ_img L_global`; Ours is `w_msg L_msg + λ_img(0.5 L_global + 0.5 L_tail)` with P16/S8/Top10. A run is early-rejected at fixed validation if PSNR < 24 dB, BER30 > 0.30, or any required metric is non-finite.

| Method | condition | w_msg | λ_img | PSNR | local PSNR | P95 | Gini | BER30 | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Ours-base | reference | 10 | 1 | 35.314 | 34.522 | 3.785e-04 | 0.0956 | 0.1148 | pass |
| Ours | reference | 10 | 1 | 35.381 | 34.612 | 3.696e-04 | 0.0920 | 0.1153 | pass |
| Ours-base | wmsg20 | 20 | 1 | 34.901 | 34.123 | 4.130e-04 | 0.0927 | 0.1154 | pass |
| Ours | wmsg20 | 20 | 1 | 34.971 | 34.220 | 4.026e-04 | 0.0889 | 0.1154 | pass |
| Ours-base | wmsg40 | 40 | 1 | 34.238 | 33.484 | 4.758e-04 | 0.0897 | 0.1147 | pass |
| Ours | wmsg40 | 40 | 1 | 34.333 | 33.607 | 4.609e-04 | 0.0856 | 0.1144 | pass |
| Ours-base | wmsg80 | 80 | 1 | 32.591 | 31.559 | 7.783e-04 | 0.1256 | 0.1143 | pass |
| Ours | wmsg80 | 80 | 1 | 33.194 | 32.479 | 5.940e-04 | 0.0834 | 0.1153 | pass |
| Ours-base | img05 | 10 | 0.5 | 34.915 | 34.134 | 4.122e-04 | 0.0936 | 0.1152 | pass |
| Ours | img05 | 10 | 0.5 | 34.968 | 34.205 | 4.044e-04 | 0.0911 | 0.1158 | pass |
| Ours-base | img025 | 10 | 0.25 | 34.698 | 33.924 | 4.313e-04 | 0.0923 | 0.1149 | pass |
| Ours | img025 | 10 | 0.25 | 34.728 | 33.967 | 4.265e-04 | 0.0906 | 0.1153 | pass |
| Ours-base | wmsg20_img05 | 20 | 0.5 | 34.534 | 33.766 | 4.466e-04 | 0.0912 | 0.1151 | pass |
| Ours | wmsg20_img05 | 20 | 0.5 | 34.576 | 33.830 | 4.391e-04 | 0.0883 | 0.1148 | pass |
| Ours-base | wmsg20_img025 | 20 | 0.25 | 34.282 | 33.526 | 4.711e-04 | 0.0895 | 0.1150 | pass |
| Ours | wmsg20_img025 | 20 | 0.25 | 34.298 | 33.557 | 4.667e-04 | 0.0874 | 0.1150 | pass |
| Ours-base | wmsg40_img05 | 40 | 0.5 | 33.798 | 33.049 | 5.250e-04 | 0.0891 | 0.1151 | pass |
| Ours | wmsg40_img05 | 40 | 0.5 | 33.841 | 33.123 | 5.134e-04 | 0.0847 | 0.1151 | pass |
| Ours-base | wmsg40_img025 | 40 | 0.25 | 33.541 | 32.798 | 5.561e-04 | 0.0881 | 0.1152 | pass |
| Ours | wmsg40_img025 | 40 | 0.25 | 33.553 | 32.832 | 5.495e-04 | 0.0851 | 0.1153 | pass |
| Ours-base | promoted_wmsg80 | 80 | 1 | 31.486 | 30.781 | 8.707e-04 | 0.0814 | 0.1099 | pass |
| Ours | promoted_wmsg80 | 80 | 1 | 32.353 | 31.610 | 7.261e-04 | 0.0871 | 0.1106 | pass |
| Ours-base | promoted_wmsg40 | 40 | 1 | 33.786 | 33.041 | 5.227e-04 | 0.0867 | 0.1114 | pass |
| Ours | promoted_wmsg40 | 40 | 1 | 33.426 | 32.810 | 5.456e-04 | 0.0707 | 0.1119 | pass |
| Ours-base | promoted_wmsg20 | 20 | 1 | 35.189 | 34.412 | 3.875e-04 | 0.0952 | 0.1108 | pass |
| Ours | promoted_wmsg20 | 20 | 1 | 35.638 | 34.906 | 3.449e-04 | 0.0892 | 0.1111 | pass |
| Ours-base | promoted_reference | 10 | 1 | 36.223 | 35.426 | 3.108e-04 | 0.1009 | 0.1110 | pass |
| Ours | promoted_reference | 10 | 1 | 36.411 | 35.653 | 2.946e-04 | 0.0946 | 0.1110 | pass |
| Ours-tail75 | wmsg80 | 80 | 1 | 33.295 | 32.574 | 5.816e-04 | 0.0847 | 0.1142 | pass |
| Ours-tail90 | wmsg80 | 80 | 1 | 33.329 | 32.637 | 5.711e-04 | 0.0796 | 0.1138 | pass |
| Ours-top5 | wmsg80 | 80 | 1 | 33.253 | 32.543 | 5.847e-04 | 0.0828 | 0.1146 | pass |
| Ours-top20 | wmsg80 | 80 | 1 | 33.270 | 32.537 | 5.879e-04 | 0.0868 | 0.1151 | pass |

## Convergence audit

Promoted checkpoints are re-evaluated at epochs 5/10/15/20 using one frozen validation image/message/crop-mask realization. `plateau_observed` requires both global- and local-PSNR spans across epochs 10/15/20 to be at most 0.15 dB. This is an operational plateau check, not proof of global optimizer convergence.

| Reader-facing method | condition | epochs | global span (dB) | local span (dB) | verdict |
|---|---|---|---:|---:|---|
| Global Reconstruction | promoted_wmsg80 | 5,10,15,20 | 2.449 | 2.752 | not_demonstrated |
| Hard Local-Tail | promoted_wmsg80 | 5,10,15,20 | 1.582 | 1.568 | not_demonstrated |
| Global Reconstruction | promoted_wmsg40 | 5,10,15,20 | 1.238 | 1.487 | not_demonstrated |
| Hard Local-Tail | promoted_wmsg40 | 5,10,15,20 | 1.379 | 1.338 | not_demonstrated |
| Global Reconstruction | promoted_wmsg20 | 5,10,15,20 | 0.790 | 0.751 | not_demonstrated |
| Hard Local-Tail | promoted_wmsg20 | 5,10,15,20 | 1.095 | 1.249 | not_demonstrated |
| Global Reconstruction | promoted_reference | 5,10,15,20 | 1.099 | 1.320 | not_demonstrated |
| Hard Local-Tail | promoted_reference | 5,10,15,20 | 0.618 | 0.632 | not_demonstrated |

Uncompleted runs are absent from the table and remain explicitly visible as missing checkpoint directories in the screen manifest; they are not silently treated as successes.
No valid screen or extension arm crossed the predeclared PSNR<24 dB or BER30>0.30 early-rejection thresholds. One interrupted wrong-initialization attempt was isolated under `fig2_stress_invalid_init_20260914/` and is excluded from all results.

## Matched operating points

Best matched-BER pair: `promoted_wmsg80` / `promoted_wmsg80` with BER30 `0.1099375` vs `0.110625`, local PSNR delta `0.8297447544035812` dB and P95 delta `-0.00014459720317972825`. The intended caption terminology is **Matched crop-robustness operating point**.
Best strong-embedding pair: `promoted_wmsg80` / `promoted_wmsg80`; the promoted w_msg=80, λ_img=1 pair is the preferred native stress case because it preserves BER30 within the requested tolerance while making local/P95 differences most observable in the shared ROI zoom.
Convergence caveat: none of the eight promoted runs met the operational 0.15 dB plateau criterion at epochs 10/15/20. The selected w_msg=80 pair is therefore an exploratory epoch-20 operating point, not a converged replacement model.
Negative result retained: at promoted w_msg=40 epoch20, Hard Local-Tail was 0.232 dB worse in local PSNR and had higher P95 than Global Reconstruction despite matched BER30. The local-tail advantage is not monotonic across all operating points.

All matched-BER candidates are in [fig2_matched_ber_pairs.csv](fig2_matched_ber_pairs.csv). Matching permits cross-condition operating points, while preferring same-condition pairs in deterministic tie-breaking. This makes the robustness comparison explicit rather than silently comparing one arbitrary checkpoint pair.

No Ours-base/Ours pair fell within the requested ±0.15 dB tolerance in the 30/32/34/36 dB target bins; this negative result is retained and no matched-PSNR figure is fabricated.
All matched-global-PSNR bins, including empty bins, are recorded in [fig2_matched_psnr_pairs.csv](fig2_matched_psnr_pairs.csv).

Tail/top-k extension screen: Ours-tail75, Ours-tail90, Ours-top20, Ours-top5. Tail75, Tail90, Top5, and Top20 were all run under w_msg=80, λ_img=1 from the same source; their fixed-validation metrics are in the main CSV and are not formal Ours. Patch-scale exploration was not run because the w_msg=80 native comparison already produced a clear local-tail stress signal, satisfying the conditional only-if-needed rule.

## Sample and ROI selection

Samples were ranked over all 50 validation images by measured local PSNR improvement, P95 improvement, maximum positive patch error difference, positive patch area, and Top10 residual reduction. The top-20 contact sheet is [fig2_top20_contact_sheet.png](../paper/figures/fig2_top20_contact_sheet.png). Two samples were then selected by the same ranking with a deterministic RGB/texture diversity proxy; the proxy is not claimed to be a semantic classifier.

| Figure sample | image id | validation index | ROI | Δlocal PSNR | ΔP95 | max positive D_patch | positive area |
|---|---|---:|---|---:|---:|---:|---:|
| 1 | 0822 | 21 | (68, 84, 32×32) | +1.448 | +2.832e-04 | 3.409e-04 | 1.000 |
| 2 | 0823 | 22 | (84, 40, 40×40) | +1.372 | +2.420e-04 | 3.028e-04 | 1.000 |

ROIs were searched over shared 32/40/48/64-pixel candidates on a stride-4 grid with a four-pixel border margin. One ROI is shared by Original, Ours-base, Ours, and all diagnostic representations for that sample. ROI coordinates and all 50 ranked samples are in [fig2_sample_ranking.csv](fig2_sample_ranking.csv).

## Figure candidates and decision

| Candidate | Visual difference /5 | Scientific fairness /5 | Direct local-tail relevance /5 | Reproducibility /5 | ICASSP suitability /5 | Decision |
|---|---:|---:|---:|---:|---:|---|
| A. Native strong-embedding matched-BER | 4 | 5 | 5 | 5 | 5 | Preferred when native difference is inspectable; final visual |
| B. Matched-global-PSNR | 3 | 5 | 5 | 5 | 5 | Secondary operating-curve evidence |
| C. Native + shared error-difference diagnostic | 5 | 5 | 5 | 5 | 4 | Supplementary explanation; D is not a native output |
| D. Equal residual-strength visualization | 5 | 3 | 3 | 5 | 3 | Stress diagnostic only; never a native quality claim |
| E. Color-tail extension | 2 | 4 | 2 | 4 | 3 | Not promoted unless an independently useful color difference is present |

The final Figure 2 uses five reader-facing columns: `Original`, `HiDDeN-64 (retrained external)`, `MaskWM-D (official external)`, `Global Reconstruction (internal baseline)`, and `Hard Local-Tail (proposed)`. Every column uses the same selected validation samples, shared ROI, and 512×512 display canvas.

Internal run identifiers map to reader-facing names: `Ours-base` → `Global Reconstruction (internal baseline)` and `Ours` → `Hard Local-Tail (proposed)`. External methods are contextual references, not strict matched comparisons; the under-converged HiDDeN retraining is not primary evidence.

The final native figure shows full and shared-ROI zoom rows. The separate diagnostic suite shows `|residual| ×10/20/30/50`, log residuals for β=100/500/1000, and shared-scale pixel/patch error differences; these are visualization-only and never model outputs or selection metrics.

## Residual and attack stress

Equal residual amplification was evaluated for every discoverable 128×128/64-bit existing run checkpoint at α = 1, 1.5, 2, 2.5, 3, 4, 5. Residual-alpha coverage is 490 rows over 70 existing models. Clipping flags by alpha: α=1: 0/70 models >1%, 0/70 >5%; α=1.5: 36/70 models >1%, 0/70 >5%; α=2: 69/70 models >1%, 2/70 >5%; α=2.5: 70/70 models >1%, 3/70 >5%; α=3: 70/70 models >1%, 5/70 >5%; α=4: 70/70 models >1%, 15/70 >5%; α=5: 70/70 models >1%, 38/70 >5%. Alpha > 1 is diagnostic stress and is not used as native evidence.

Crop 50/40/30 (five fixed mask repeats), Gaussian σ = 0.005/0.01/0.02, and JPEG quality 90/70/50 are recorded in [fig2_attack_stress_summary.csv](fig2_attack_stress_summary.csv) for the selected matched pair only. They are not used for Figure 2 selection; attack corruption is never presented as watermark-quality improvement.

## Artifacts

- [fig2_strong_embedding_results.csv](fig2_strong_embedding_results.csv)
- [fig2_matched_ber_pairs.csv](fig2_matched_ber_pairs.csv)
- [fig2_matched_psnr_pairs.csv](fig2_matched_psnr_pairs.csv)
- [fig2_residual_alpha_sweep.csv](fig2_residual_alpha_sweep.csv)
- [fig2_final_visual.png](../paper/figures/fig2_final_visual.png), [PDF](../paper/figures/fig2_final_visual.pdf), [editable PPTX](../paper/figures/fig2_final_visual.pptx)

## Formal versus exploratory status

Every row and figure in this report is exploratory/qualitative stress evidence. It must not replace formal Table I values, redefine Ours, or be described as a new formal method. No formal checkpoint was overwritten; no selective sharpening, blur, saturation, per-method normalization, attack realization, or per-method ROI was used.
