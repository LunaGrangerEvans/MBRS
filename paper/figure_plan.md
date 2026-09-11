# Final paper figure plan (archival evidence freeze)

> The current four-page main-body selection is the compact two-figure/two-table set in [figure_table_plan_4page.md](figure_table_plan_4page.md) and [four_page_main_layout.md](four_page_main_layout.md). This file preserves the broader five-figure/supplementary evidence freeze for provenance and is not the active main-body layout.

Final asset audit: 2026-09-10. Scope: MBRS project-owned evidence only. **Figs. 1–5 and S1 are ready for drafting, pending print-size layout checks.** Selected numerical figures were refreshed using [render_paper_freeze.py](/root/workspace/GLX/icassp/MBRS/experiments/render_paper_freeze.py), following the unified checkpoint reevaluation. The figure audit independently checked the outputs and captions. No training was performed. JPEG+OKLab is excluded.

The main comparison is Global continuation versus Hard Patch16 / stride8 / Top10% raw-MSE selection, global/local weights 0.5/0.5. Both are seed17 continuation epoch20. The eight frozen configurations remain in the main/ablation tables; primary figures emphasize Global and Hard Top10. TrustMark is supplemental REFERENCE evidence.

## 1. Final selected assets and readiness

These are actual existing assets, not proposed filenames. Prefer the PDFs for typesetting and PNGs for preview. The final figures use the following exact panel arrangements; superseded layouts are not instructions for further figure generation.

| Slot | Final content | Selected files | Status |
|---|---|---|---|
| Fig. 1 | Method: encoder–mask–decoder, global/local losses, 23/225 training selection, separate evaluation grid | [PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig01_method.png) · [PDF](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig01_method.pdf) | Ready for drafting; print-size check pending |
| Fig. 2 | Predetermined 7/20/42 actual-RGB examples, each a 3×3 block | [07 PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/actual_rgb_test_07.png) · [20 PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/actual_rgb_test_20.png) · [42 PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/actual_rgb_test_42.png) · [three-page PDF](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/artifact_observation_main.pdf) | Ready for drafting; arrange the existing three sample blocks at print size |
| Fig. 3 | Three panels: globalPSNR / localPSNR / BER at 30,40,50,70,100% | [PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig03_quality_robustness.png) · [PDF](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig03_quality_robustness.pdf) | Ready for drafting; print-size check pending |
| Fig. 4 | Two panels: mean per-image Lorenz curve / paired Gini | [PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig04_concentration.png) · [PDF](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig04_concentration.pdf) | Ready for drafting; clipped native32 values verified |
| Fig. 5 | Two panels: signed P95 MSE / Top10 LPIPS deltas for all 50 image indices | [PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig05_pixel_perceptual.png) · [PDF](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig05_pixel_perceptual.pdf) | Ready for drafting; 96%/62% and mixed LPIPS outcome verified |
| Fig. S1 | Two panels: globalPSNR / full LPIPS versus pre-ECC accuracy | [PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/figS1_external_reference.png) · [PDF](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/figS1_external_reference.pdf) | Ready for drafting; corrected TrustMark PSNR verified; REFERENCE only |

The [render provenance JSON](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/provenance.json) records the two frozen tables, per-image CSV, renderer, and all ten refreshed PNG/PDF hashes. All 14 hashes were independently checked and matched. All five refreshed PNGs and the three original RGB example PNGs were visually inspected during this task. The PDFs were verified as existing hashed artifacts; final paper-sized PDF placement is not yet inspected.

This numbering supersedes the historical [figure/table plan](/root/workspace/GLX/icassp/MBRS/reports/paper_figure_table_plan.md). Old metric plots are excluded, not upgraded to publication-ready status.

## 2. Provenance and the metric boundary

### Authoritative paper-freeze-v1 handoff

The current freeze, produced under the requested protocol, supersedes earlier report tables for all final numeric figures. Use [main_table.csv](/root/workspace/GLX/icassp/MBRS/paper/main_table.csv), [external_reference_table.csv](/root/workspace/GLX/icassp/MBRS/paper/external_reference_table.csv), [paired_diagnostics.json](/root/workspace/GLX/icassp/MBRS/paper/paired_diagnostics.json), [frozen per-image CSV](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv), and [evidence manifest](/root/workspace/GLX/icassp/MBRS/paper/evidence_manifest.json). The exact producer is [freeze_paper_evidence.py](/root/workspace/GLX/icassp/MBRS/experiments/freeze_paper_evidence.py). These files were inspected after the freeze update. The figure refresh is complete; final selected assets are linked in Section 1. The retained source, data, and artwork hashes were verified after refresh.

| Frozen cache ID | Table method | Existing normalized encoded-output cache |
|---|---|---|
| 00 | Global continuation | [00_outputs.pt](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/00_outputs.pt) |
| 01 | Hard patch16/stride16/Top25 | [01_outputs.pt](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/01_outputs.pt) |
| 02 | Hard patch16/stride8/Top25 | [02_outputs.pt](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/02_outputs.pt) |
| 03 | Hard patch16/stride8/Top10 — primary method | [03_outputs.pt](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/03_outputs.pt) |
| 04 | Soft patch16/stride16/T=0.25 | [04_outputs.pt](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/04_outputs.pt) |
| 07 | Multi-scale hard Top25, weights0.7/0.3 | [07_outputs.pt](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/07_outputs.pt) |
| 08 | Excess patch16/stride16/threshold1/scale3 | [08_outputs.pt](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/08_outputs.pt) |
| 09 | Gradient-aware Top10/alpha2 | [09_outputs.pt](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/09_outputs.pt) |

All eight cache files exist. The producer saves `encoded`, `checkpoint`, and `checkpoint_sha256`; cached tensors are normalized encoder outputs, **not pre-clipped RGB**. Apply the frozen conversion before deriving fresh quality/patch arrays. Main-table rows cover eight methods and the per-image table covers400 method/image records. Figures may emphasize00/03 while the full eight-row table preserves all ablations.

### Shared inputs and checkpoints

The [fixed formal manifest](/mnt/wmcontent/GLX/icassp/MBRS/reports/uniform_eval_manifest.pt) contains 50 RGB views of size 128×128 and 50 fixed 64-bit messages. Its SHA-256 is `36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339`. Indices are zero-based manifest positions, not source dataset filenames. The historical manifest does not preserve original filenames; do not reconstruct an unsupported filename association.

The [RGB provenance record](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/provenance.json) records all 50 crop coordinates, display rules, and checkpoint hashes. The manifest and all four listed checkpoint hashes were independently recomputed and matched during this audit.

| Paper label | Exact epoch20 checkpoint | Verified SHA-256 |
|---|---|---|
| Global continuation | [checkpoint](/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/controlled_seed17_global_continuation/checkpoint_0020.pth) | `31fddd342f8bd8bd003e4fbc0cf10878035b1cda81d3e10a647833aad84e5248` |
| Hard Top10 | [checkpoint](/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50/checkpoint_0020.pth) | `74fd12a6147b20e68477d9439cf6c416181bc48f07cfa3c3384989f48850f92c` |
| Hard Top25 | [checkpoint](/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/controlled_seed17_hard_patch16_stride8_top25_weight50/checkpoint_0020.pth) | `ce3949ade1cd298b4c55e5290bff7cc0bf240656172b18b9c72155355591eb16` |
| Gradient-aware Top10 alpha2 | [checkpoint](/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_contentaware_gradient_patch16_stride8_top10_alpha2_global0.5_local0.5/checkpoint_0020.pth) | `2a4b12a61c1a437815fd2fc36fbb428d7aaf0fcc1da1f63ff68dc7b48f53be83` |

Training-history context is in the [controlled continuation protocol](/root/workspace/GLX/icassp/MBRS/reports/controlled_continuation_protocol.md). Source model and Adam state are shared within seed17, with 20 continuation epochs, batch16, LR=1e-4, message-loss coefficient10, and RandomCrop(0.3,1.0). This is provenance of completed experiments, not a request to resume them.

### Range and aggregation contract

| Evidence family | Actual convention | Paper treatment |
|---|---|---|
| Historical hard/soft method SVGs | Training objectives on normalized model tensors; no measured image-quality values | Keep the true training convention; do not insert clipping into the historical training loss |
| `artifact_observation` PNG/PDF/native RGB | `round(255 × clip((x+1)/2,0,1))`; nearest-neighbor enlargement | Valid actual-display evidence, with 8-bit quantization disclosed |
| Controlled plots/master table and old concentration/perceptual plots | Direct normalized tensors, nominal `[-1,1]`, with some encoded values outside that range; pixel PSNR uses numerator4; legacy SSIM and unclipped LPIPS | Historical evidence only; **not publication-ready under the frozen quality convention** |
| Extended-quality CSVs and paper-freeze-v1 | Float RGB `clip((x+1)/2,0,1)` before quality metrics; LPIPS then receives `2*RGB-1` | Use paper-freeze-v1 for final figures; do not evaluate these metrics on the quantized Fig. 2 PNGs |
| Historical `visualizations/external_baselines` PNGs | Clipped-RGB quality, but TrustMark globalPSNR is mean per-image dB rather than dataset-MSE PSNR | Excluded; these old files remain stale and not publication-ready |
| Selected `visualizations/paper_freeze_v1` PNG/PDF | Frozen clipped-RGB quality; S1 globalPSNR corrected to dataset-MSE aggregation | Ready for drafting; final print-size layout check pending |

Primary sources: [legacy controlled evaluator](/root/workspace/GLX/icassp/MBRS/experiments/evaluate_controlled_seed17.py), [legacy patch/SSIM implementation](/root/workspace/GLX/icassp/MBRS/experiments/analyze_patch_distortion.py), [frozen extended evaluator](/root/workspace/GLX/icassp/MBRS/experiments/evaluate_extended_image_quality.py), and [metric-consistency audit](/root/workspace/GLX/icassp/MBRS/reports/external_metric_consistency_audit.md).

Use these exact definitions throughout Figs. 3–5:

- Global PSNR: `10 log10(1 / mean_i MSE_i)` from clipped float RGB. This is not the arithmetic mean of per-image PSNR.
- Evaluation grid: 32×32 non-overlapping patches, stride32, a 4×4 grid of 16 patches per image. This differs from the method's 16×16/stride8 training grid.
- Top25 localPSNR: select the four highest-MSE evaluation patches per image, average their MSE, convert to PSNR, then average those per-image dB values. This is the actual newer `top25_local_psnr` CSV implementation. Legacy `worst_psnr` instead converts the dataset mean of those tail MSEs to dB. Neither quantity is the single worst patch, and they cannot be silently interchanged.
- P95 MSE: compute the95th percentile of the16 native32 patch MSEs separately for each image, then average across images. The paper value is not a percentile pooled over800 patches.
- Evaluation Top10 means `ceil(16 × 0.10)=2` patches, actually 12.5% of the grid. Report that rounding. Pixel-MSE, LPIPS, and SSIM tails are ranked separately by their own metric; they need not occupy the same locations.
- Full SSIM: `pytorch_msssim`, range1, window7, valid Gaussian filtering. Local SSIM: native32, window5, range1. Legacy SSIM uses the project's zero-padded window5/range2 calculation and cannot be fixed by axis relabeling.
- LPIPS: AlexNet v0.1, native32 local support; clipped RGB is mapped back to LPIPS's expected `[-1,1]` input. Full-image LPIPS and the mean of the two highest patch LPIPS values are separate quantities.
- Full-image MS-SSIM, if referenced in the accompanying table: three scales, weights `[0.3,0.3,0.4]`, window7. No local MS-SSIM, DISTS, or GMSD claims: the latter two remain unavailable.
- Gini, population CV, Top10/Mean, and Top10 energy share are per-image patch-MSE statistics, then averaged over images. Top10 energy share is the sum of the two highest patch energies divided by total patch energy. Lower concentration does not by itself establish less visible distortion.

Global and Top10 have approximately 0.774% and 0.764% out-of-range encoded channel values before display clipping. Consequently, dividing legacy MSE by four is insufficient to reproduce clipped metrics. Re-evaluate or use the existing clipped CSVs. Ratios and rankings can also change after clipping.

## 3. Fig. 1 — method

Use the final `fig01_method` PNG/PDF linked above. The renderer now explicitly shows the clean encoder output, rectangle-mask channel, decoder scores, training-only local branch, and separate native32 evaluation grid. This replaces the older stride16/Top25 SVG.

The upper lane is host plus 64-bit message → encoder → clean output → rectangle mask → decoder scores/message MSE. The lower lane compares clean output with host, computes global MSE and overlapping patch scores, and forms the objective. Losses operate before the attack; local selection adds no inference module.

Verified against the [selected config](/root/workspace/GLX/icassp/MBRS/experiments/config_controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_128_m64.json), [loss implementation](/root/workspace/GLX/icassp/MBRS/experiments/losses.py), and [training source](/root/workspace/GLX/icassp/MBRS/experiments/train_local_patch.py):

- A 128×128 input produces a 15×15 grid of overlapping 16×16 patches at stride8: 225 scores.
- Ceiling-rounded Top10 selects `ceil(225×0.10)=23` scores.
- Main objective: `10 L_message + 0.5 L_global + 0.5 L_tail`; control: `10 L_message + L_global`.
- “RGB MSE” in this training schematic refers to the original normalized RGB-tensor training objective. It does not introduce the later evaluation-only clipping step.
- Selected overlapping patches do not imply exactly 10% unique pixel coverage.

**Exact caption:**

> Figure 1. Fine-grained overlapping hard-tail supervision for crop-robust watermarking. The encoder embeds a 64-bit message in a 128×128 RGB host, and the decoder estimates message scores after a rectangular mask channel. During training, the clean encoded image and host define global image MSE and 225 overlapping 16×16 patch-MSE scores at stride8. The local loss averages the 23 highest scores, corresponding to ceiling-rounded Top10% selection. The main objective is 10 times message MSE plus 0.5 times global image MSE plus 0.5 times local-tail MSE; the Global continuation control uses global image MSE with coefficient1 and no local term. The local branch is training-only. Evaluation separately uses sixteen non-overlapping 32×32 patches.

**Layout check:** Keep the training-only note, 23/225 count, and evaluation-grid note legible after reduction. Decoder scores are correctly unbounded by a claimed sigmoid; BER uses threshold0.5. No method-data correction remains pending.

## 4. Fig. 2 — predetermined actual-RGB examples

Reuse the existing RGB files in Section 1, without regenerating or altering their pixels. Preserve sample order **7 → 20 → 42**. Each sample has a 3×3 grid: columns Original / Global continuation / Hard Top10; rows full128×128 / low-texture32×32 crop / high-texture32×32 crop. The PDF contains one page per sample, not an already assembled single-page figure. Use these blocks as subfigures (a–c).

| Manifest index | Content descriptor, not source filename | Low-texture top-left (x,y) | High-texture top-left (x,y) |
|---:|---|---|---|
| 7 | Snow, skier, dark clothing | (96,64) | (64,64) |
| 20 | Sand, crocodile, shadow edges | (64,96) | (32,64) |
| 42 | Sky, pyramid, foreground edges | (0,0) | (96,96) |

Coordinates are measured from the upper-left corner of the native128×128 view; all ROIs are32×32. The [RGB renderer](/root/workspace/GLX/icassp/MBRS/experiments/render_artifact_observation.py) selects the minimum/maximum original luminance variance among sixteen non-overlapping32×32 blocks, with row-major tie breaking. Luminance uses displayed-original RGB and weights [0.299,0.587,0.114]. Neither model output nor measured improvement enters the ROI selection. “Low/high texture” is a variance proxy, not artifact severity.

The [RGB provenance](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/provenance.json) supplies all50 coordinates and checkpoint hashes. [Native RGB files](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/native_rgb), [all-50-image HTML](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/artifact_observation.html), [auxiliary Top25/Gradient panel](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/artifact_observation_supplement.png), and [observation guide](/root/workspace/GLX/icassp/MBRS/reports/artifact_observation_guide.md) remain supplementary inspection resources. Preserve the predetermined examples even when the method differences are subtle.

**Exact caption:**

> Figure 2. Actual RGB outputs on predetermined formal-test indices7,20, and42, shown in that order. Within each sample, columns show the original, Global continuation, and Hard Top10; rows show the full128×128 view and the same low- and high-texture32×32 regions across methods. Regions are selected solely by minimum and maximum original-image luminance variance on a non-overlapping32×32 grid, with row-major tie breaking. Images are converted by clip((x+1)/2,0,1), rounded to8-bit RGB, and enlarged with nearest-neighbor interpolation. No residual amplification, sharpening, or contrast adjustment is applied. The examples illustrate actual outputs and visible differences, not consistent human-perceptual superiority of Hard Top10.

**Layout check:** Preserve coordinates, pixel levels, nearest-neighbor interpolation, and method order. Check three-block legibility at the final two-column width and retain a native-scale viewing reference in the supplement. No real-browser rendering of the HTML is claimed; the static figures are the drafting assets.

## 5. Fig. 3 — global quality, localPSNR, and BER

Use final `fig03_quality_robustness`: a 1×3 grid of clean globalPSNR, clean Top25 localPSNR, and BER versus nominal retained area **30/40/50/70/100% only**. There is no35% point. The first two panels show point estimates with numeric annotations; the BER curves use the same two methods. Data come from frozen main-table rows00/03, not the older controlled quality JSON.

| Series | GlobalPSNR dB | Mean per-image Top25 localPSNR dB | BER30 | BER40 | BER50 | BER70 | BER100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global continuation | 36.263172 | 35.522213 | 0.1131250 | 0.0416250 | 0.0014375 | 0 | 0 |
| Hard Top10 | 36.442300 | 35.754376 | 0.1129375 | 0.0418125 | 0.0015000 | 0 | 0 |

The [freeze producer](/root/workspace/GLX/icassp/MBRS/experiments/freeze_paper_evidence.py) replays integer BER counts using `decoder(encoded * mask) > 0.5`. Encoded outputs are unquantized normalized tensors for this attack protocol, while clean quality uses clipped float RGB. Each crop condition has50 images ×5 repeats =250 trials and16000 evaluated message bits. A mask is shared within each batch16; zero outside the mask corresponds to display midgray. Nominal retained areas undergo integer side-length rounding. This is a fixed rectangle-mask protocol, not physical crop-and-resize or decoded-PNG robustness. The [extension manifest](/mnt/wmcontent/GLX/icassp/MBRS/reports/controlled_crop35_40_manifest.pt) provides the40% masks used here;35% masks exist historically but are not plotted.

**Exact caption:**

> Figure 3. Quality and crop robustness of the seed17 epoch20 Global continuation and Hard Top10 models on the same50-image formal manifest. Left: clean-image globalPSNR from dataset-mean clipped-RGB MSE. Middle: Top25 localPSNR, averaging per-image PSNR of the four highest-MSE patches on a native32×32 non-overlapping grid. Right: raw64-bit BER at nominal retained areas30%,40%,50%,70%, and100%, using the fixed normalized-tensor rectangle-mask protocol with five repeats per image and decoder threshold0.5. Quality evaluation clips RGB; the replayed decoder-input protocol uses unquantized normalized outputs. Hard Top10 improves global and localPSNR, while BER changes are small and mixed across crop areas.

**Interpretation/layout:** Improvements from unrounded table values are+0.179128dB globally and+0.232163dB locally. The paired-diagnostics mean per-image globalPSNR delta is+0.165961dB, a different statistic. Do not mix those aggregations. Curves nearly overlap; retain the table for exact BER values. Check numeric annotation/legend legibility at print size. No historical seed errorbars or significance claim is attached.

## 6. Fig. 4 — mean Lorenz curve and paired Gini

Use final `fig04_concentration`, a 1×2 grid. Left: mean per-image Lorenz curves for00/03 and an equal-energy diagonal. Right: paired Gini, x=Global and y=Hard Top10, with an identity diagonal.

The renderer reads the saved00/03 normalized outputs and original manifest, converts both with `clip((x+1)/2,0,1)`, and computes sixteen non-overlapping32×32 patch-MSE scores per image. For each image, it sorts the16 scores ascending, normalizes cumulative sums by that image's total patch MSE, and prepends zero. The displayed curve averages these50 normalized curves at the17 common patch-fraction positions. It does **not** pool all800 patches into one Lorenz curve or normalize after averaging.

Gini points come from the frozen per-image CSV. The actual row order was checked: each selected method contains indices0–49 exactly once and in order, so positional pairing in the renderer is valid for these hashed inputs. The visual is not a matched-globalPSNR comparison.

**Exact caption:**

> Figure 4. Residual-energy concentration on clipped RGB using sixteen native32×32 non-overlapping patches per image. Left: the mean of50 per-image Lorenz curves, each formed by sorting patch MSE and normalizing cumulative energy within that image; the diagonal denotes equal patch energy. Right: Gini coefficients paired by formal-test image index for Global continuation and Hard Top10; points below the diagonal indicate lower concentration with Hard Top10. Gini decreases on49 of50 images (98%), with mean values0.095007 and0.086897, respectively. The concentration change is modest and does not by itself establish perceptual superiority.

**Interpretation/layout:** MeanΔGini=−0.008109840; medianΔ=−0.007310786. Closely spaced Lorenz curves should remain visible without implying a larger effect. Check line contrast and paired-point visibility at print size. The clipped-output Lorenz reconstruction is complete; the earlier “missing per-patch source” gap no longer applies to this selected asset.

## 7. Fig. 5 — all-image signed pixel/perceptual deltas

Use final `fig05_pixel_perceptual`, a 1×2 grid. Both panels show **all50 fixed indices0–49**, in manifest order, with signed deltas Hard Top10 minus Global. Left: per-image P95 patch MSE. Right: per-image Top10 patch LPIPS. Negative values are improvements and are colored blue; nonnegative values are orange. Both panels retain a horizontal zero line and their own scientific-notation y-axis.

P95 is computed within each image's16 native32 patches; it is not a pooled percentile. Top10 LPIPS is the mean of the two highest LPIPS patch values, because ceil(16×0.10)=2. LPIPS ranks its own patches, not the P95-MSE locations. All quality measures use the clipped-RGB contract; LPIPS receives the corresponding conversion back to its expected range. Data are frozen per-image rows00/03, and no example subset or sorting by improvement is used.

| Signed paired metric | Images with lower value | Mean Hard−Global | Median Hard−Global |
|---|---:|---:|---:|
| P95 patch MSE | 48/50 =96% | −0.0000171980721 | −0.0000156407586 |
| Top10 patch LPIPS | 31/50 =62% | +0.00000487025362 | −0.0000331192277 |

The percentages and means were independently recomputed from the frozen CSV and agree with the artwork and paired diagnostics. Patch-tail LPIPS averages0.00136127 for Global and0.00136614 for Hard Top10: a slight worsening despite a majority of negative paired changes. Full-image LPIPS improves, but it is a separate metric and is not this panel's y-axis.

**Exact caption:**

> Figure 5. Signed pixel-tail and perceptual-tail changes for all50 formal-test image indices, with Hard Top10 minus Global continuation on the vertical axes. Left: per-image P95 MSE over sixteen native32×32 patches. Right: the mean LPIPS of the two highest-LPIPS patches per image, corresponding to ceiling-rounded Top10% selection. Metrics use clipped RGB, with the required LPIPS input-range conversion. Negative values indicate improvement. P95 MSE decreases on48/50 images (96%), whereas patch-tail LPIPS decreases on31/50 (62%); mean changes are−1.72×10⁻⁵ and+4.87×10⁻⁶, respectively. Thus pixel-tail improvement is consistent, while mean perceptual-tail LPIPS slightly worsens despite improvement on most images.

**Layout check:** Keep negative and positive deltas, zero lines, scientific exponents, and all50 indices visible. Check that annotation boxes do not hide extreme points after reduction; the upper-left LPIPS summary lies near a positive outlier. Preserve this outcome rather than replacing it with a correlation, distribution, or cherry-picked qualitative panel. No human-perception or causal claim is supported.

## 8. Fig. S1 — two-panel TrustMark reference

Use final `figS1_external_reference`, a 1×2 grid: clean-image globalPSNR versus pre-ECC bit accuracy on the left; full-image LPIPS versus the same accuracy on the right. There are **no localPSNR or Gini panels** in the selected S1. Four method markers identify MBRS Global, MBRS Hard Top10, TrustMark Q, and TrustMark P; colors identify retained areas70/50/40/30%. The100% condition remains in the table, not this plot.

The renderer reads the corrected frozen main/external tables. TrustMark globalPSNR coordinates are **Q42.293755dB and P48.107281dB**, from dataset mean MSE, matching MBRS aggregation. These replace the old mean-per-image-PSNR coordinates. Quality concerns clean outputs, repeated against each crop condition's accuracy; it is not attacked-image quality.

The [TrustMark raw-bit audit source](/root/workspace/GLX/icassp/MBRS/external_baselines/audit_trustmark_raw_bits.py), [raw-bit audit report](/root/workspace/GLX/icassp/MBRS/reports/trustmark_raw_bit_audit.md), [per-trial data](/mnt/wmcontent/GLX/icassp/MBRS/reports/trustmark_raw_bits/per_trial.csv), and [external crop protocol](/root/workspace/GLX/icassp/MBRS/reports/external_crop_protocol.md) document the reference semantics. TrustMark uses61 data+35 parity+4 schema bits, the official BCH-5 protected100-bit packet, decoder-logit threshold0, unchanged strength, PNG quantization, and official decoder preprocessing. MBRS uses64 uncoded bits and threshold0.5. Each crop condition has50 images ×5 repeats.

**Exact caption:**

> Figure S1. Supplemental quality–robustness reference points for MBRS Global continuation, MBRS Hard Top10, and TrustMark Q/P at nominal retained areas70%,50%,40%, and30%. Left: clean-image globalPSNR computed from dataset-mean clipped-RGB MSE. Right: full-image LPIPS. Vertical axes show pre-ECC bit accuracy:64 uncoded message bits for MBRS and100 transmitted packet bits for TrustMark, comprising61 data bits,35 parity bits, and4 schema bits. Shapes indicate methods and colors indicate crop conditions. TrustMark's corrected dataset-MSE PSNR is42.293755dB for Q and48.107281dB for P. All cross-family points are REFERENCE only because payload, ECC, quantization, and decoding pipelines differ; they do not establish strict equal-payload superiority.

**Interpretation/layout:** Preserve REFERENCE labels and the packet distinction. Raw BER/accuracy, ECC exact-message success, and decode/detection flags remain separate in the table. A decode flag is not a calibrated detection rate on unwatermarked images. At print size check LPIPS tick spacing and the two-row method/crop legend; scientific notation or fewer ticks may be a typesetting refinement. Corrected numerical coordinates and PNG/PDF exports are complete.

## 9. Remaining checks for the manuscript owner

No selected figure awaits a metric rebuild or new method diagram. Remaining work is no-training production review:

1. Place the PDFs and existing Fig.2 blocks in the actual paper layout; check font size, panel ordering, crop visibility, line contrast, axis exponents, and legend overlap. Match each caption above to its selected panel arrangement.
2. Preserve explicit units/aggregations: dataset-MSE globalPSNR, mean per-image localPSNR, mean per-image P95, and all-image signed deltas. Retain the separate unquantized normalized-tensor BER protocol.
3. Keep figure/table/source provenance together. The render manifest hashes tables, per-image CSV, renderer, and outputs. Fig.4 also depends on00/03 caches and the formal manifest; their identities are traced through the evidence manifest and cache checkpoint metadata, but the renderer's own JSON does not hash those cache files. Adding cache-file hashes would strengthen archival reproducibility; it is not missing numerical evidence or a drafting blocker.
4. Freeze the interpretation as single-seed pixel-tail/concentration improvement with small mixed BER changes and mixed perceptual-tail evidence. No multi-seed significance, final matched-PSNR perceptual advantage, human-rated superiority, or strict superiority over TrustMark is claimed.

Verification for this plan: actual absolute-link existence; manifest/four qualitative checkpoint hashes earlier in the audit; all14 refreshed render-provenance hashes;50 ordered image pairs; independently recomputed96%/62%/98% signed-improvement counts; visual inspection of selected PNGs. No training or figure generation was required. Only `paper/figure_plan.md` was modified.

## Appendix — legacy audit, excluded from current selection

This appendix describes historical assets only. Their limitations do not change the current drafting-ready status of the selected `paper_freeze_v1` figures.

| Historical asset/source | Actual issue | Final disposition |
|---|---|---|
| [Hard-worst SVG](/root/workspace/GLX/icassp/MBRS/reports/mbrs-framework-hard-worst.svg) and [soft-worst SVG](/root/workspace/GLX/icassp/MBRS/reports/mbrs-framework-soft-worst.svg) | Hard diagram shows stride16,64 patches,Top25 selection; older decoder bound is unsupported | Superseded by final Fig.1 with23/225 and decoder scores |
| [Historical quality/BER suite](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_suite_20260907) and [suite source](/root/workspace/GLX/icassp/MBRS/experiments/generate_paper_figure_suite.py) | Different methods/epochs/seed populations and legacy tensor-space quality | Not publication-ready for this comparison; final Fig.3 uses frozen tables |
| [Older three-figure source](/root/workspace/GLX/icassp/MBRS/experiments/make_paper_figures.py) | Some quality values hard-coded | Not a final data source |
| [Legacy Lorenz curve](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/matched_psnr_residual_concentration/lorenz_curve.png) and [legacy paired Gini](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/perceptual_tail_visualization/paired_gini_global_vs_top10.png) | Unclipped nominal [-1,1] numerical evidence; historical renderer recipe not fully recovered | Not publication-ready under frozen metrics; final Fig.4 has an explicit retained renderer |
| [Legacy pixel/perceptual scatter](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/perceptual_tail_visualization/top10mean_vs_top10_lpips.png) and [paired LPIPS](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/perceptual_tail_visualization/paired_lpips_global_vs_top10.png) | Legacy range/SSIM and past grid/pairing inconsistencies; old64% LPIPS count | Excluded; final Fig.5 displays frozen all-image signed deltas and62% |
| [Legacy qualitative heatmap composite](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/perceptual_tail_visualization/qualitative_residual_perceptual_tail_v2.png) | Independently scaled heatmaps do not support shared-intensity comparison | Excluded; Fig.2 uses actual RGB at fixed indices |
| [Old four-panel external plot](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/external_baselines/quality_robustness_tradeoff.png) and [old generator](/root/workspace/GLX/icassp/MBRS/experiments/generate_quality_robustness_pareto.py) | Clipped RGB but TrustMark globalPSNR was mean per-image dB; localPSNR/Gini panels also differ from final layout | Not publication-ready; final S1 is two panels with corrected dataset-MSE PSNR |

Supporting historical audits: [metric consistency](/root/workspace/GLX/icassp/MBRS/reports/external_metric_consistency_audit.md), [perceptual evidence](/root/workspace/GLX/icassp/MBRS/reports/perceptual_evidence_audit.md), [perceptual audit source](/root/workspace/GLX/icassp/MBRS/experiments/audit_perceptual_evidence.py), and [matched-PSNR report](/root/workspace/GLX/icassp/MBRS/reports/matched_psnr_residual_concentration_and_perceptual_tail.md). The closest old matched-PSNR pair is epoch1/epoch1, not the final model. Historical Pearson correlations were sometimes called Spearman, and16-grid counts were mixed with32-grid counts. Do not import those annotations into final captions.

Legacy PSNR uses range2 on unclipped normalized tensors; legacy SSIM uses window5, zero padding, and max_value2. New quality uses clipped [0,1], valid-filter SSIM, and explicit LPIPS conversion. Clipping changes rankings and normalized concentration, so dividing old MSE by4 or relabeling old axes is not a valid refresh. The selected figures have already received the required data/artwork refresh.
