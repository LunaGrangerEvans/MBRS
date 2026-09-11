# Paper evidence freeze checklist

**PAPER READY FOR DRAFTING — YES.** The complete twelve-section v1 draft and the selected figure package are available. This means ready for scientific drafting and review, not ready for submission: primary-source citations, novelty review, venue formatting, and final print-size layout remain editorial requirements.

## Frozen scope

- Main method: seed17, Hard raw-MSE selection, patch16 / stride8 / Top10%, global/local weights 0.5/0.5, message MSE weight 10, continuation20.
- Control: Global continuation, image weight1, identical source epoch100, optimizer restoration, LR, batch size, training duration, and crop protocol.
- Eight specified methods only in the paper main table; historical extra seeds and JPEG+OKLab excluded.
- This freeze used existing checkpoints for evaluation. No training, new seed, ratio, alpha, patch size, loss, or backbone selection was performed.

## Audit results

| Required check | Result | Concrete evidence / treatment |
|---|---|---|
| All eight main-table rows have the same quality definition | PASS | [main_table.csv](../paper/main_table.csv), 400 image-level records in mounted `reports/paper_freeze_v1/per_image.csv`; reused `evaluate_extended_image_quality.evaluate` |
| Numeric rows trace to checkpoints and configuration | PASS | [evidence_manifest.json](../paper/evidence_manifest.json); eight epoch20 hashes, configs and historical JSON bindings; registry checks seed17/source/duration before inference |
| Shared Global/main source | PASS | Source SHA-256 `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`; current model/Adam/BN restoration supported by checkpoint, runner, and resolved config |
| Crop BER recomputed with shared coordinates and counts | PASS | All eight methods replayed base and crop40 extension masks; every 100/70/50/40/30 BER agrees with original integer-count results to <1e-7 |
| Crop semantics accurately described | PASS | Rectangle masking on 128×128 normalized tensors, no resize-back; independent training side-fraction draws, not uniform area; nominal evaluation percentages use floor-rounded pixel sides |
| Training/evaluation patch distinction | PASS | Training225 patch16/stride8 candidates, top23 selected; evaluation16 native32/stride32 patches; Top25 uses4, Top10 uses2 (effective12.5%) |
| Consistent quality clipping, SSIM, LPIPS, and aggregation | PASS | RGB clip to[0,1]; SSIM valid Gaussian7, local SSIM5; LPIPS AlexNetv0.1; MS-SSIM3 scales [0.3,0.3,0.4]; draft §5 |
| PSNR/local-PSNR/P95 reduction clarified | PASS | Global PSNR from dataset mean MSE; local PSNR averaged per-image in dB; P95 is mean per-image quantile. Legacy pooled reductions excluded |
| Concentration fractions interpreted correctly | PASS | Recomputed lower-is-better fractions directly from paired records: Gini/CV98%, Top10Mean/share92%; old summary's direction bug not copied |
| TrustMark global PSNR aggregated consistently | PASS | Recomputed from saved per-image MSE: Q42.293755, P48.107281; older mean-image-PSNR summaries and Pareto coordinates not copied |
| TrustMark comparison is REFERENCE | PASS | Packet61 data +35 parity +4 schema; PNG quantization and native decoder resize disclosed; raw-bit, ECC exact and decode flag kept separate |
| Every selected figure has a frozen source | PASS for draft assets | [figure_plan.md](../paper/figure_plan.md); updated Fig1/3/4/5/S1 with PNG/PDF and hash manifest under mounted `visualizations/paper_freeze_v1`; Fig2 reuses existing RGB provenance |
| Qualitative selection independent of method gain | PASS | Predetermined test indices7/20/42; ROI chosen from original-image low/high variance; no replacement by favorable cases |
| No human superiority claim | PASS | Draft explicitly reports absent observer study and mixed local LPIPS; images are illustrations |
| No crop-causes-concentration claim | PASS | Introduction separates motivation from unproven concentration hypothesis; crop training held fixed |
| Honest study boundaries | PASS | Seed17-only, small sample, prior test exposure, no final matched-PSNR causal claim, incomplete historical RNG/environment records |

## Issues repaired before drafting

1. The four-method extended table could not supply eight complete ablation rows. Existing eight checkpoints were re-evaluated under the same quality routine; none was retrained.
2. Earlier external PSNR used a different reduction from MBRS. The paper reference table applies dataset-average MSE to the original per-image records.
3. Old normalized-tensor figures do not match clipped-RGB paper metrics. Only the selected scientific plots were refreshed. Old graphics remain historical and are not relabeled as new data.
4. The old framework artwork showed stride16/Top25. Its two-lane concept is retained in the corrected compact diagram with patch16/stride8/23-of-225, training-only local supervision, and unconstrained decoded scores.
5. The higher-is-better generic summary logic is inappropriate for Gini/CV. Paper improvement rates are independently computed from paired per-image records, rather than copied from that field.

## Remaining evidence limitations, not requests for experiments

- No pristine blind-test claim: earlier test diagnostics informed research decisions. This cannot be repaired by a new document or renamed partition.
- No cross-seed significance/equivalence claim, human-perception claim, or broad dataset/backbone generalization claim.
- No final-epoch matched-global-PSNR causal evidence. The local-minus-global PSNR gain mixes reductions and is descriptive.
- MBRS image-quality metrics describe clipped floating-point RGB, while frozen BER describes normalized unquantized tensor inference. PNG-deployment robustness is outside the stated result.
- Historical Adam/BN restoration is supported by artifacts and source. Per-step RNG trajectories and historical environment identity were not recorded completely.
- The existence of more image metrics does not validate residual concentration as a perceptual proxy. The Excess row directly illustrates why relative concentration cannot be judged alone.

There is no remaining experiment required for the **narrow descriptive claim of this draft**. These limitations remain in the manuscript rather than triggering new training.

## Editorial work remaining

All 13 requested sections are written in English, with inline `[CITATION NEEDED]` markers. Verify MBRS/HiDDeN/StegaStamp/TrustMark, crop robustness, DIV2K, top-k/tail-risk prior work, SSIM/MS-SSIM/LPIPS, and concentration definitions. Assess novelty against the primary literature before making a submission-level contribution claim. A full reference list is intentionally absent until verification.

The draft is approximately 4,200 words before captions and the full auxiliary tables. It still needs conference page-budget compression, final figure placement/print-size checks, author information, and bibliographic formatting. None authorizes new experiments.

## Reproduction

From the MBRS root:

```bash
python experiments/freeze_paper_evidence.py
python experiments/render_paper_freeze.py
python experiments/verify_paper_freeze.py
```

The first command performs inference from the eight existing endpoints and verifies original BER counts; the second renders only selected figures from saved outputs; the third checks hashes, table aggregation, paired conclusions, artifact links, and manuscript structure. Existing checkpoints and training code are not edited by these commands. The freeze script received only import/lint edits after its initial run; its current source hash is recorded in the evidence manifest.
