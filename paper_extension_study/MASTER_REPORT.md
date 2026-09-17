# Validation-Strengthening Extension Study

## 1. Executive summary

This extension study was run without modifying the frozen final method, `paper_bundle/`, the Prism manuscript, historical outputs, or existing paper assets. New outputs live under `paper_extension_study/` and are supplementary until explicitly approved.

Feasible work completed:

- paired image-level bootstrap confidence intervals;
- Global + OKLab-only component ablation;
- Global-RGB0.5 causal control;
- fixed-grid fidelity–robustness operating curves;
- cautious empirical upper-tail-risk interpretation;
- Figure 1 design recommendation;
- reference-gap audit.

The multi-seed phase was not run. The pre-registered seeds were `17, 23, 42`, but exact seed-specific epoch-100 source checkpoints were unavailable for seeds23 and42. Reusing seed17's source or changing to full-scratch training would violate the registered protocol; the block is documented in `RESOURCE_LIMITATIONS.md`.

The strongest new evidence is the paired bootstrap: all eight listed quality/tail metrics have 95% image-level intervals excluding zero for every available method comparison, while every BER interval includes zero. The component controls do not show that a global RGB weight change alone explains the frozen Ours result. The operating curves show a favorable local/perceptual trend across the common PSNR overlap, but BER30 is mixed; no dominance claim is made.

## 2. Bootstrap confidence intervals

Source: the frozen 50-image project-test per-image CSV explicitly referenced by the authoritative bundle. Procedure: 20,000 paired image-index resamples with replacement, analysis seed `20260916`, difference `second method − first method`, and image-level BER contributions rather than individual bits.

Output files:

- `bootstrap/bootstrap_results.csv`
- `bootstrap/bootstrap_summary.md`
- `bootstrap/bootstrap_distributions.npz`
- `bootstrap/bootstrap_metadata.json`

For each comparison—Global vs Ours, Global vs Hard, and Hard vs Ours—the 95% percentile interval excludes zero for PSNR, Top-25 Local PSNR, P95, P99, LPIPS, Global CIEDE2000, Top10 CIEDE2000, and Gini. All BER100/70/50/40/30 intervals include zero. This is reported as interval behavior, not as a generic statistical-significance claim.

The Global→Ours observed per-image PSNR difference is `+0.596953233 dB` in the bootstrap input. This differs from the formal table delta `+0.591737 dB` because the formal PSNR is computed from dataset mean MSE, while the registered image-level bootstrap resamples per-image PSNR contributions. This is an aggregation distinction, not a contradiction.

## 3. Component ablation

The Global + OKLab-only branch uses the same seed17 epoch-100 source, state restoration, 20-epoch continuation, augmentation, batch size, deterministic settings, and evaluator as the frozen study. Its objective is:

```text
L = 10 L_msg + 0.5 L_RGB + 0.059149764 L_OKLab
```

It has no local-tail term and was not used to select a new method. Project-test summary:

| Method | PSNR | Top-25 Local PSNR | P95 MSE | P99 MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MBRS crop-trained global | 36.263172 | 35.522213 | 3.016531971e-04 | 3.199585360e-04 | 0.00234960 | 3.535857 | 5.159104 | 0.1131250 |
| Global + OKLab | 35.943842 | 35.207559 | 3.227863646e-04 | 3.422455385e-04 | 0.00236554 | 3.577205 | 5.215556 | 0.1132500 |
| Hard Local-Tail | 36.442300 | 35.754376 | 2.844551472e-04 | 3.010726967e-04 | 0.00221518 | 3.484790 | 5.075666 | 0.1129375 |
| Ours | 36.854909 | 36.141895 | 2.619925590e-04 | 2.782322409e-04 | 0.00191711 | 3.254091 | 4.772229 | 0.1131250 |

The Global + OKLab-only branch is a descriptive Tail-off / OKLab-on cell. It does not match or exceed the frozen baseline on the listed project-test quality metrics, so it cannot be presented as a replacement or improvement to Ours. Its validation metrics are preserved in `component_ablation/global_oklab_only/validation_metrics.json` and its project-test metrics in `project_test_metrics.json`.

## 4. RGB-weight control

The registered `Global-RGB0.5-Control` branch uses:

```text
L = 10 L_msg + 0.5 L_RGB
```

with no local-tail or OKLab term, from the same source checkpoint and under the same 20-epoch protocol. Project-test summary:

| Method | PSNR | Top-25 Local PSNR | P95 MSE | P99 MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Global-RGB0.5-Control | 35.164486 | 34.447721 | 3.813118118e-04 | 4.026092408e-04 | 0.00261828 | 4.027174 | 5.858652 | 0.1130625 |

This control is weaker than the frozen Global baseline on the listed quality metrics. It does not support explaining the frozen Ours gains as a simple reduction of the global RGB coefficient. It remains a causal control, not a candidate method; the full report is `component_ablation/rgb05_control/report.md`.

## 5. Multi-seed reproducibility

The pre-registered seed set was exactly `17, 23, 42`. Exact epoch-100 crop-trained Global source checkpoints were available for seed17, seed29, and seed41, but not seed23 or seed42. The protocol explicitly prohibits substituting seed17's source, using seed29/41 as replacements, or switching to full-scratch training.

Therefore no seed23/42 experiments were launched and no multi-seed summary is reported. This is a protocol-availability block, not a performance result. See `RESOURCE_LIMITATIONS.md` and the source gate in `PRE_REGISTERED_PLAN.md`.

## 6. Fidelity–robustness operating curves

The frozen Global and Ours checkpoints were evaluated without retraining using:

```text
x_alpha = x + alpha (x_hat - x)
```

The alpha grid was frozen before project-test evaluation and used identically for validation and project-test:

```text
0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.00
```

Outputs:

- `operating_curve/validation_curve.csv`
- `operating_curve/project_test_curve.csv`
- `operating_curve/curve_report.md`
- `operating_curve/figures/`

The project-test PSNR ranges overlap from `36.854909` to `39.351276 dB`. Across five evenly spaced interpolated PSNR locations in this overlap, Ours has lower P95 MSE, higher Top-25 Local PSNR, lower LPIPS, and lower Global CIEDE2000. BER30 is mixed and is slightly higher for Ours at most interior interpolated locations. The curve therefore supports a cautious range-level local/perceptual trend, not unconditional BER or total-metric dominance.

## 7. Method interpretation

`method_interpretation.md` explains that:

```text
L_tail = (1/K) sum of the K largest patch errors
```

can be viewed as an empirical upper-tail risk objective or CVaR-like upper-tail supervision. The note explicitly avoids claiming exact formal CVaR equivalence. It preserves the distinction between Global RGB average distortion, Hard Local-Tail upper-tail local distortion, and OKLab color-aware refinement.

This interpretation is optional drafting guidance and should not enter the paper until a verified reference is added.

## 8. Figure 1 recommendation

`figure1_revision_notes.md` recommends a clearer visual progression:

```text
Global RGB → mean distortion control
Hard Local-Tail → upper-tail local distortion control
Global OKLab → color-aware refinement
```

The note preserves the unchanged inference architecture, training-only loss boundary, frozen coefficient, and patch geometry. No figure was modified or regenerated.

## 9. Reference gaps

`reference_gap_audit.md` marks all requested categories as TODO because the authoritative bundle and its bibliography contain no verified BibTeX entries. Categories include robust deep image watermarking, MBRS, HiDDeN, MaskWM, LPIPS, OKLab, CIEDE2000, and tail-risk/CVaR-like interpretation. No references were fabricated.

## 10. Negative and mixed findings

- All BER bootstrap intervals include zero; no BER improvement is statistically established by the registered interval procedure.
- Operating-curve BER30 is noisy/mixed, and Ours is slightly worse at most interior interpolated common-PSNR locations.
- The Global + OKLab-only and RGB0.5 controls do not outperform the frozen Global baseline on the listed project-test quality metrics.
- Formal Ours Gini is lower than Global but higher than Hard Local-Tail; bootstrap intervals for these comparisons exclude zero, but Gini remains diagnostic only.
- Multi-seed reproducibility for seeds23/42 is unavailable under the exact source-checkpoint policy.
- Per-image bootstrap PSNR deltas differ from aggregate-table PSNR deltas because the two analyses use different PSNR aggregation units.

## 11. Evidence strong enough to enter the main paper

The following can support a cautious main-paper or appendix sentence if space allows:

- image-level paired bootstrap intervals for the frozen natural project-test comparisons, stated as “the paired bootstrap interval excludes/includes zero” and labeled as supplementary stability evidence;
- the fixed-protocol statement that the quality/tail intervals exclude zero while BER intervals include zero;
- a brief statement that the Global-RGB0.5 control does not reproduce the frozen Ours pattern, if the component-causality question must be addressed.

These additions must not alter the frozen main table, method definition, or claim strength.

## 12. Evidence that should remain supplementary

- full bootstrap distributions and per-metric CSV;
- Global + OKLab-only checkpoint, logs, validation/project-test rows, and 2×2 component table;
- Global-RGB0.5 control;
- complete fidelity–robustness curves and figures;
- the upper-tail-risk interpretation note;
- Figure 1 design recommendations;
- the reference-gap audit;
- the blocked multi-seed phase documentation.

Figure 2 remains the separate external 512×512 qualitative comparison defined by the authoritative bundle.

## 13. Claims that should remain unchanged

Keep the frozen paper claims and limits unchanged:

- Ours improves absolute local-tail fidelity under the controlled internal protocol.
- Ours improves measured global/local quality and perceptual/color metrics at the natural frozen operating point.
- Crop robustness is approximately preserved.
- Internal matched-PSNR analysis shows that improvements are not explained only by higher global PSNR.
- External methods are qualitative/contextual references at matched display PSNR.
- Hard Local-Tail and OKLab are training-only losses; inference architecture is unchanged.

Do not add state-of-the-art, universal external superiority, significantly stronger crop robustness, human perceptual superiority, uniformly distributed residuals, Gini-as-primary-objective, external BER ranking, or validation-as-formal-test claims.

## 14. Contradictions with the current paper bundle

No scientific contradiction was found. The extension adds supplementary branches and analyses; it does not supersede the frozen evidence. The only apparent numeric discrepancy is the Global→Ours bootstrap PSNR difference (`+0.596953233 dB`) versus the paper-story rounded formal delta (`+0.592 dB`), which is explained by per-image PSNR resampling versus dataset-mean-MSE PSNR aggregation. The operating curve at alpha1 reproduces the frozen Global/Ours natural values within floating-point output precision.

## Decision table

| Evidence item | Supports paper? | Main paper / supplement / exclude | Reason |
|---|---|---|---|
| Paired image-level bootstrap CIs | Yes, cautiously | Supplement; concise stability sentence optional | Intervals exclude zero for quality/tail metrics and include zero for BER; no generic significance claim. |
| Global + OKLab-only component ablation | Yes, as causal context | Supplement | Descriptive Tail-off/OKLab-on cell; does not replace Ours. |
| Global-RGB0.5 control | Yes, as causal context | Supplement | Shows the global-weight change alone does not reproduce the frozen pattern. |
| Multi-seed 17/23/42 | No result available | Exclude pending exact sources | Seeds23/42 lack exact epoch-100 source checkpoints; protocol forbids substitution. |
| Fixed-grid operating curves | Yes, cautiously | Supplement | Local/perceptual trend across overlap; BER30 is mixed. |
| Upper-tail-risk/CVaR-like note | Potentially | Supplement or main-method interpretation after citation | Requires a verified reference and must retain cautious wording. |
| Figure 1 revision notes | Not experimental evidence | Editorial guidance | Recommends clearer branch labels without changing the frozen figure. |
| Reference-gap audit | Not evidence | Pre-submission action | No verified bibliography entries exist in the authoritative source set. |
