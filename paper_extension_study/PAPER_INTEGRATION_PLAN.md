# Paper Integration Plan

This is a recommendation only. Do not edit the Prism manuscript or `paper_bundle/` until explicitly approved. The frozen method and formal evidence remain authoritative.

## A. What should enter the four-page main paper

1. Keep the current frozen method definition and natural project-test Table I unchanged.
2. Add a compact sentence or footnote identifying the corrected PSNR estimator: PSNR is computed from aggregate mean clipped-RGB MSE, not the mean of per-image PSNR.
3. Add one cautious bootstrap sentence:

   > Paired image-level bootstrap intervals (20,000 resamples) excluded zero for the reported fidelity/tail metrics, while the BER30 interval included zero; therefore no directional BER improvement is claimed.

4. If space permits, add one sentence:

   > The fidelity trend persists across stochastic continuation seeds from a common frozen source checkpoint.

5. If practical cost is relevant, add one sentence:

   > The training-only losses added approximately 1.36% to mean training-step time in our benchmark, while parameter count, inference graph, and peak inference memory were unchanged.

## B. What should remain supplementary or omitted

- Full factorial component table and bootstrap distributions.
- Full Global+OKLab-only and RGB0.5 control rows.
- Full operating curves and interpolation tables.
- Per-seed continuation checkpoints, logs, and all individual seed metrics.
- Full overhead timing distributions and environment details.
- Method-interpretation note, Figure 1 revision notes, reference audit, and negative/mixed findings.

## C. Should Table I be restructured?

No. Preserve Table I as the natural frozen source-resolution formal comparison: MBRS crop-trained global, Hard Local-Tail, and Ours. Do not insert RGB0.5-Control or Global+OKLab-only into the formal main table, and do not mix matched-PSNR or display-space values into it.

## D. Should the 2×2 factorial table appear in the main paper?

No, not as a full table under the four-page constraint. It is valuable supplementary causal context, but the fixed-RGB-weight design excludes the primary RGB-weight-1.0 baseline and contains diminishing/mixed interaction findings. A one-sentence summary may be used if the reviewer concern warrants it; use “complementary effects” and not “synergy.”

## E. Should matched-PSNR Table II remain or be replaced by an operating-curve figure?

Keep the compact internal matched-PSNR Table II in the main paper. It directly supports the claim that the improvements are not explained only by higher global PSNR and preserves the source-resolution metric boundary. Put the operating-curve figure in supplementary material; do not replace Table II with the external Figure 2 display-space comparison.

## F. Bootstrap sentence(s)

Recommended main-paper wording:

> In paired image-level bootstrap analysis over the 50 project-test images, the 95% intervals excluded zero for PSNR, Top-25 Local PSNR, P95/P99, LPIPS, and CIEDE2000, whereas the BER30 interval included zero. This supports stable fidelity differences under image resampling but does not establish a directional BER improvement.

Use the corrected PSNR bootstrap output, and label all remaining values as natural frozen 128×128 project-test evidence.

## G. Overhead sentence

Recommended wording:

> Hard Local-Tail and OKLab are training-only losses: the benchmark measured approximately 1.36% mean training-step overhead, with identical parameter count, inference architecture, and peak inference memory for Global and Ours.

Do not state that inference latency is exactly equal; the measured latency difference is noisy.

## H. Continuation-seed sentence

Recommended wording:

> With the same frozen seed17 epoch-100 source checkpoint, the primary Ours-versus-Global fidelity direction persisted for continuation seeds 17, 23, and 42; Gini and BER30 remained mixed, and this experiment is not full end-to-end multi-seed reproducibility.

## I. Claims that must not be made

- State of the art or universal superiority over HiDDeN/MaskWM.
- Significantly stronger crop robustness.
- Human perceptual superiority.
- Uniform residual distribution.
- Gini as a primary objective.
- External BER ranking or superiority.
- Full end-to-end reproducibility.
- Synergy between Tail and OKLab.
- The external `40.72 dB` display result as a formal `128×128` result.
- Any historical `w_msg=80`, Ours-base, obsolete figure/table, or validation-only OKLab result as the final method/evidence.
