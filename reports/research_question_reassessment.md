# Research-question reassessment

## Original idea

The initial idea was that crop training would concentrate watermark distortion into a small number of local regions and that a worst-patch loss would repair this concentration.

## Supported claims

- Crop-robust training increases absolute visual distortion relative to no-crop training.
- A global average image loss does not explicitly constrain the upper tail of local distortion.
- Local tail optimization can reduce Top25/P95 patch MSE without materially changing BER.
- Fine-grained Patch16 is more useful than coarse Patch32/64 under the controlled protocol.
- Overlapping Patch16 gives the strongest current controlled result: `ΔWorstPSNR=+0.233 dB`, `ΔTop25 MSE=-0.000062`, and `ΔP95 MSE=-0.000090`.

## Partially supported claims

- Overlap improves localization: stride8 improves Worst PSNR by only about `0.030 dB` over stride16. Boundary fragmentation exists, but it is not the primary bottleneck.
- The best loss improves the tail more than the global image average, but only slightly: local-specific gain is about `+0.050 dB`.
- Multi-scale supervision is plausible but the fixed `0.7/0.3` mixture diluted the stronger Patch16 signal and reached only `+0.175 dB` Worst PSNR.

## Unsupported claims

- Crop training increases spatial concentration of distortion. The top25/global ratio decreases from approximately `1.374` for no-crop to `1.181` for crop-trained Global.
- Hard worst-patch training produces a large or statistically established improvement. The controlled seed17 effect is small and no cross-seed significance claim is allowed by the current protocol.
- MSE-worst patches are necessarily perceptually worst patches. At the formal 32×32 scale, MSE-to-LPIPS and MSE-to-SSIM-rank correlations are near zero or negative.

## Claims requiring additional evidence

- Whether the modest improvement is underestimated by the 32×32 evaluation scale. Existing checkpoint analysis shows the overlap gain is about `+0.248 dB` at 8×8/16×16 analysis and `+0.233 dB` at 32×32, so the mismatch is real but small.
- Whether a content-normalized or perceptual tail metric is a better problem definition.
- Whether the result transfers beyond the current 128×128, 64-bit, RandomCrop protocol.

## Recommended framing

The defensible research question is:

> Can an explicit local-tail objective reduce high-distortion watermark regions under crop-robust training, while preserving message recovery and global visual quality?

The paper should emphasize **worst-region quality and tail risk**, with spatial uniformity as a secondary interpretation. It should not claim that crop training itself creates concentrated artifacts, and it should not frame the contribution as a general PSNR improvement.

The strongest current claim is:

> Under a controlled seed17 continuation protocol, overlapping fine-grained hard patch supervision reduces the measured local distortion tail with negligible BER change, but the effect is modest and is partly accompanied by a global PSNR improvement.
