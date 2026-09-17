# WOFA versus the MBRS frozen crop protocol

This is a protocol comparison, not a pooled leaderboard. The MBRS reference is the
project's [frozen crop protocol](external_crop_protocol.md) and fixed manifest; WOFA
entries are either
direct observations from the author release or `PAPER-REPORTED REFERENCE`; they are
not converted into MBRS-style numbers.

## Side-by-side protocol

| Dimension | WOFA (CVPR 2025) | MBRS frozen protocol | Fairness consequence |
|---|---|---|---|
| Input | 200×200 OPA blocks | 128×128 fixed manifest tensors | Different spatial scale and source view. |
| Payload | 30 raw bits; no reported ECC | 64 raw bits; no ECC | BER cannot be compared at equal payload/rate. WOFA nominal rate is `30/200²=7.5e-4`; MBRS is `64/128²=3.90625e-3`. |
| Embedder/extractor | Stage-I FC message↔pattern plus Stage-II U-Net-like image↔pattern | MBRS encoder/decoder with frozen controlled checkpoints | Different architecture and trained operating point. |
| Partial-theft object | Binary mask selects stolen content; ratio is stolen/mask area | Square rectangle mask selects retained content on original canvas | The area labels are opposite concepts; same percentage is not the same attack. |
| Mask shape | Dataset masks random/irregular; early curriculum also squares/rectangles | Rectangle only; fixed persisted coordinates | WOFA tests arbitrary object silhouettes; MBRS tests controlled rectangles. |
| Background | Stolen transformed content composited onto a new same-size background | Pixels outside retained rectangle are zeroed in normalized tensor space | Unknown-background interference and zero-fill are different channels. |
| Crop-and-resize | No standalone crop-and-resize-back specified; internal geometry resampling unspecified | No crop-and-resize-back interpolation | Both retain a full canvas, but this does not make their inputs equivalent. |
| Geometric attacks | Translation ≤50% side, rotation ≤45°, scaling ±25%; mixed settings | Frozen crop audit has no geometry in the formal crop columns | WOFA's native robustness includes a harder geometry problem. |
| Channel attacks | Stage-I Gaussian noise up to σ=0.1; Stage-II JPEG QF95 and resize 80–125% | Formal crop columns replay raw unquantized masked tensors; channel attacks are separate studies | No common JPEG/noise/resize contract. |
| Training | OPA 61,990 200×200 blocks, 4:1 split; two-stage progressive training; Adam 5e-5 | 800 DIV2K train; controlled 50-image evaluation; MBRS continuation checkpoints | Dataset, schedule, and optimization are not matched. |
| Robustness metric | BAR only | raw bit BER / bit accuracy over 64 bits; five repeats per crop | BAR complement is not enough to repair protocol/denominator/attack mismatch. |
| Quality metric | PSNR/SSIM/LPIPS, details not fully frozen in paper | RGB `[0,1]` PSNR, SSIM win7, 3-scale MS-SSIM, AlexNet LPIPS, Top25 local PSNR, P95 MSE, Gini | Same metric names do not imply same reductions or preprocessing. |
| Native evaluation | OPA test plus SOIM/matteImageNet generalization | fixed 50-image MBRS manifest, SHA-256 `36790b02…fbfd5339` | WOFA paper results are not the same sample. |

## Area-label mapping

WOFA's mathematical mask `m` denotes the stolen portion copied into the fused image;
the paper's area bins are mask/stolen area. MBRS's formal mask `M` denotes the
retained rectangle and preserves it on the original canvas. A hypothetical same-shape
conversion would map `WOFA stolen 30%` to `MBRS retained 70%`, but this is only an
area complement. It does not match mask shape, position distribution, background
contents, geometric transformation, or decoder preprocessing.

The supplied WOFA sample code makes the distinction concrete:

```text
I_compo = mask * I_w + (1-mask) * I_bg
```

MBRS's frozen crop replay is instead conceptually:

```text
attacked = retained_rectangle_mask * encoded_tensor
```

The first is compositing against a new image; the second is masking the original
canvas with zero outside. Do not put these in one `BER@area` column without separate
protocol names.

## Metric fairness decisions

### BER / BAR

Not fair as a strict comparison. WOFA reports BAR over 30 bits under OPA-derived
stolen masks and fused backgrounds; MBRS reports raw BER over 64 bits under fixed
rectangle retained masks. The paper does not report exact-message success, detection,
false-positive rate, ECC, or a complete denominator/aggregation contract. `BER=1-BAR`
is algebraically possible only as a descriptive transformation of the same WOFA
trial, not as an apples-to-apples MBRS result.

### Image quality

Not fair as a strict ranking. The paper-reported WOFA Table 3 row is 39.76 dB / 0.9945
/ 0.0188 at 200×200, 30 bits, OPA test data. MBRS's current formal operating points
are measured at 128×128, 64 bits, DIV2K fixed views, with explicit clipping and
aggregation. WOFA's paper does not freeze enough implementation detail to prove that
PSNR/SSIM/LPIPS reductions match the current evaluator. Its lower payload rate also
changes the visual-quality operating point.

### Crop robustness

A future adapter may test two separately named conditions:

1. `WOFA-native`: irregular stolen mask + new same-size background + geometry/channel
   settings from the paper.
2. `WOFA-rectangle-canvas probe`: 200×200 rectangular mask, explicit background, and
   retained-area labels 100/70/50/40/30 after declaring whether labels refer to stolen
   or retained area.

Neither should be merged into MBRS's formal `rectangle retained mask on original
canvas` table unless the model is actually evaluated with the exact same input size,
payload, fill/background, quantization, and decoder semantics.

## Position relative to existing external baselines

| Candidate | Relevance to partial retention | Executability in current project | Recommended role |
|---|---|---|---|
| TrustMark-Q/P | Moderate; partial masks but different ECC/preprocessing | Official inference completed | Strong external reference only; existing project already labels it `REFERENCE`. |
| HiDDeN, 64-bit retrained | Lower native focus on partial theft, but unified 64-bit evaluation exists | Executable on fixed manifest; explicitly a reimplementation, not official pretrained | Current unified external baseline/reference for the controlled table. |
| WOFA | **Highest conceptual relevance**: partial theft + unknown background + geometry | **Blocked end-to-end** by missing Stage-I weights and incomplete release | Official external reference; paper-reported reference until complete release. |

### Is WOFA better than HiDDeN as the main external baseline?

There are two different answers:

- **Research-question relevance:** yes, WOFA is a better-matched method to cite and
  discuss than HiDDeN.
- **Current paper's reproducible primary numeric baseline:** no. The 64-bit HiDDeN
  reimplementation has a declared unified contract and completed run, whereas WOFA
  has no valid message decode in the released bundle.

Therefore WOFA should be added beside HiDDeN as the method-specific external
reference, not substituted into the strict controlled table.

## WOFA + MBRS local-tail loss decision

**Not recommended in this audit.** A local-tail experiment would require choosing a
new WOFA 30-bit/200×200 training recipe, resolving missing Stage-I artifacts, deciding
whether local loss applies to RGB residuals or the intermediate pattern, and defining
a new crop/background/geometry evaluation. That would measure a new hybrid method,
not a faithful fine-tune of the paper's released system. Wait for complete official
training code and verifiable full checkpoints before considering a controlled
fine-tuning study.
