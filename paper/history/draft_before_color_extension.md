# Overlapping Patch-Tail Supervision for Local Distortion Control in Crop-Robust Watermarking

Draft v1 — evidence frozen from eight existing seed17 continuation checkpoints. This is a complete research draft, not a submission-formatted manuscript. All numerical tables below use the paper-freeze-v1 evaluation contract. The main configuration was fixed before this freeze; no new training or model selection was performed. Figure locations and production status are specified in [figure_plan.md](figure_plan.md).

## 1. Abstract

Robust image watermarking must balance message recovery against image distortion. A global image mean-squared-error objective constrains average distortion but does not explicitly supervise its spatial upper tail. We study a simple intervention in a crop-trained MBRS encoder–decoder: overlapping patch-level hard-tail supervision. The method ranks 16×16 patches with stride 8 by their raw reconstruction MSE and averages the highest-scoring 10%, combined equally with global image MSE. We isolate its effect through seed17 continuations from the same pretrained checkpoint, restoring Adam and BatchNorm state and keeping training duration and crop protocol fixed. On 50 fixed image–message pairs, the method increases global PSNR by 0.179 dB and Top25 local PSNR by 0.232 dB, while reducing mean per-image P95 patch MSE by approximately 5.70%. Gini decreases on 98% of images, indicating a change in relative spatial concentration alongside the reduction in absolute error. BER at 30% retained area changes from 0.113125 to 0.112938, with small changes at the other evaluated crop levels. SSIM, three-scale MS-SSIM, and full-image LPIPS improve, whereas the local LPIPS tail remains mixed. These results support local pixel-tail control within a controlled MBRS setting, without establishing human perceptual superiority or strict superiority over external watermarking systems.

## 2. Introduction

An image watermark should remain recoverable after image modification while preserving the appearance of the host image. Learning-based watermarking systems train an encoder to embed a message and a decoder to recover it through an attack layer [CITATION NEEDED]. Robustness imposes a competing requirement on fidelity: an embedding that survives partial content removal must remain decodable from less visual information. This motivates examining both global distortion and its spatial distribution, rather than treating a single image-quality average as a complete description of embedding fidelity.

Global reconstruction MSE is a useful image objective because it is inexpensive, differentiable, and directly related to PSNR. However, images with the same average squared error can distribute that error differently. A small number of high-error regions may coexist with a low image-wide average. This is an objective-design limitation, not evidence that crop training necessarily increases relative spatial concentration. We therefore ask a narrower question: can explicit supervision of locally high pixel distortion reduce the upper tail while preserving crop recovery in an otherwise fixed watermarking pipeline?

We investigate an overlapping hard-tail loss using the existing MBRS encoder–decoder. At each update, raw patch MSE determines which patches receive additional supervision. The selected errors, rather than a separate perceptual or content score, form the local objective. Fine patches and overlap provide candidate regions at a finer spatial granularity than the non-overlapping grid used for evaluation. The resulting method changes only training supervision and adds no inference module.

The main comparison is between a global-MSE continuation and an equal-duration local-tail continuation from the same seed17 crop-trained source. This control separates the local-loss effect from the benefit of additional training. We also examine existing non-overlap, overlap, selection-ratio, soft-weighting, multi-scale, excess-risk, and content-aware configurations. The method is evaluated through absolute local errors, normalized concentration statistics, message BER, and independent perceptual measurements.

Our contributions are threefold:

1. A precisely specified overlapping patch-tail objective that reduces local pixel-error tails in a controlled crop-trained MBRS continuation.
2. An image-paired evaluation separating absolute fidelity, relative concentration, and bit recovery, with explicit training-versus-evaluation patch definitions.
3. Evidence that improved pixel-tail control does not imply uniformly improved perceptual tails; released external models are reported as reference operating points with their protocol differences exposed.

The novelty claim is limited to the tested application and controlled evidence. We do not claim that top-k risk objectives themselves are new; the relationship to empirical tail-risk optimization requires appropriate prior-work attribution [CITATION NEEDED].

## 3. Related Work

### 3.1 Robust deep watermarking

Deep watermarking jointly learns image embedding and message extraction under differentiable or simulated distortions. HiDDeN, MBRS, StegaStamp, and TrustMark provide relevant contexts for learned embedding, attack simulation, and fidelity–robustness trade-offs [CITATION NEEDED]. The present study uses the repository's MBRS message-processing encoder and decoder, rather than introducing a new backbone. Claims about architectural ancestry and differences from the original publication should be checked against the corresponding papers before submission [CITATION NEEDED].

### 3.2 Crop robustness

Partial removal differs from small additive perturbations because it removes some embedded evidence entirely. Robustness to this operation depends on message distribution, learned decoding, and the exact treatment of removed pixels [CITATION NEEDED]. Our experimental operation retains a rectangle on the original canvas and masks the exterior. We distinguish this from cropping the rectangle into a smaller image and resizing it: the latter changes geometry and is outside the demonstrated claim.

### 3.3 Image fidelity and local distortion

PSNR measures reconstruction error, whereas SSIM, MS-SSIM, and LPIPS capture different structural or learned feature discrepancies [CITATION NEEDED]. A spatial tail objective and a perceptual objective are not interchangeable. Hard top-k aggregation is related to empirical upper-tail risk; concentration statistics describe inequality in an error distribution [CITATION NEEDED]. Our evaluation uses these ideas to test whether improvements persist beyond an image average, while checking perceptual tails independently. A lower concentration statistic alone does not establish better fidelity or human invisibility.

## 4. Method

### 4.1 Encoder, message, and decoder

Let an RGB host tensor be \(x\in[-1,1]^{3\times128\times128}\) and its binary message be \(m\in\{0,1\}^{64}\). The encoder produces \(\hat x=E_\theta(x,m)\). The implementation reshapes the message into an 8×8 spatial array, expands its features, combines them with image features, and predicts a three-channel image through a final convolution. The decoder maps an attacked output to 64 continuous message estimates. The experiment uses the non-diffusion encoder–decoder and message MSE; it does not add an adversarial discriminator to this controlled trainer.

The message objective is

\[
\mathcal L_{\mathrm{msg}}=\frac1{64}\sum_{b=1}^{64}
\big[D_\phi(A(\hat x))_b-m_b\big]^2.
\]

At evaluation, a decoded entry is classified as 1 if it exceeds 0.5.

### 4.2 Crop training

The repository's `RandomCrop(0.3,1.0)` draws height and width fractions independently and uniformly between \(\sqrt{0.3}\) and 1, rounds them to pixel dimensions, and samples a rectangle position. The attacked image is \(A(\hat x)=M\odot\hat x\), where the rectangle has mask value 1 and the exterior 0. The output canvas remains 128×128. A zero in the normalized domain corresponds to mid-gray after display conversion.

The retained area is therefore not uniformly distributed between the endpoint areas. We retain this exact implementation for every controlled branch. It is important to specify these semantics rather than infer them from the attack class name.

### 4.3 Global image objective

The baseline uses

\[
\mathcal L_g=\frac1{3HW}\|\hat x-x\|_2^2.
\]

This penalizes every residual entry equally. It constrains total squared error but does not directly specify a high-error regional subset. A decrease in this objective can improve many regions uniformly without necessarily changing the normalized spatial error distribution.

### 4.4 Hard local-tail supervision

Extract valid 16×16 patches with stride 8 from the unmodified encoder output and host. At 128×128, the grid has \(N=((128-16)/8+1)^2=225\) patches. For patch \(i\), define

\[
e_i=\frac1{3p^2}\sum_{c,u,v\in P_i}
(\hat x_{cuv}-x_{cuv})^2,
\qquad p=16.
\]

For each image independently, let \(\mathcal I\) contain the \(k=\lceil0.10N\rceil=23\) highest-MSE patches. The loss is

\[
\mathcal L_t=\frac1k\sum_{i\in\mathcal I}e_i,
\qquad
\mathcal L=10\mathcal L_{\mathrm{msg}}+
0.5\mathcal L_g+0.5\mathcal L_t.
\]

The Global continuation instead uses \(10\mathcal L_{\mathrm{msg}}+\mathcal L_g\). Training MSE is evaluated on the normalized, unclipped tensors. Reporting clipped display-domain metrics does not retroactively change that loss scale.

Selection operates on raw patch MSE. There is no gradient-activity normalization, color-space transform, or perceptual model in the main selector. Gradients propagate through the selected error values; the top-k membership is piecewise constant, with possible changes at ranking boundaries. Overlapping selected patches can cover the same pixel more than once, increasing its contribution accordingly. Selecting 10% of patch indices does not mean selecting 10% of unique pixels.

### 4.5 Fixed design and cost

The final configuration is patch16, stride8, Top10%, and global/local weights 0.5/0.5. We examine its components through already completed controlled alternatives in Section 7. The module is used only for training. It requires patch reductions and selection over 225 scores per image but introduces no encoder or decoder parameters. We do not report an unmeasured training-time overhead or runtime improvement.

![Figure 1. Frozen method diagram.](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig01_method.png)

*Figure 1. Encoder–attack–decoder flow with training-only image objectives. The local term compares clean encoded patches against host patches, selecting 23 of 225 candidate errors. The evaluation grid is separate. Both methods share the source and continuation protocol.*

## 5. Experimental Setup

### 5.1 Data and fixed manifest

The project uses 800 DIV2K training images. It divides the 100 public DIV2K validation images into a 50-image development partition and a 50-image project test partition [CITATION NEEDED]. The latter is not the official DIV2K test set. The image loader resizes to 140×140 and crops to 128×128. We reuse the persisted formal tensors and their fixed 64-bit messages; we do not reconstruct test views from guessed filenames or choose new crops.

The base manifest SHA-256 is `36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339`. An existing extension supplies the crop40 masks. The manifest contains 50 image–message pairs and five fixed attack repeats. Qualitative examples retain predetermined manifest indices 7, 20, and 42. In the actual-RGB observation panels, zoom regions are selected from reference-image variance, independently of comparative model improvement.

### 5.2 Controlled continuation

All eight rows begin from the same seed17 crop-trained Global epoch100 checkpoint, whose SHA-256 is `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`. They restore model parameters, BatchNorm state, and Adam moments, reset the continuation learning rate to \(10^{-4}\), and train for 20 epochs with batch size 16 and message-loss coefficient 10. The endpoint is always continuation epoch20. The controlled runner uses one visible GPU, deterministic algorithm settings, sorted file order, explicit loader generators, and matched random initialization of data/message/attack streams.

Configuration, checkpoints, and logs support these controls. Complete per-step RNG histories were not retained, so they do not prove equivalence between arbitrary interrupted and uninterrupted executions. Historical full-scratch, duplicate, and additional-seed runs are excluded from the paper comparison.

### 5.3 Frozen quality measurements

All quality measurements first map host and output to \([0,1]\) RGB and clip out-of-range values. Quality is evaluated on clean watermarked images. Let \(\mu_n\) be the resulting RGB MSE for test image \(n\). Global PSNR is

\[
\mathrm{PSNR}=10\log_{10}\!\left(\frac1{\frac1{50}\sum_n\mu_n}\right).
\]

Full SSIM uses `pytorch_msssim`, a Gaussian 7×7 window, sigma 1.5, range 1, and valid filtering without zero padding. Three-scale MS-SSIM uses a 7×7 window and explicitly fixed scale weights \([0.3,0.3,0.4]\); it is not the conventional five-scale setting. Full-image and local LPIPS use AlexNet LPIPS v0.1; clipped RGB is remapped to \([-1,1]\) for that model [CITATION NEEDED].

Local pixel and perceptual evaluation uses **32×32 non-overlapping patches**, giving 16 patches per image. It does not reuse the 16×16 training patches. Denote the descending evaluation MSE scores by \(q_{n,(1)}\ge\cdots\ge q_{n,(16)}\). Define

\[
t_n^{25}=\frac14\sum_{j=1}^4q_{n,(j)},\qquad
\mathrm{Top25LocalPSNR}=\frac1{50}\sum_n10\log_{10}(1/t_n^{25}).
\]

This is the current evaluator's mean per-image local PSNR, not PSNR from pooled tail MSE and not single-worst-patch PSNR. P95 patch MSE means the average of the per-image 95th percentiles, using linear interpolation across 16 patch scores. We consistently use this aggregation throughout the draft. Because global and local PSNR aggregate differently, subtracting their improvements is descriptive rather than a matched-fidelity causal estimate.

For perceptual tails, Top10% means \(\lceil16\times0.10\rceil=2\) native32 patches, an effective 12.5% of this finite grid. LPIPS selects its own highest two scores; SSIM selects its own lowest two. Local SSIM uses a 5×5 valid Gaussian window. These patches are not selected by MSE. Earlier 16→32 resized LPIPS proxies and legacy normalized-domain SSIM values are excluded from the formal tables.

### 5.4 Concentration measurements

For each image, let \(q_1,\dots,q_N\) be its nonnegative native32 patch MSE scores, with \(N=16\) and mean \(\bar q\). We compute population CV, \(\mathrm{std}(q)/\bar q\), and

\[
G=\frac{\sum_{i=1}^N\sum_{j=1}^N|q_i-q_j|}{2N\sum_iq_i}.
\]

For \(k=2\), define Top10/Mean as the average of the largest \(k\) errors divided by \(\bar q\), and Top10 energy share as their sum divided by the total error. Each statistic is computed per image and then averaged. The two tail-concentration measures obey

\[
\mathrm{EnergyShare}=\frac{k}{N}\,\mathrm{Top10/Mean}=0.125\,\mathrm{Top10/Mean}.
\]

They provide interpretable alternative scales, not independent corroborating tests. Gini and CV are invariant to uniform positive rescaling of all patch errors in one image. Their changes therefore complement, but cannot replace, absolute tail-error measurements [CITATION NEEDED].

### 5.5 Robustness measurements and provenance

Evaluation masks retain nominal areas 100%, 70%, 50%, 40%, and 30%. Partial crops use square sides \(\lfloor128\sqrt a\rfloor\), so actual retained area is slightly below its nominal label. Each saved position is reused across methods and across the original batch grouping. There is **no crop-and-resize-back interpolation** in MBRS's attack operation.

We replay the established raw normalized-output protocol, \(D_\phi(M\odot\hat x)\), without display clipping or PNG quantization. BER is the total number of erroneous bits divided by \(50\times5\times64\). Thus quality evaluates display-domain fidelity, while robustness evaluates the frozen tensor decoding protocol. Saved-image deployment robustness is not claimed by this experiment.

The paper freeze re-evaluated the eight existing checkpoints, recomputed 400 image-level quality records, and replayed their crop bit counts. The latter match the original controlled counts. [evidence_manifest.json](evidence_manifest.json) binds tables, evaluator source, manifests, and checkpoint hashes. This current binding does not reconstruct unrecorded historical environments.

## 6. Main Results

Table 1 reports the full controlled comparison in [main_table.md](main_table.md). The primary pair is reproduced here. Bold identifies the fixed pair in the full table rather than claiming that the main method wins every metric.

| Method | PSNR ↑ | SSIM ↑ | 3-scale MS-SSIM ↑ | Full LPIPS ↓ | Top25 local PSNR ↑ | P95 patch MSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Global continuation | 36.263172 | 0.950759 | 0.983036 | 0.002349598 | 35.522213 | 0.000301653 |
| Hard patch16 / stride8 / Top10% | 36.442300 | 0.952541 | 0.983690 | 0.002215177 | 35.754376 | 0.000284455 |

The main method gains 0.179128 dB in global PSNR and 0.232163 dB in Top25 local PSNR. Mean per-image P95 patch MSE decreases by 0.000017198, approximately 5.70%. Local PSNR improves on 96% of test images and P95 MSE decreases on 96%. This supports a consistent pixel-tail effect in the evaluated sample rather than a change driven only by the three displayed examples.

| Nominal retained area | Global BER ↓ | Hard patch16 / stride8 / Top10% BER ↓ | Difference |
|---|---:|---:|---:|
| 100% | 0.0000000 | 0.0000000 | 0.0000000 |
| 70% | 0.0000000 | 0.0000000 | 0.0000000 |
| 50% | 0.0014375 | 0.0015000 | +0.0000625 |
| 40% | 0.0416250 | 0.0418125 | +0.0001875 |
| 30% | 0.1131250 | 0.1129375 | −0.0001875 |

Robustness is essentially preserved in the descriptive sense of small observed absolute BER changes. At 50% and 40%, the method incurs one and three additional bit errors respectively among 16,000 evaluated bits; at 30%, it has three fewer. Repeated masks and bits are not independent image samples, and these counts do not establish statistical equivalence. At equal measured BER at 70% retained area, the method gives a better internal fidelity operating point. At 40% and 50%, the points exhibit small robustness–quality trade-offs rather than strict Pareto dominance.

*Figure 2. Predetermined image indices 7, 20, and 42, shown in that order. Each existing panel compares Original, Global continuation, and Hard Top10 in full RGB and common low-/high-variance reference-image regions. Display uses clipping, 8-bit rounding, and nearest-neighbor enlargement; no residual amplification or sharpening is applied. These examples are illustrative, not human-subject evidence.*

Existing Fig. 2 panels: [7](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/actual_rgb_test_07.png), [20](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/actual_rgb_test_20.png), [42](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/actual_rgb_test_42.png).

![Figure 3. Frozen quality and crop BER.](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig03_quality_robustness.png)

*Figure 3. Global and Top25 local PSNR on clean images, followed by raw BER under the shared rectangle masks. Lines connect retained-area conditions to aid reading; they do not represent fitted functions or confidence intervals.*

## 7. Ablation Study

Table 2 is [ablation_table.md](ablation_table.md); all rows are completed continuations from the same source. Soft T=0.25 is the previously selected Soft comparator on pixel-tail criteria. It is not claimed to be optimal for every metric.

**Overlap.** Holding patch16 and Top25% fixed, moving from stride16 to stride8 increases local PSNR from 35.716045 to 35.747490 dB and decreases P95 MSE from 0.000287625 to 0.000285606. Gini also decreases. This is a modest incremental effect consistent with finer candidate placement, not proof that patch-boundary artifacts were the unique cause.

**Selection ratio.** With patch16 and stride8 fixed, Top10% slightly improves local PSNR over Top25%: 35.754376 versus 35.747490 dB. P95 MSE and Gini also decrease. However, Top25% has marginally better global PSNR, SSIM, MS-SSIM, and full LPIPS. The main method is therefore a fixed local-pixel-tail operating point, not a universal winner. We do not reopen the ratio decision based on the refreshed table.

**Soft weighting.** The soft comparator normalizes patch MSE by each image's detached mean, applies softmax at T=0.25, and detaches the weights before computing weighted raw error. Its local PSNR of 35.646696 dB is below the overlapping Hard Top10 configuration. This comparison changes both selection style and stride; it is not a pure hard-versus-soft attribution. The stride16 Hard Top25 row provides a closer grid-matched context, though their tail weighting still differs.

**Multi-scale.** The existing alternative combines patch16/stride16 and patch32/stride32 Top25% losses with weights 0.7/0.3 within the local term. Its local PSNR of 35.688070 dB falls below the main method. It attains lower full LPIPS than the main configuration, reinforcing the distinction between pixel-tail and perceptual criteria.

**Excess penalty.** This alternative penalizes squared normalized excess above an image's detached mean error, with threshold 1 and scale 3. It lowers Gini to 0.086731 but reduces global PSNR to 35.903160 dB and local PSNR to 35.210189 dB. This is an important counterexample: a more even spatial distribution can still accompany worse absolute distortion. Concentration should always be read jointly with MSE or fidelity metrics.

**Gradient-aware selection.** This alternative divides patch MSE by \(1+2G_{\mathrm{norm}}\) to rank patches, where original-image activity is robustly normalized per image. Selected raw MSE still supplies the loss. The frozen implementation uses luminance forward differences, not the earlier Sobel-based diagnostic definition. Its global/local PSNR trade-off is weaker than the fixed main method; its native local LPIPS tail is better. We treat it as an alignment ablation and do not combine numerical coefficients from incompatible historical activity definitions.

## 8. Residual Concentration Analysis

Absolute tail improvements may accompany a nearly uniform reduction in residual amplitude. We therefore measure relative concentration on the same native32 evaluation grid.

| Statistic ↓ | Global continuation | Hard patch16 / stride8 / Top10% | Images with lower statistic |
|---|---:|---:|---:|
| Gini | 0.095007 | 0.086897 | 98% |
| Population CV | 0.173718 | 0.159015 | 98% |
| Top10/Mean | 1.295702 | 1.270921 | 92% |
| Top10 energy share | 0.161963 | 0.158865 | 92% |

The joint decrease in absolute tail error and normalized concentration indicates that the effect is not explained solely by uniformly scaling all residual values. This evidence supports a distributional change under local supervision. It does not isolate a complete causal mechanism, prove that every image improves, or establish human perceptual salience.

We deliberately make no claim that crop training increases concentration relative to no-crop training. The controlled study holds crop training fixed and asks how explicit local-tail supervision changes the resulting image. The Excess ablation further shows why a concentration-only success criterion would be misleading.

![Figure 4. Residual concentration.](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig04_concentration.png)

*Figure 4. Mean per-image Lorenz curves and paired Gini values, computed from clipped-RGB MSE on native32 non-overlapping patches. Each scatter point is one fixed image. Points below the identity line have lower concentration under Hard Top10; 98% satisfy this condition. No legacy unbounded-tensor plot is relabeled as this result.*

## 9. Perceptual Analysis

Global SSIM increases by 0.001782, three-scale MS-SSIM by 0.000654, and full LPIPS decreases by 0.000134421. Per-image SSIM and MS-SSIM improve on 86% of images, and full LPIPS improves on 76%. These metrics provide modest full-image support, but should not be equated with a human-subject preference result.

The local evidence is more nuanced:

| Perceptual tail | Global continuation | Hard patch16 / stride8 / Top10% | Images improved |
|---|---:|---:|---:|
| Bottom10 native32 SSIM ↑ | 0.902273 | 0.905504 | 66% |
| Top10 native32 LPIPS ↓ | 0.001361274 | 0.001366144 | 62% |

Local SSIM improves in the average and on a majority of images. Local LPIPS decreases for 62% of images but increases slightly in its dataset mean; the paired median change is −0.000033119. The mean and median therefore convey different aspects of the distribution. We describe this as mixed local LPIPS evidence, not an overall local perceptual-tail improvement. Native32 patches also have limited context relative to the feature network's original operating setting.

The mismatch constrains the interpretation of residual concentration. A reduction in pixel-error inequality cannot by itself validate an improvement in perceived artifact salience. Reference-based RGB examples can expose small residual effects, but normal viewing conditions, observer variability, and subjective visibility were not formally measured. We exclude unsupported human imperceptibility claims.

![Figure 5. Pixel and perceptual tail changes.](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig05_pixel_perceptual.png)

*Figure 5. Signed per-image changes in P95 patch MSE and Top10 native32 LPIPS for all 50 images, in fixed manifest order. Negative values indicate improvement. Pixel-tail improvements are more consistent; LPIPS has both regressions and improvements, with a slightly positive mean change despite a majority of negative changes.*

## 10. External Reference Comparison

We report released TrustMark Q and P models as external references, using the same 50 host views and their previously saved outputs. Official source and checkpoint identities are recorded in the project audit [CITATION NEEDED]. The encoding strength is unchanged. Table 3, [external_reference_table.md](external_reference_table.md), uses the same quality aggregation as the internal table.

| Method | Comparison | PSNR ↑ | Top25 local PSNR ↑ | Gini ↓ | Full LPIPS ↓ |
|---|---|---:|---:|---:|---:|
| MBRS Global continuation | Internal control | 36.263172 | 35.522213 | 0.095007 | 0.002349598 |
| MBRS Hard patch16 / stride8 / Top10% | Internal control | 36.442300 | 35.754376 | 0.086897 | 0.002215177 |
| TrustMark Q | REFERENCE | 42.293755 | 39.707519 | 0.355392 | 0.000964749 |
| TrustMark P | REFERENCE | 48.107281 | 46.572266 | 0.196823 | 0.000304701 |

TrustMark achieves substantially higher measured fidelity. Its released Q/P models occupy different fidelity–crop-recovery operating points: P has higher fidelity, while Q has lower observed pre-ECC BER under partial retention. At nominal 50% retained area, Q/P packet BER is 0.051920/0.079400 and ECC exact-message success is 0.584/0.368. At 30%, packet BER is 0.190520/0.250120, while neither model recovers the complete expected message in the recorded trials. Individual-bit accuracy and complete-message success are different quantities.

TrustMark's official decoder exposes pre-ECC logits thresholded at zero. Its transmitted packet comprises 61 data bits, 35 BCH parity bits, and 4 schema bits; raw packet BER is measured over all 100. MBRS instead reports 64 uncoded bits. TrustMark operates through 8-bit PIL images and native decoder resizing, including bilinear input resizing; MBRS's frozen crop decoder receives a masked normalized floating-point tensor. The host views and exterior masks are shared, but payload, ECC, quantization, model training, and decoder preprocessing differ.

These reference points contextualize the design space; they are not evidence of strict MBRS superiority, an apples-to-apples raw payload BER ranking, or a universal Pareto frontier. Higher relative Gini in a high-fidelity external model is also not evidence that its absolute distortion or perceptual quality is worse. StegaStamp and HiDDeN have no verified executable pretrained result in the audited setup and receive no numerical claims here.

![Figure S1. External reference operating points.](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/figS1_external_reference.png)

*Figure S1. Clean PSNR and full LPIPS versus pre-ECC bit accuracy at nominal 70%, 50%, 40%, and 30% retained areas. Marker shape identifies the model and color identifies retained area. TrustMark points are REFERENCE only: 100 transmitted packet bits and native PIL preprocessing differ from MBRS's 64 raw bits and tensor decoding. PSNR uses the same dataset-MSE reduction in all rows. No cross-family Pareto-dominance boundary is inferred.*

## 11. Limitations

**Scope and sample size.** The evidence concerns one seed, one encoder–decoder family, one 128×128 representation, one 64-bit payload, and 50 fixed host/message pairs. It does not establish cross-seed superiority or generalization across datasets and backbones. Repeated crop masks are not additional independent images.

**Development exposure.** The project test partition has been repeatedly inspected during earlier diagnostics and method/ratio comparisons. It is held out from parameter updates, but cannot be presented as a pristine blind final test for the entire research history. The freeze prevents further outcome-based method selection; it cannot undo earlier exposure.

**Finite-grid and metric choices.** Native32 evaluation yields only 16 regional scores per image; Top10% uses two patches and P95 relies on interpolation. Training and evaluation grids intentionally differ. The MS-SSIM variant, local SSIM support, LPIPS context, clipping, and percentile aggregation must remain explicit. No additional metric was selected to replace an unfavorable result.

**Fidelity and robustness domains.** Quality uses clipped display-domain images, while the established internal BER protocol uses unquantized normalized outputs. We do not claim equivalent performance after image saving, quantization, physical cropping, resize-back, or other untested transformations.

**Attribution.** Improved global and local fidelity occur together. Scale-invariant concentration statistics provide complementary evidence, but no final-checkpoint matched-global-PSNR causal comparison is established. Existing nearest-PSNR early checkpoints cannot substitute for that claim.

**Perception and external fairness.** Human ratings were not collected; local LPIPS is mixed. External models differ in packet construction and preprocessing, so their reference comparison cannot support strict superiority statements. The results justify a focused study of pixel-defined local-tail control rather than state-of-the-art or universal perceptual claims.

**Historical reproducibility.** Current hashes identify present artifacts and the reevaluation sources. They do not reconstruct every library version or per-step RNG state used in prior training. Epoch20 model and optimizer states, resolved configurations, and full epoch logs remain available.

## 12. Conclusion

Overlapping hard supervision of high-MSE patches improves the measured local pixel-error tail in a controlled crop-trained MBRS continuation. The fixed patch16/stride8/Top10% configuration improves global fidelity and reduces normalized concentration with small observed changes in raw crop BER. Existing ablations show that absolute fidelity, concentration, and perceptual tails can favor different configurations. In particular, local LPIPS remains mixed despite improved pixel-tail measurements. The demonstrated contribution is therefore explicit local pixel-tail control under the stated MBRS protocol, with external systems providing reference fidelity–robustness context.

## Citation completion notes (editorial; remove before submission)

All `[CITATION NEEDED]` tokens are intentional placeholders, not fabricated references. Complete attribution for learned watermarking, the original MBRS architecture/protocol, HiDDeN, StegaStamp, TrustMark Q/P release correspondence, crop/partial-removal robustness, empirical top-k/tail-risk objectives, DIV2K, SSIM, MS-SSIM, LPIPS, and concentration statistics. Verify the actual claims against primary publications. A contribution/novelty check against local-risk watermarking literature remains necessary before submission. No bibliography entries or priority claims have been invented.
