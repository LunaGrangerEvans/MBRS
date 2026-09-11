# Controlling Local Distortion Tails in Crop-Robust Image Watermarking

*Evidence-based draft, 11 September 2026. The main method uses the frozen formal-test evidence; the OKLab extension uses validation evidence only. Reference placeholders identify literature still requiring bibliographic verification. Numerical sources and claim boundaries are indexed in [claim_evidence_map.md](claim_evidence_map.md).*

## 1. Abstract

Crop-robust watermarking must balance message recovery from partial image content against embedding fidelity. Global reconstruction mean squared error (MSE) controls average distortion but does not explicitly constrain its spatial upper tail. We investigate a simple training objective for a crop-trained MBRS encoder–decoder: rank overlapping 16×16 patches at stride 8 by raw MSE and supervise the highest-scoring 10%, combined equally with global image MSE. In seed17 controlled continuations from a shared source checkpoint, the method increases global PSNR by 0.179 dB and Top25 local PSNR by 0.232 dB, and reduces mean per-image P95 patch MSE by 5.70% on 50 fixed test image–message pairs. Gini decreases on 98% of images, while observed crop BER changes remain small. Full-image SSIM, three-scale MS-SSIM, and LPIPS improve modestly, but the local LPIPS tail remains mixed. A separate validation-only extension adds global OKLab regularization, further reducing absolute image and color distortion with approximately preserved crop BER, at the cost of increased normalized concentration relative to the pixel-tail method. The results distinguish absolute distortion, spatial concentration, and perceptual/color quality, supporting controlled local pixel-tail supervision without establishing human imperceptibility or strict superiority over external systems.

## 2. Introduction

Image watermarking embeds a message that should remain recoverable after an image is modified. Robustness and fidelity are competing design objectives: the embedding must retain decodable information while limiting its effect on the host image [CITATION NEEDED]. Partial image removal makes this balance particularly relevant because only part of the embedded signal remains available to the decoder. We study fidelity within a crop-trained watermarking system, focusing on the spatial distribution of embedding distortion.

The image-wide average is an incomplete description of that distribution. Two watermarked images can have equal global MSE yet differ in the errors of their most distorted regions. A global MSE objective penalizes every residual value but does not explicitly constrain a selected set of high-error patches. This motivates local-tail supervision: improving average reconstruction is useful, but it need not control how much distortion remains near the upper end of the regional error distribution.

Our experiment holds the watermarking backbone and crop-training protocol fixed. We add a hard patch-tail objective to the existing MBRS encoder–decoder, using overlapping patches to obtain candidate regions at a finer spatial granularity than the evaluation grid. The selected patches are ranked by raw pixel MSE; no perceptual model or content normalization is used in the main selector. Comparing equal-duration continuations from the same source separates this intervention from the benefit of additional training.

We evaluate three distinct outcomes. Absolute local distortion is measured by tail MSE and local PSNR. Relative spatial concentration is measured by Gini, CV, and normalized tail-energy statistics. Structural and learned-feature metrics assess whether pixel-tail improvements also appear under other quality criteria. These measurements should not be collapsed into a single claim about human visibility. In particular, lower concentration need not imply lower absolute error or better perceptual quality.

Qualitative inspection also identifies small chromatic residuals in the watermarked outputs, motivating a lightweight color-aware extension. We evaluate global OKLab regularization as an addition to the fixed pixel-tail loss, not as a replacement for the formal main method. Its validation results expose a useful distinction: absolute color error can decrease even when normalized spatial concentration increases.

The contributions are:

1. **A local distortion-tail formulation of watermark fidelity.** We examine crop-trained watermarking through both global reconstruction and regional upper-tail distortion, rather than relying on an image-wide average alone.
2. **A simple overlapping hard patch-tail objective.** In the controlled MBRS setting, it reduces measured high-error local distortion while approximately preserving observed crop BER, without changing the backbone.
3. **Complementary evidence across distinct objectives.** Controlled ablations, concentration and perceptual analyses, and a validation-only OKLab extension show how absolute distortion, spatial concentration, and perceptual/color quality can improve together or exhibit trade-offs.

These contributions concern the tested formulation and evidence. Hard top-k aggregation itself is not claimed as a new optimization principle [CITATION NEEDED]. Nor does this study isolate a causal increase in distortion or concentration caused by adding crop training to an otherwise identical no-crop model. Crop-related fidelity cost motivates the question; the formal comparison measures how local supervision changes an already crop-trained system.

## 3. Related Work

### 3.1 Robust deep watermarking

Learned watermarking jointly optimizes an encoder and decoder through an image-distortion channel. HiDDeN, MBRS, StegaStamp, and TrustMark are relevant contexts for embedding design, robustness, and fidelity [CITATION NEEDED]. We use the project's MBRS message-processing encoder–decoder and investigate its image objective. The present implementation and controlled training protocol should be distinguished from all components and settings of the original MBRS publication [CITATION NEEDED].

### 3.2 Partial-removal robustness

Recovery after partial content removal depends on the embedding, decoder, and exact definition of the attack [CITATION NEEDED]. Retaining a rectangle on the original canvas is different from extracting and resizing that rectangle. Our conclusions apply to the specified rectangle-mask protocol; they do not establish robustness to every operation described as cropping.

### 3.3 Fidelity, local risk, and color

PSNR, SSIM, MS-SSIM, and LPIPS assess different aspects of image reconstruction [CITATION NEEDED]. Hard selection of high-error samples or regions is related to empirical upper-tail risk objectives [CITATION NEEDED]. Gini and normalized dispersion statistics instead summarize relative inequality [CITATION NEEDED]. We combine these views without assuming that one substitutes for the others.

Color-space regularization provides an additional way to express image discrepancies. OKLab defines a color representation with lightness and opponent-color coordinates, while CIEDE2000 measures color difference in CIELAB [CITATION NEEDED]. We use the former as an auxiliary differentiable training representation and the latter for independent evaluation. The manuscript does not equate either numerical measure with demonstrated observer preference.

## 4. Method

### 4.1 Crop-Robust Watermarking Backbone

Let \(x\in[-1,1]^{3\times H\times W}\) be a host image and \(m\in\{0,1\}^{64}\) its message, with \(H=W=128\). The encoder produces \(\hat x=E_\theta(x,m)\). The implementation processes image features and an expanded 8×8 message representation, concatenates the two feature streams, and predicts an RGB output. A decoder estimates the message from the attacked encoded image. No new backbone parameters or inference modules are introduced.

For a binary rectangle mask \(M\), the attack is \(A(\hat x)=M\odot\hat x\). Removed pixels are zero in the normalized tensor domain, corresponding to mid-gray after display conversion. The canvas remains 128×128, without crop-and-resize-back interpolation. Training draws height and width fractions independently between \(\sqrt{0.3}\) and 1 using the existing RandomCrop implementation; the retained area is consequently not uniformly sampled over its endpoint interval.

The message loss is

\[
\mathcal L_{\mathrm{msg}}=\frac1{64}\sum_{b=1}^{64}
\left(D_\phi(M\odot\hat x)_b-m_b\right)^2.
\]

At evaluation, a decoder score above 0.5 is classified as bit 1. Scores are not assumed to be sigmoid probabilities. The controlled trainer uses image and message objectives without adding an adversarial discriminator.

### 4.2 Limitation of Global Distortion Loss

The baseline image objective is

\[
\mathcal L_g=\frac{1}{3HW}\sum_{c,u,v}(\hat x_{cuv}-x_{cuv})^2.
\]

It constrains average squared reconstruction error but not a separately identified regional upper tail. Reducing \(\mathcal L_g\) may reduce errors throughout an image without substantially changing their relative distribution. Conversely, making the distribution more uniform does not guarantee a reduction in its average or upper-tail magnitude. We therefore supervise local errors directly and report absolute and normalized measurements separately.

### 4.3 Hard Local-Tail Supervision

For valid patch \(P_i\) of size \(p\times p\), define

\[
e_i=\frac1{3p^2}\sum_{c,(u,v)\in P_i}(\hat x_{cuv}-x_{cuv})^2.
\]

The main method uses \(p=16\) and stride \(s=8\), yielding
\(N=((128-16)/8+1)^2=225\) candidate patches. Let \(\mathcal I_k\) index the highest \(k=\lceil0.10N\rceil=23\) scores for each image. The local objective and total loss are

\[
\mathcal L_t=\frac1k\sum_{i\in\mathcal I_k}e_i,
\qquad
\mathcal L_{\mathrm{hard}}=10\mathcal L_{\mathrm{msg}}+
0.5\mathcal L_g+0.5\mathcal L_t.
\]

The Global continuation uses \(10\mathcal L_{\mathrm{msg}}+\mathcal L_g\). Both RGB losses operate on the original normalized tensors during training. The selector ranks raw MSE and the selected raw MSE values are optimized; ranking does not use perceptual scores or gradient activity.

Gradients flow through selected patch errors. Overlap lets candidate regions cross non-overlapping-grid boundaries, but also means a pixel can contribute to several selected patches. Selecting 10% of patch indices does not imply selecting exactly 10% of unique pixels. The objective is used only during training; inference retains the same encoder and decoder.

![Main method](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig01_method.png)

*Figure 1. Global reconstruction and overlapping raw-MSE tail supervision branch from the clean encoder output. The main local loss selects 23 of 225 training patches. The separate native32 evaluation grid is not used to define training selection.*

### 4.4 Color-Aware OKLab Extension

The extension retains the complete hard-tail objective and adds a global color term. Convert host and encoded tensors to clipped sRGB:

\[
C=\operatorname{clip}((x+1)/2,0,1),\qquad
\hat C=\operatorname{clip}((\hat x+1)/2,0,1).
\]

Let \(T\) denote the standard sRGB-to-OKLab transformation: inverse sRGB transfer, linear-sRGB-to-LMS conversion, a cube-root nonlinearity, and the LMS-to-OKLab matrix [CITATION NEEDED]. For pixel \(u\), set
\(\delta_u=T(\hat C_u)-T(C_u)=(\delta L_u,\delta a_u,\delta b_u)\). We use

\[
\mathcal L_{\mathrm{OK}}=\frac1{HW}\sum_u
\left(\sqrt{\delta L_u^2+\delta a_u^2+\delta b_u^2+\epsilon}-\sqrt\epsilon\right),
\qquad \epsilon=10^{-12},
\]

\[
\mathcal L_{\mathrm{extension}}=\mathcal L_{\mathrm{hard}}+
\lambda\mathcal L_{\mathrm{OK}}.
\]

This is the mean of pixel-wise color distances, not the distance between mean image colors, so opposite spatial color shifts do not cancel. The small offset makes identical-image distance zero. A linear continuation of the cube root below LMS=\(10^{-12}\) gives finite gradients near black. Chroma-only distance, using \(\delta a^2+\delta b^2\), is recorded diagnostically but is not a second loss.

The existing study calibrated \(\lambda=0.059149764\) using source-model output-gradient norms on two training batches, targeting an initial color-gradient norm of 25% of the weighted RGB image-objective norm. One subsequent validation-driven comparison halved it to \(0.029574882\). These are the two completed extensions reported here. Both start from the shared source checkpoint and retain the 20-epoch protocol. CIEDE2000 is evaluation-only, and the extension is not promoted to the formal main method.

## 5. Experimental Setup

### 5.1 Data and evidence partitions

The project uses 800 DIV2K training images and divides the 100 public DIV2K validation images into 50 development-validation and 50 project-test images [CITATION NEEDED]. The latter is not the official DIV2K test set. The training loader resizes to 140×140 and samples 128×128 views. Formal evaluation reuses persisted image tensors and fixed 64-bit messages rather than reconstructing views from filenames.

The eight-method main comparison uses 50 fixed formal-test image–message pairs. The OKLab extension uses a separate fixed 50-image validation manifest and its own shared crop masks; candidate formal-test evaluation was not opened because the complete validation acceptance gate was not met. Validation numbers are never substituted into the main test table. Historical test diagnostics informed prior research decisions, so the formal test partition cannot be presented as globally pristine or blind throughout development.

### 5.2 Controlled training history

All main-table methods and the two color extensions are completed seed17 continuations from the same crop-trained Global epoch100 checkpoint. Each restores model parameters, BatchNorm state, and Adam state, then continues for 20 epochs at learning rate \(10^{-4}\), batch size 16, and message-MSE coefficient 10. The final checkpoint is always continuation epoch20. The runner uses one visible GPU, deterministic settings, sorted file order, explicit loader generators, and matched initialization of data/message/attack random streams. No additional model training is part of preparing this manuscript.

Checkpoint/configuration records support these controls; full per-step RNG histories were not saved. The claim is therefore tied to the controlled artifacts, without asserting reconstructed identity of arbitrary interrupted and uninterrupted training histories.

### 5.3 Quality and tail metrics

All reported image-quality values follow the current frozen convention: clipped RGB in [0,1], evaluated on clean encoded images. With per-image MSE \(\mu_n\), global PSNR is

\[
\mathrm{PSNR}=10\log_{10}\left(1/\frac1{50}\sum_n\mu_n\right).
\]

Full SSIM uses a Gaussian 7×7 window, sigma 1.5, data range 1, and valid filtering. Three-scale MS-SSIM uses a 7×7 window and scale weights [0.3,0.3,0.4]; it is not the conventional five-scale setting. LPIPS uses AlexNet v0.1, with clipped RGB remapped to [-1,1] at its input [CITATION NEEDED].

Pixel-tail and concentration evaluation uses 32×32 non-overlapping patches, producing 16 patch MSE values per image. If their descending order is \(q_{n,(1)},\ldots,q_{n,(16)}\), define

\[
t_n^{25}=\frac14\sum_{j=1}^4q_{n,(j)},\quad
\mathrm{Top25MSE}=\frac1{50}\sum_nt_n^{25},\quad
\mathrm{Top25LocalPSNR}=\frac1{50}\sum_n10\log_{10}(1/t_n^{25}).
\]

P95 patch MSE is the mean of per-image 95th percentiles, using linear interpolation across the 16 scores. Thus local PSNR is averaged in dB, whereas global PSNR is computed from dataset-average MSE. Neither local metric is the single worst patch. Their improvement difference is descriptive, not a matched-global-PSNR causal estimate.

Local LPIPS and SSIM also use native32 patches. Top10 LPIPS averages the two highest scores; Bottom10 SSIM averages the two lowest, with each metric ranking its own patches. The ceiling-rounded two-of-sixteen selection is effectively 12.5% of this evaluation grid. Local SSIM uses a 5×5 valid Gaussian window. These definitions differ explicitly from training's 23-of-225 selection.

### 5.4 Concentration and color metrics

For one image with nonnegative patch errors \(q_i\), mean \(\bar q\), and \(N=16\), we compute

\[
G=\frac{\sum_{i,j}|q_i-q_j|}{2N\sum_iq_i},\qquad
\mathrm{CV}=\frac{\operatorname{std}_{\mathrm{population}}(q)}{\bar q}.
\]

Top10-to-Mean is the average of the two highest patch errors divided by \(\bar q\); energy share is their sum divided by total patch error. Each statistic is computed per image, then averaged. Since energy share equals \((2/16)\) times Top10-to-Mean, the two quantities are alternative representations, not independent evidence. Gini and CV are unchanged by uniform positive scaling of all patch errors [CITATION NEEDED].

For the validation-only color analysis, CIEDE2000 is computed per pixel after conversion to CIELAB with a D65 reference [CITATION NEEDED]. We average pixel distances within 5×5 sliding windows at stride 1, obtaining 15,376 scores per image. Their highest 1,538 scores form the Top10 color tail. Global mean, patch P95, and Top10 values are averaged across validation images. This color grid is not a change to the native32 pixel-tail evaluation or patch16 training objective. CIEDE2000 includes lightness as well as chromatic differences; chroma-only OKLab diagnostics provide complementary evidence about color components specifically.

### 5.5 Robustness and traceability

Formal evaluation uses nominal retained areas 100%, 70%, 50%, 40%, and 30%, with five fixed repeats. A partial-retention square has side \(\lfloor128\sqrt a\rfloor\); actual retained pixel area can be slightly below the nominal label. Masks and batch assignments are shared across compared methods. BER is the number of erroneous decoded bits divided by \(50\times5\times64\); repeated masks are not additional independent images.

The frozen internal BER protocol decodes masked normalized floating-point outputs without display clipping or PNG quantization. Consequently, clipped-RGB quality and tensor-domain robustness describe two explicitly documented parts of the existing evaluation. We do not claim that those BER values are already verified for saved-image deployment.

The numerical sources are [main_table.csv](main_table.csv), [paired_diagnostics.json](paired_diagnostics.json), the [frozen per-image records](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv), [OKLab validation CSV](../reports/crop_global_oklab_validation_candidates.csv), and [external_reference_table.csv](external_reference_table.csv). Main checkpoint and evaluator bindings are in [evidence_manifest.json](evidence_manifest.json). All table values below are readings or explicitly calculated differences from these artifacts; legacy metric conventions are excluded.

## 6. Main Results

Table 1 summarizes the eight completed formal-test configurations. All local variants use global/local image weights 0.5/0.5; Global uses 1/0. Bold identifies the fixed primary pair, not a claim that it wins every column.

| Configuration | PSNR ↑ | SSIM ↑ | 3-scale MS-SSIM ↑ | Full LPIPS ↓ | Top25 local PSNR ↑ | P95 MSE ↓ | BER30 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Global continuation** | 36.263172 | 0.950759 | 0.983036 | 0.002349598 | 35.522213 | 0.000301653 | 0.1131250 |
| Hard patch16 / stride16 / Top25% | 36.418147 | 0.952313 | 0.983600 | 0.002200142 | 35.716045 | 0.000287625 | 0.1131875 |
| Hard patch16 / stride8 / Top25% | 36.445329 | 0.952631 | 0.983713 | 0.002205026 | 35.747490 | 0.000285606 | 0.1131875 |
| **Hard patch16 / stride8 / Top10%** | 36.442300 | 0.952541 | 0.983690 | 0.002215177 | 35.754376 | 0.000284455 | 0.1129375 |
| Soft patch16 / stride16 / T=0.25 | 36.350297 | 0.951587 | 0.983356 | 0.002211002 | 35.646696 | 0.000291918 | 0.1133750 |
| Multi-scale Top25%: 0.7×patch16/stride16 + 0.3×patch32/stride32 | 36.391394 | 0.952069 | 0.983516 | 0.002198487 | 35.688070 | 0.000289416 | 0.1130625 |
| Excess patch16 / stride16 / threshold1 / scale3 | 35.903160 | 0.946346 | 0.981532 | 0.002351493 | 35.210189 | 0.000321119 | 0.1129375 |
| Gradient-aware patch16 / stride8 / Top10% / alpha2 | 36.332801 | 0.952017 | 0.983473 | 0.002199208 | 35.624195 | 0.000293327 | 0.1134375 |

Relative to Global continuation, the main method improves global PSNR by 0.179128 dB and Top25 local PSNR by 0.232163 dB. Mean Top25 patch MSE decreases from 0.000291832 to 0.000275957; mean per-image P95 MSE decreases by approximately 5.70%. Top25 local PSNR improves and P95 MSE decreases on 96% of formal-test images. The result therefore appears across the image sample rather than only in selected qualitative examples.

| Nominal retained area | Global BER ↓ | Hard patch16 / stride8 / Top10% BER ↓ | Change |
|---|---:|---:|---:|
| 100% | 0.0000000 | 0.0000000 | 0.0000000 |
| 70% | 0.0000000 | 0.0000000 | 0.0000000 |
| 50% | 0.0014375 | 0.0015000 | +0.0000625 |
| 40% | 0.0416250 | 0.0418125 | +0.0001875 |
| 30% | 0.1131250 | 0.1129375 | −0.0001875 |

The observed BER changes are small. At 70%, measured BER is equal and fidelity improves. At 40% and 50%, slightly higher BER accompanies better quality, so the method does not strictly dominate the control at every crop level. “Essentially preserved” describes these observed differences; it is not a statistical equivalence claim.

The [fixed RGB examples](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/artifact_observation_main.pdf) retain test indices 7, 20, and 42. Their regions are chosen from the original image's low/high luminance variance, not comparative model gains. The [quality–robustness figure](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig03_quality_robustness.png) presents the same frozen metrics. These visualizations illustrate the numerical comparison without establishing human-rated superiority.

## 7. Ablation Study

The [ablation table](ablation_table.md) provides changes under the same frozen metric definition. The comparisons below distinguish controlled factor changes from configurations that change several design choices together.

### 7.1 Patch Size

The authoritative eight-row table does not contain an isolated single-scale patch-size sweep with the other variables fixed. Patch16 is the chosen main configuration, but the available formal evidence does not establish it as an optimal patch size. Its gains are measured on a separate native32 evaluation grid, which shows that the result is not confined to reusing the training patch grid. The multi-scale row tests a combined objective and must not be relabeled as a pure patch-size ablation. Historical incompatible results are not imported to fill this gap.

### 7.2 Overlap

At fixed patch16 and Top25%, reducing stride from 16 to 8 increases local PSNR from 35.716045 to 35.747490 dB and lowers P95 MSE from 0.000287625 to 0.000285606. Gini also decreases. This modest gain is consistent with more flexible regional placement, but does not isolate patch boundaries as the unique cause.

### 7.3 Top-k Selection

For patch16/stride8, changing Top25% to Top10% increases local PSNR from 35.747490 to 35.754376 dB and further lowers P95 MSE and Gini. However, Top25% has marginally better global PSNR, SSIM, MS-SSIM, and full LPIPS. Top10% is therefore a fixed pixel-tail operating point, not the winner of every quality criterion or evidence of a universal optimal selection ratio.

### 7.4 Hard versus Soft

The soft alternative uses detached softmax weights applied to per-image mean-normalized patch MSE, at temperature 0.25. Its grid-matched stride16 comparison with Hard Top25% yields lower local PSNR: 35.646696 versus 35.716045 dB. Their effective spatial weights and support differ, so the result concerns these particular objectives. Comparing the soft row directly with the main method also changes stride and selection support; it cannot assign the entire difference to hard versus soft aggregation. Temperature 0.25 is the existing historical comparator, not a newly selected optimum.

### 7.5 Multi-scale

The multi-scale alternative combines non-overlapping patch16 and patch32 hard Top25% losses with local mixture weights 0.7/0.3. Local PSNR is 35.688070 dB, below the main method, while full LPIPS is slightly lower. Increasing the number of scales is not a uniform improvement in this completed comparison.

### 7.6 Excess

The Excess objective penalizes squared normalized error above a detached per-image mean, with threshold 1 and scale 3. It attains Gini 0.086731, slightly below the main method, yet global and local PSNR fall to 35.903160 and 35.210189 dB. This is a direct example of improved normalized concentration coexisting with worse absolute distortion. Concentration alone is an insufficient success criterion.

### 7.7 Gradient-aware Selection

The gradient-aware alternative ranks patch MSE by \(e_i/(1+2G_{i,\mathrm{norm}})\), using robust per-image normalization of original-image luminance-gradient activity. It still optimizes selected raw MSE. Its global/local PSNR improvements are smaller than the main method's, while its local LPIPS tail is lower. The row illustrates a pixel/perceptual trade-off. Earlier activity diagnostics used different feature definitions; their alignment coefficients are not combined with the frozen implementation's measurements.

## 8. Residual Concentration Analysis

Absolute error reductions do not automatically imply a change in relative spatial concentration. Table 3 reports normalized statistics on the same native32 non-overlapping grid.

| Statistic ↓ | Global continuation | Hard patch16 / stride8 / Top10% | Images with lower values |
|---|---:|---:|---:|
| Gini | 0.095007 | 0.086897 | 98% |
| Population CV | 0.173718 | 0.159015 | 98% |
| Top10-to-Mean | 1.295702 | 1.270921 | 92% |
| Top10 energy share | 0.161963 | 0.158865 | 92% |

Together with the absolute tail reductions, these results indicate a change beyond uniformly scaling all residual errors. They support reduced concentration in the controlled comparison, but do not identify a complete causal mechanism or guarantee better perceived quality. The dependence between Top10-to-Mean and energy share prevents treating them as two independent confirmations.

![Concentration analysis](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig04_concentration.png)

*Figure 2. Mean per-image Lorenz curves and paired Gini measurements from clipped-RGB native32 patch MSE. Each point corresponds to one fixed test image; points below the identity line have lower Gini under Hard Top10. The close Lorenz curves are retained at their actual scale.*

The study does not show that crop training causes residual concentration to increase. It instead holds the crop-trained source fixed and measures the effect of local-tail supervision. The Excess result and the color extension further demonstrate why relative concentration must be interpreted jointly with absolute error.

## 9. Perceptual Quality Analysis

Full-image SSIM and three-scale MS-SSIM increase by 0.001782 and 0.000654, while full LPIPS decreases by 0.000134421. SSIM/MS-SSIM improve on 86% of images and full LPIPS on 76%. These measurements provide modest full-image quality support.

The local outcome is mixed:

| Native32 perceptual tail | Global continuation | Hard patch16 / stride8 / Top10% | Images improved |
|---|---:|---:|---:|
| Bottom10 SSIM ↑ | 0.902273 | 0.905504 | 66% |
| Top10 LPIPS ↓ | 0.001361274 | 0.001366144 | 62% |

Local SSIM improves in its average and for a majority of images. For local LPIPS, 62% of images improve and the paired median change is negative, but the dataset mean increases slightly. Reporting only the improved-image fraction would conceal the regressions. The main method therefore does not establish an overall local LPIPS-tail improvement. Native32 LPIPS also evaluates limited image context and should not be interpreted as a validated human visibility test.

![Pixel and perceptual tail changes](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig05_pixel_perceptual.png)

*Figure 3. Signed per-image changes for all 50 fixed test images. P95 pixel error decreases on 96%, while Top10 LPIPS decreases on 62% with a slightly positive mean change. Improvements and regressions are both shown.*

The evidence does not validate residual concentration as a proxy for human perceptual salience. Small chromatic residuals in the fixed RGB examples motivate color-aware analysis, but neither those examples nor the measured metrics demonstrate improved human imperceptibility. No human-subject evaluation was conducted.

## 10. Color-Aware Extension Analysis

**All results in this section are validation-only.** They compare the incumbent Hard Top10 and two already completed global-OKLab continuations on the same fixed validation views and crop masks. The larger weight was calibrated on training data; its Gini regression prompted the single half-weight comparison. Both results are reported, and neither replaces the formal-test main method.

| Validation configuration | PSNR ↑ | Top25 local PSNR ↑ | Global CIEDE2000 ↓ | Top10 patch CIEDE2000 ↓ | Patch P95 CIEDE2000 ↓ | Gini ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Hard patch16 / stride8 / Top10%, no color term | 36.410613 | 35.652988 | 3.598625 | 5.230329 | 5.124218 | 0.094569 |
| Same hard objective + global OKLab, λ=0.029574882 | 36.650197 | 35.879871 | 3.465417 | 5.051264 | 4.946862 | 0.097303 |
| Same hard objective + global OKLab, λ=0.059149764 | 36.818150 | 36.032252 | 3.365012 | 4.920614 | 4.816130 | 0.100017 |

At the smaller weight, global/local PSNR improve by 0.239585/0.226884 dB; global and Top10 CIEDE2000 decrease by 3.70% and 3.42%. At the larger weight, the corresponding PSNR gains are 0.407537/0.379264 dB, and color-error reductions are 6.49% and 5.92%. Both extensions reduce global, Top10, and P95 CIEDE2000 on all 50 validation images. Chroma-only OKLab distance also decreases, from 0.01095297 without the color term to 0.01056424 and 0.01027138 at the smaller and larger weights. This supports a chromatic component to the measured gain rather than relying solely on total color distance, which also includes lightness.

| Validation configuration | SSIM ↑ | 3-scale MS-SSIM ↑ | Full LPIPS ↓ | BER50 ↓ | BER40 ↓ | BER30 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Hard Top10, no color term | 0.954982 | 0.984839 | 0.002140650 | 0.0158750 | 0.0435625 | 0.1110000 |
| + global OKLab, λ=0.029574882 | 0.957396 | 0.985650 | 0.001940944 | 0.0158750 | 0.0437500 | 0.1108125 |
| + global OKLab, λ=0.059149764 | 0.959187 | 0.986227 | 0.001858625 | 0.0158125 | 0.0438125 | 0.1108125 |

BER at 70% and 100% is zero for these three validation models. The other BER differences are small, and the full-image quality metrics improve. However, Gini rises by 2.89% and 5.76% relative to incumbent Hard Top10. CV, Top10-to-Mean, and energy share also rise. The color objective thus improves absolute fidelity while partially reducing the incumbent's concentration advantage. The smaller-weight model still has lower Gini than validation Global continuation, but this does not mean it preserves the full improvement of Hard Top10.

Neither candidate met the study's complete validation gate: both exceeded its 2% Gini-increase tolerance, and the smaller weight did not reach its 5% color-benefit target. These are operational criteria rather than statistical or perceptual thresholds. The extension's positive measurements remain useful, but should be described together with this limitation. There is also no matched-strength additional-RGB-regularizer control, so the benefit cannot be attributed exclusively to the geometry of OKLab rather than partly to stronger image regularization.

The [fixed validation RGB/heatmap comparison](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/crop_global_oklab/fixed_color_comparison.png) uses predetermined indices 0, 10, and 20, with a shared color-difference scale. It is illustrative, not an observer study. The result supports the qualified conclusion that **OKLab regularization can complement the pixel-tail objective by reducing absolute chromatic distortion in validation, while introducing a modest trade-off in normalized residual concentration**.

## 11. External Reference Comparison

Released TrustMark Q/P models provide reference fidelity–recovery operating points [CITATION NEEDED]. Table 6 uses the same frozen quality aggregation on the same formal host views, with the official encoding strength unchanged. **Every TrustMark–MBRS comparison is REFERENCE only.**

| Method | Comparison | PSNR ↑ | Top25 local PSNR ↑ | Gini ↓ | Full LPIPS ↓ |
|---|---|---:|---:|---:|---:|
| MBRS Global continuation | Internal control | 36.263172 | 35.522213 | 0.095007 | 0.002349598 |
| MBRS Hard patch16 / stride8 / Top10% | Internal main method | 36.442300 | 35.754376 | 0.086897 | 0.002215177 |
| TrustMark Q | REFERENCE | 42.293755 | 39.707519 | 0.355392 | 0.000964749 |
| TrustMark P | REFERENCE | 48.107281 | 46.572266 | 0.196823 | 0.000304701 |

TrustMark achieves substantially higher measured fidelity. Q/P also differ in recovery: at nominal 50% retained area, pre-ECC packet BER is 0.051920/0.079400 and ECC-corrected exact-message success is 0.584/0.368. At 30%, packet BER is 0.190520/0.250120, while neither model recovers the complete expected message in the recorded trials. P favors fidelity; Q has lower observed partial-retention packet BER.

TrustMark transmits 100 bits comprising 61 data bits, 35 BCH parity bits, and 4 schema bits. Its pre-ECC bits are obtained by thresholding official decoder logits at zero. MBRS instead reports 64 uncoded message bits. TrustMark uses 8-bit image outputs and native decoder resizing, whereas MBRS's frozen BER decodes masked normalized float tensors. Payload, ECC, quantization, decoder preprocessing, and training protocols therefore differ even though the host views and exterior-mask coordinates are shared.

The [reference operating-point figure](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/figS1_external_reference.png) keeps those distinctions explicit. Raw-bit accuracy, exact-message success, and the decode flag are different quantities; a successful flag is not a calibrated detection rate on unwatermarked images. Neither the table nor the figure establishes strict MBRS superiority or a universal Pareto frontier. Higher normalized Gini in a high-fidelity reference model also does not imply larger absolute distortion or worse human appearance.

## 12. Limitations

**Experimental scope.** The formal result covers seed17, one MBRS encoder–decoder family, 128×128 inputs, a 64-bit message, and 50 fixed test pairs. It does not establish cross-seed significance, broad generalization, or statistical equivalence of BER. The fixed table lacks an isolated patch-size ablation, and several cross-configuration comparisons change more than one design factor.

**Development exposure.** Earlier diagnostics and method comparisons examined the project test partition. Its separation from training updates does not make it a pristine blind test across the entire research history. The OKLab extension remains a development-validation observation without candidate formal-test confirmation.

**Metric and attack scope.** Training and evaluation patch grids differ intentionally; ceiling-rounded tails and per-image percentile aggregation must remain explicit. Three-scale MS-SSIM and native32 LPIPS have the specified support. Quality measures clipped RGB while BER uses unquantized normalized outputs. Robustness to physical cropping, resize-back, or saved-image deployment is not established by rectangle-mask BER.

**Attribution.** Global and local fidelity improve together; the study does not provide a final-checkpoint matched-PSNR causal separation. Normalized concentration provides complementary evidence, not a complete mechanism. Similarly, the OKLab extension adds regularization strength without a matched-strength RGB-only control. The current formal evidence does not isolate the incremental fidelity cost of crop training against a matched no-crop counterpart.

**Perception and external fairness.** Full-image quality metrics improve, but local LPIPS remains mixed and human ratings are absent. Observed chromatic residuals do not establish a systematic color-block-artifact detector or subjective salience benefit. TrustMark comparisons remain reference-only because of incompatible payload and pipeline details. We do not claim state of the art, universal imperceptibility, or strict superiority over external methods.

**Historical reproducibility.** Source paths, final checkpoints, configurations, epoch logs, and current hashes are retained. They do not reconstruct every historical library version or per-step random state. The color extension uses the separately tested standard-sRGB OKLab implementation; earlier incompatible color-conversion experiments are outside its evidence base.

## 13. Conclusion

Overlapping hard patch-tail supervision reduces measured local pixel distortion in a controlled crop-trained MBRS continuation while approximately preserving observed crop BER. The fixed patch16/stride8/Top10% method improves global fidelity, absolute local tails, and normalized spatial concentration. Full-image perceptual metrics provide modest additional support, but mixed local LPIPS results prevent equating concentration reduction with human imperceptibility. A validation-only global OKLab extension further lowers image and chromatic error while partially trading away normalized concentration gains. Together, the results support treating absolute distortion, spatial concentration, and perceptual/color quality as related but distinct objectives in robust image watermarking.
