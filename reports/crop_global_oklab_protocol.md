# Crop + global OKLab: bounded continuation protocol

This new experiment is authorized after draft v1. The existing manuscript and its chosen method stay archived as the preceding evidence freeze. Only seed17 is used.

## Question and fixed comparison

Can adding a global pixel-averaged OKLab distance to the crop-trained Hard MSE patch16/stride8/Top10 method lower color error while preserving its quality/BER operating point?

New objective: `10 message_MSE + 0.5 global_RGB_MSE + 0.5 hard_patch_MSE + lambda mean_pixel_OKLab_distance`.

Global OKLab means mean **per-pixel** Euclidean color distance over the entire image, not distance between image-average colors. Chroma and signed a/b shifts are diagnostics only. The RGB and Hard terms retain the existing normalized-domain loss convention. The color transform receives clipped sRGB [0,1], performs inverse sRGB transfer, then the author's linear-sRGB→LMS→OKLab transformation. Color distance is `sqrt(dL²+da²+db²+1e-12)-1e-6`.

Both controls (Global and Hard Top10) already have epoch20 endpoints. The candidate starts from their same epoch100 crop-trained seed17 Global checkpoint, including Adam and BatchNorm state. It is not initialized from the stronger endpoint and given unfair extra training. It uses 20 epochs, LR1e-4, batch16, 800 training images, 50 validation images, deterministic single GPU, identical data-loader/message/crop RNG initialization. No JPEG training is used.

## Pre-outcome lambda calibration

Before training, use two fixed batches from the **training partition** and the source checkpoint in eval mode. Compare output-tensor gradient norms of the added color objective and the incumbent weighted image objective. Fix lambda so its color gradient norm is 25% of the image-objective norm (ratio of summed norms across the two batches). Save per-batch norms, image fingerprints, source hash and the resulting lambda. This is a loss-scale calibration, not a validation/test performance search; no optimizer step is taken.

## Validation gate and maximum scale

Validation uses the existing independent 50-image validation manifest and five frozen rectangle masks per retained-area condition, created once with a private local seed. Checkpoints and metrics are frozen at epoch20; no early-epoch model selection.

Evaluate against incumbent Hard Top10 on the same validation inputs. The operational (not statistical) preservation gate is:

- PSNR decrease at most0.10dB; Top25 local PSNR decrease at most0.10dB;
- SSIM decrease at most0.001;
- mean per-image P95 native32 MSE no more than2% higher; Gini no more than2% higher;
- crop30 and crop40 BER no more than0.001 higher; crop50 no more than0.0005 higher; crop70/100 no more than0.0003125 higher.

Target color benefit: global CIEDE2000 and mean Top10 sliding5×5 patch CIEDE2000 each at least5% lower, with a majority of images improving the latter. CIEDE2000 is computed independently in CIELAB D65 using skimage; it is never optimized.

Run one candidate first. If it meets both gates, stop tuning. If preservation fails, allow **one** validation-driven retry from the same source with half lambda. If preservation passes but color benefit is below5%, allow **one** retry with twice lambda. Maximum two full continuations; no new seed, patch, ratio, backbone, or objective family. Among candidates satisfying both gates choose the smaller validation Top10 color error; otherwise retain the incumbent and report failure/trade-off. Only then evaluate the selected candidate on the existing formal test manifest once. Formal results never trigger a third run. Success cannot be guaranteed by this procedure.

## Implementation correction discovered before training

The legacy `color_losses.py` used the author's XYZ→LMS matrix directly on linear RGB. That is not standard sRGB→OKLab. Its old JPEG run must not be treated as a valid test of standard OKLab; reported measured CIEDE2000 values remain measurements of those old checkpoints. Preserve legacy code/results and use a separately tested module for this experiment.

Official formula: [B. Ottosson, Oklab implementation](https://bottosson.github.io/posts/oklab/), specifically the linear-sRGB conversion, not the XYZ conversion. Reference tests cover white, gray, RGB primaries, zero/tint distances, independent CIEDE2000 behavior, finite black gradients, and gradcheck.
