# Controlling Local Distortion Tails in Crop-Robust Image Watermarking

## 1. Abstract

Crop-robust watermarking must preserve message recovery after partial image removal while limiting its fidelity cost. Global RGB mean-squared error controls average distortion but does not explicitly supervise its spatial upper tail. We study a simple hard local-tail objective for a crop-trained MBRS encoder–decoder. The method ranks overlapping 16×16 patches with stride 8 by raw MSE and averages the highest 10%, combined equally with global image MSE. In seed17 controlled continuations from a shared source checkpoint, it improves PSNR by 0.179128 dB and Top25 local PSNR by 0.232163 dB over Global continuation. Mean per-image P95 native32 patch MSE decreases from 0.000301653 to 0.000284455, while Gini decreases on 98% of fixed test images. Crop BER changes remain small. SSIM, three-scale MS-SSIM, and full-image LPIPS improve modestly, but local LPIPS remains mixed. A validation-only global OKLab extension further reduces absolute color distortion while slightly weakening normalized concentration. Results support local pixel-tail control in the tested MBRS setting without establishing human perceptual superiority or strict superiority over external systems.

## 2. Introduction

Learned image watermarking must balance message recovery and host-image fidelity [CITATION NEEDED]. Crop robustness makes this balance especially visible because the decoder must recover the message from less spatial evidence after partial image removal.

Global RGB MSE is differentiable and useful for image fidelity, but it averages error over the image. Two outputs can have similar global MSE while differing substantially in their most distorted regions. Global MSE therefore does not explicitly control the local distortion tail.

We study this issue in the existing crop-trained MBRS encoder–decoder. The intervention is intentionally small: retain the backbone and crop channel, rank fine-grained overlapping patches by raw MSE, and add the highest-error patches to the image objective. The main configuration uses Patch16, stride8, Top10%, and global/local image-loss weights 0.5/0.5. It is compared with an equal-duration Global continuation from the same source checkpoint.

This paper makes three contributions. First, it formulates crop-robust watermark fidelity from a local distortion-tail perspective rather than relying only on global average distortion. Second, it introduces a simple hard patch-tail objective that suppresses highly distorted local pixel regions while approximately preserving crop BER. Third, it analyzes residual concentration, perceptual metrics, controlled variants, and a color-aware OKLab extension, showing that absolute distortion, spatial concentration, and perceptual/color quality are related but distinct objectives. Hard top-k optimization itself is not claimed as a new general principle [CITATION NEEDED].

## 3. Method

### 3.1 Crop-Robust Watermarking Backbone

Let x be a normalized 128×128 RGB host image and m a 64-bit message. The encoder produces the clean encoded image y=E(x,m). A rectangle mask M forms the decoder input z=M⊙y. The decoder outputs 64 scores, which are thresholded at 0.5 for bit evaluation. The crop channel retains the original canvas and masks pixels outside a randomly placed rectangle.

The message objective is

L_msg = (1/64) sum_b [D(M⊙y)_b − m_b]^2.

The backbone and decoder are unchanged. The proposed method only changes the training image objective.

### 3.2 Limitation of Global Distortion Loss

The Global continuation uses

L_global = (1/(3HW)) ||y−x||².

This controls average RGB error but does not identify a high-error regional subset.

### 3.3 Hard Local-Tail Supervision

For a 16×16 patch P_i with stride 8, define raw patch MSE

e_i = (1/(3|P_i|)) sum_(c,u,v in P_i) (y_cuv−x_cuv)².

The 128×128 image produces 225 overlapping candidates. We select k=ceil(0.10×225)=23 patches with the largest e_i values and define

L_tail = (1/k) sum_(i in Top10(e)) e_i.

The main training loss is

L = 10 L_msg + 0.5 L_global + 0.5 L_tail.

Selection uses raw MSE only. The local branch is used during training and adds no inference module. Overlap means that 10% of patch indices is not exactly 10% of unique pixels.

### 3.4 Color-Aware OKLab Extension

The extension retains the complete Hard Top10 objective and adds a global per-pixel OKLab distance:

L_OK = (1/HW) sum_u [sqrt(ΔL_u²+Δa_u²+Δb_u²+ε)−sqrt(ε)].

L_ext = L + λ L_OK.

The transform receives clipped sRGB values and uses the standard linear-sRGB OKLab conversion [CITATION NEEDED]. CIEDE2000 is evaluation-only. The two completed validation weights are λ=0.059149764 and λ=0.029574882. Neither is promoted to the formal main method.

## 4. Experiments and Results

### 4.1 Setup

The project uses 800 training images and two fixed 50-image partitions from its DIV2K-based split [CITATION NEEDED]. Inputs are 128×128 with 64-bit messages. Controlled continuations use seed17, a shared crop-trained Global epoch100 source, restored Adam/BatchNorm state, batch size 16, learning rate 10⁻⁴, and 20 epochs. The training attack is the existing RandomCrop(0.3,1.0) channel.

Quality uses the current clipped-RGB evaluator: PSNR, SSIM, three-scale MS-SSIM, full-image LPIPS, native32 patch MSE, and native32 concentration. Robustness uses raw 64-bit BER under fixed rectangle masks at nominal retained areas 100%, 70%, 50%, 40%, and 30%. Legacy metric values are excluded.

### 4.2 Main Controlled Result

| Method | PSNR | SSIM | MS-SSIM | Full LPIPS | Top25 local PSNR | P95 MSE | Gini | BER50 | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Global continuation | 36.263172 | 0.950759 | 0.983036 | 0.002349598 | 35.522213 | 0.000301653 | 0.095007 | 0.0014375 | 0.1131250 |
| **Hard Patch16 / stride8 / Top10%** | **36.442300** | **0.952541** | **0.983690** | **0.002215177** | **35.754376** | **0.000284455** | **0.086897** | **0.0015000** | **0.1129375** |

Relative to Global, the main method improves PSNR by 0.179128 dB and Top25 local PSNR by 0.232163 dB. Mean per-image P95 patch MSE decreases by approximately 5.70%. Gini decreases on 98% of test images. BER changes are small: at 50% retained area it changes from 0.0014375 to 0.0015000, and at 30% from 0.1131250 to 0.1129375. This supports approximate preservation of observed crop robustness, not statistical equivalence at every crop ratio.

### 4.3 Key Ablation Results

| Variant | ΔPSNR | ΔTop25 local PSNR | ΔP95 MSE | ΔGini | ΔBER30 |
|---|---:|---:|---:|---:|---:|
| Hard Patch16 / stride16 / Top25% | +0.154975 | +0.193832 | −0.000014029 | −0.005976 | +0.0000625 |
| Hard Patch16 / stride8 / Top25% | +0.182157 | +0.225278 | −0.000016048 | −0.006403 | +0.0000625 |
| **Hard Patch16 / stride8 / Top10%** | **+0.179128** | **+0.232163** | **−0.000017198** | **−0.008110** | **−0.0001875** |
| Soft patch16 / stride16 / T=0.25 | +0.087124 | +0.124483 | −0.000009735 | −0.005915 | +0.0002500 |

Overlap improves Hard Top25 relative to stride16. Top10 gives the strongest local pixel-tail result among these Hard variants, while Top25 has slightly better global and perceptual averages. The Soft comparator improves the tail less than the main Hard configuration. The current evidence does not contain an isolated patch-size sweep; Patch16 is therefore not claimed to be globally optimal. Multi-scale, Excess, and Gradient-aware variants are supporting ablations: Excess shows that lower concentration can coexist with worse absolute fidelity, and Gradient-aware selection gives a different pixel/perceptual trade-off.

### 4.4 Residual Concentration and Perceptual Quality

The main method lowers Gini from 0.095007 to 0.086897, CV from 0.173718 to 0.159015, Top10-to-Mean from 1.295702 to 1.270921, and energy share from 0.161963 to 0.158865. These normalized statistics are interpreted jointly with absolute P95 and Top25 MSE; lower concentration alone is not a perceptual claim.

Full-image SSIM, MS-SSIM, and LPIPS improve modestly. Local Bottom10 SSIM improves on 66% of images. Local Top10 LPIPS improves on 62%, but its mean changes slightly upward from 0.001361274 to 0.001366144. Local LPIPS therefore remains mixed.

### 4.5 Color-Aware Extension

The OKLab extension is validation-only. At λ=0.029574882, relative to incumbent Hard Top10, PSNR increases by 0.239585 dB, Top25 local PSNR by 0.226884 dB, global CIEDE2000 decreases by 3.70%, and Top10 5×5 CIEDE2000 decreases by 3.42%. BER remains close. Gini increases from 0.094569 to 0.097303. At λ=0.059149764, color and absolute fidelity improve further, but Gini increases to 0.100017. Both candidates improve color error on all 50 validation images, but neither meets the complete preservation gate. OKLab can complement pixel-tail supervision by reducing absolute chromatic distortion, with a small normalized-concentration trade-off.

### 4.6 External Reference

TrustMark Q/P provide higher-fidelity external reference operating points: PSNR 42.293755/48.107281 and full LPIPS 0.000964749/0.000304701. Their payload, ECC, quantization, decoder preprocessing, and robustness semantics differ from MBRS. TrustMark is therefore a **REFERENCE comparison** describing different fidelity–robustness regimes, not a strict superiority benchmark.

## 5. Conclusion

Hard Patch16/stride8/Top10 raw-MSE supervision improves global fidelity, absolute local pixel tails, and normalized residual concentration in the tested crop-trained MBRS setting while approximately preserving observed crop BER. Full-image perceptual metrics improve, but local LPIPS remains mixed. A validation-only OKLab extension further reduces absolute color distortion and improves PSNR, while partly weakening the concentration advantage. These results support treating absolute distortion, spatial concentration, perceptual quality, and color quality as distinct objectives.

The evidence is limited to one seed, one backbone, and 50 formal test pairs, with prior project exposure to the test partition. It does not establish human perceptual superiority, cross-seed statistical superiority, or strict superiority over TrustMark. Citation placeholders remain for prior watermarking methods, image-quality metrics, tail-risk terminology, datasets, and color models.
