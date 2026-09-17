# Claims and Limits

Safe claims:
- Ours improves absolute local-tail fidelity under the controlled internal protocol.
- Ours improves global PSNR, local PSNR, LPIPS, and CIEDE2000.
- Crop robustness is approximately preserved.
- Internal matched-PSNR analysis shows that improvements are not explained only by higher global PSNR.
- External methods are qualitative/contextual references at matched display PSNR.
- Hard Local-Tail and OKLab are training-only losses; inference architecture is unchanged.

Do not claim:
- state of the art
- universal superiority over HiDDeN or MaskWM
- significantly stronger crop robustness
- human perceptual superiority
- residuals become uniformly distributed
- Gini is the primary objective
- external BER superiority
- validation results as formal project-test results

Important distinctions:
1. Natural frozen source-resolution project-test:
   Global PSNR 36.263172
   Ours PSNR 36.854909

2. Internal source-resolution matched-PSNR project-test:
   Global 36.910772
   Ours 36.906757

3. External qualitative display-space matched-PSNR:
   mean validation display PSNR = 40.72 dB on 512x512 canvas

Never mix these three evaluation spaces.
