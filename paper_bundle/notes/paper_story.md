# Main Paper Story

Problem:
Global RGB reconstruction controls average distortion but does not explicitly control the upper tail of local distortion.

Controlled baseline:
MBRS crop-trained global.

Intermediate ablation:
Hard Local-Tail.

Final method:
Ours = Hard Local-Tail + global OKLab.

Final objective:
L = 10 L_msg + 0.5 L_RGB + 0.5 L_tail + 0.059149764 L_OKLab

Hard Local-Tail:
- patch size 16
- stride 8
- 225 overlapping patches
- hard Top-10%
- 23 selected patches

Training:
- 128x128 RGB
- 64-bit payload
- RandomCrop(0.3, 1.0)
- seed17
- same epoch-100 crop-trained source checkpoint
- restore Adam and BatchNorm state
- 20 continuation epochs
- lr = 1e-4
- no extra inference module

Primary formal evidence:
Compared with MBRS crop-trained global at the natural frozen operating point, Ours:
- +0.592 dB global PSNR
- +0.620 dB Top-25 local PSNR
- about 13.1% lower P95 patch MSE
- about 18.4% lower LPIPS
- lower CIEDE2000
- identical BER30 = 0.113125

Matched-PSNR internal analysis:
- source-resolution project-test PSNR:
  Global 36.910772 dB
  Ours 36.906757 dB
- PSNR mismatch = 0.004014 dB
- BER30 remains identical
- Ours retains modest local-tail and clearer perceptual/color improvements

External matched-PSNR qualitative comparison:
- four methods are validation-calibrated to mean display PSNR = 40.72 dB
- common 512x512 display canvas
- methods:
  HiDDeN-64
  MaskWM-D_64
  MBRS crop-trained global
  Ours
- this is a qualitative/display-space comparison only
- external BER is not strictly comparable

Main conceptual story:
Global RGB controls average distortion
→ Hard Local-Tail controls absolute local upper-tail distortion
→ OKLab refines color/perceptual fidelity
→ crop robustness is approximately preserved
