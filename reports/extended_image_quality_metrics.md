# Extended full-reference image quality

Frozen controlled checkpoints and fixed formal manifest only. No training was run.

## Sanity

- ssim_identity: 1.00000000
- ms_ssim_identity_3scale: 1.00000000
- ssim_noise: 0.68141377
- ms_ssim_noise_3scale: 0.68756682
- DISTS: N/A; no mature implementation installed and no implementation added.
- GMSD: N/A; no mature implementation installed and no implementation added.

## Summary

| Method | PSNR | SSIM | MS-SSIM | DISTS | GMSD | LPIPS | Top25 local PSNR | Bottom10 SSIM | Top10 LPIPS | Gini | CV | Top10/Mean | Top10 energy share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Global continuation | 36.263172 | 0.950759 | 0.983036 | N/A | N/A | 0.00234960 | 35.522213 | 0.902273 | 0.00136127 | 0.095007 | 0.173718 | 1.295702 | 0.161963 |
| Hard Patch16 stride8 Top25 | 36.445329 | 0.952631 | 0.983713 | N/A | N/A | 0.00220503 | 35.747490 | 0.905780 | 0.00129904 | 0.088604 | 0.162244 | 1.277077 | 0.159635 |
| Hard Patch16 stride8 Top10 | 36.442300 | 0.952541 | 0.983690 | N/A | N/A | 0.00221518 | 35.754376 | 0.905504 | 0.00136614 | 0.086897 | 0.159015 | 1.270921 | 0.158865 |
| Gradient-aware Top10 alpha2 | 36.332801 | 0.952017 | 0.983473 | N/A | N/A | 0.00219921 | 35.624195 | 0.905216 | 0.00121320 | 0.089410 | 0.163465 | 1.278817 | 0.159852 |

## Direction

SSIM/MS-SSIM higher is better; DISTS/GMSD/LPIPS lower is better; Bottom10 SSIM higher is better; SSIM TailGap and perceptual tail metrics lower are better.

## Scope

Native32 patch LPIPS and SSIM are reported. MS-SSIM is full-image only because patch support is not sufficient. DISTS/GMSD remain N/A due environment availability.
