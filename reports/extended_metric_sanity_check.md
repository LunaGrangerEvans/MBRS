# Extended metric sanity check

执行日期：2026-09-10。输入范围统一为 [0,1]；没有训练模型。

| Metric | identity | Gaussian perturbation | Expected direction | Status |
|---|---:|---:|---|---|
| SSIM | 1.00000000 | 0.68048477 | perturbation lowers | PASS |
| 3-scale MS-SSIM, weights [0.3,0.3,0.4] | 1.00000000 | 0.69430465 | perturbation lowers | PASS |
| LPIPS | existing implementation retained | existing implementation retained | perturbation raises | PASS in existing audit |
| DISTS | N/A | N/A | lower is better | BLOCKED: no mature installed implementation |
| GMSD | N/A | N/A | lower is better | BLOCKED: no mature installed implementation |

MS-SSIM uses a mathematically valid 3-scale configuration for 128×128 images. Patch MS-SSIM is omitted because the 32×32 local support is insufficient for a defensible multi-scale calculation. DISTS/GMSD were not implemented from copied code and no dependency was added.
