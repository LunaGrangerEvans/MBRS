# Figure 2 MaskWM consistency check

Audit scope: current Figure 2 qualitative assets and the fixed content-selector validation manifest only. No project-test sample or formal Table I input was opened.

## Verdict

**PASS for the current visual asset.** The current MaskWM provenance manifest hash matches the fixed validation manifest, all 50 evaluated 128×128 outputs and 50 native 512×512 display outputs are present, and the five displayed PSNR values re-compute from the corresponding native display images.

## Evidence

- Fixed manifest: `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt`
- Fixed manifest SHA-256: `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`
- MaskWM checkpoint: `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/checkpoints/maskwm/D_64bits.pth`
- MaskWM checkpoint SHA-256: `0eb1b2bfe37e0479a74483a2d07cd4a28fd113c4d83c4e4396c12fad9e5393b2`
- MaskWM provenance manifest SHA-256: `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`
- Evaluated 128×128 output files: `50`; native 512×512 display files: `50`.
- Existing qualitative PSNR annotation maximum absolute re-computation error: `2.100e-06 dB`.
- Existing montage MaskWM panel maximum rasterized-panel MSE against its native source: `154.259` uint8²; the nonzero value is expected from montage rasterization and ROI overlay.
- The existing qualitative figure prints PSNR only; SSIM and LPIPS are not shown in its annotations.

## Logged scope issue

`reports/maskwm_results.csv` records legacy reference rows with manifest hash(es) `36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339`. They do not match the current fixed validation manifest (`60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`) and were not used for the regenerated figures. This is a stale/out-of-scope report row, not a mismatch in the current displayed MaskWM asset.

The regenerated Figure 2 uses the current `maskwm_validation/D_64bits` provenance and outputs, so the stale legacy row cannot silently supply a metric or image.
