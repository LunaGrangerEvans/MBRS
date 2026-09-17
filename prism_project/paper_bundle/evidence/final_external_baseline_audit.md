# Final external-baseline audit

Scope: validation assets used by the final qualitative Figure 2. External methods are reference baselines, not strictly matched training comparisons.

## HiDDeN-64

- Checkpoint: `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt`; SHA-256 `a2181ef05278de48d5af28129474d508a3178b849b9b8655f8a3cb46c671a7e8`.
- Display/evaluation cache: `/mnt/wmcontent/GLX/icassp/MBRS/results/fig2_stress/external_baselines/hidden_64bit_validation.pt`; cache manifest SHA-256 `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`.
- Host/reference: current fixed 50-image validation manifest, 128×128 normalized tensors; cache shape is 50×3×128×128 and the cache was checked against the manifest provenance.
- Display path: saved normalized output → clipped RGB[0,1] → common 512×512 bilinear canvas. Figure PSNR and residual use that exact displayed pair.
- Recomputed displayed-canvas PSNR: 30.141 dB; the value is derived from the current cache and current validation host, not from an unrelated formal-test row.
- Status: **PASS** for checkpoint identity, host alignment, and display-source identity; this is a retrained external reference, not an official pretrained checkpoint.

## MaskWM-D_64

- Released checkpoint: `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/checkpoints/maskwm/D_64bits.pth`; SHA-256 `0eb1b2bfe37e0479a74483a2d07cd4a28fd113c4d83c4e4396c12fad9e5393b2`.
- Provenance: `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits/provenance.json`; manifest SHA-256 `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`.
- Evaluated 128×128 outputs: `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits` (50 files); native display outputs: `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits/native_512` (50 files).
- Preprocessing: fixed manifest 128×128 host → official-style 512 canvas → MaskWM 256 model → native 512 output; the native 512 PNG is the final displayed image. The 128 PNG is the common-resolution quality artifact.
- Host/reference: validation manifest sample index is preserved one-to-one; no sample or message substitution was made.
- PSNR: the displayed-canvas mean is 38.988 dB; the existing native-display annotations re-compute with maximum absolute error 2.100e-06 dB against the native displayed output and the corresponding 512 host canvas.
- Residual: final renderer computes mean-channel absolute RGB residual from the exact displayed output/reference pair, uses one ×10 visualization scale and shared per-sample color range; no per-method normalization is applied.
- Status: **PASS**. The higher fidelity is retained as a measured property of the released checkpoint/protocol, not treated as an error or external-superiority claim.

## Boundary

Neither external method is used to select the final training hyperparameters. External PSNR/visuals are contextual references only; their released/retrained preprocessing and training protocols differ from MBRS.
