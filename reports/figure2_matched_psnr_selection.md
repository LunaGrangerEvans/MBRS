# Figure 2 all-method matched-PSNR selection report

Figure 2 was regenerated from the fixed 50-image validation manifest. All four frozen-checkpoint outputs use one global α per method; no per-image tuning, retraining, or project-test data was used.

- Common validation target: `40.72 dB` on the common 512×512 display canvas.
- HiDDeN-64 α: `0.295820698`; frozen endpoint `30.140571 dB`; matched validation `40.720000 dB`.
- MaskWM-D_64 α: `0.819258720`; frozen endpoint `38.988422 dB`; matched validation `40.720000 dB`.
- MBRS crop-trained global α: `0.937583178`; frozen endpoint `40.160196 dB`; matched validation `40.720000 dB`.
- Ours = Hard Local-Tail + global OKLab α: `0.993847698`; frozen endpoint `40.666397 dB`; matched validation `40.720000 dB`.

| Sample | ID | Validation index | Shared ROI `(x,y,w,h)` | Global PSNR | Ours PSNR | mismatch | Δ local PSNR (Ours−Global) | Δ P95 MSE (Ours−Global) | Global ROI CIEDE | Ours ROI CIEDE | Δ ROI CIEDE | shared residual vmax ×10 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0846` | 45 | `(48,28,32,32)` | 42.564091 | 42.576550 | 0.012459 | +0.056475 | -2.141e-06 | 2.590156 | 2.487896 | +0.102259 | 0.353390 |
| 2 | `0821` | 20 | `(84,32,32,32)` | 39.990453 | 39.993395 | 0.002941 | +0.039611 | -2.510e-06 | 2.594594 | 2.522333 | +0.072260 | 0.409826 |

## Sources and outputs

- `Original`: `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt`
- `HiDDeN-64`: `/mnt/wmcontent/GLX/icassp/MBRS/results/fig2_stress/external_baselines/hidden_64bit_validation.pt`
- `MaskWM-D_64`: `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits/native_512`
- `MBRS crop-trained global`: `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/controlled_seed17_global_continuation/checkpoint_0020.pth`
- `Ours = Hard Local-Tail + global OKLab`: `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/checkpoint_0020.pth`
- `HiDDeN_checkpoint`: `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt`
- `MaskWM_checkpoint`: `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/checkpoints/maskwm/D_64bits.pth`
- `MaskWM_provenance`: `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits/provenance.json`
- `matched_validation_outputs`: `/mnt/wmcontent/GLX/icassp/MBRS/reports/external_matched_psnr/matched_validation_outputs_512.pt`

## Rendering contract

- Columns: Original; HiDDeN-64; MaskWM-D64; MBRS Global; Ours.
- Rows: Sample 1, Zoom, Residual ×10, Sample 2, Zoom, Residual ×10.
- Red ROI boxes appear only on full-image rows. Zoom and residual tiles have no borders.
- Residual maps use one shared per-sample range across all five columns and are mean absolute displayed-RGB residuals amplified ×10.
- Caption: “Matched-PSNR qualitative comparison; mean display PSNR = 40.72 dB. All four methods are shown at validation-calibrated operating points. HiDDeN-64 and MaskWM-D_64 use frozen external checkpoints with only the global residual-strength scale adjusted. Residual maps use a shared scale within each sample and are amplified ×10 for visualization.”

## Generated outputs

- `/root/workspace/GLX/icassp/MBRS/paper/figures/figure2_matched_psnr.png` (SHA-256 `6e080bbfe5297dc9014e6a383c41b864d6ba481b0feee7492bd80d394d81bf34`)
- `/root/workspace/GLX/icassp/MBRS/paper/figures/figure2_matched_psnr.pdf` (SHA-256 `7ec26c88a5fb63d64c8e23d7ef823b25ca81d43ac50be09e81eadbd10153e971`)
- `/root/workspace/GLX/icassp/MBRS/paper/figures/figure2_matched_psnr.pptx` (SHA-256 `337f1c5b8615e337c43c613b8fdc2f6c48877c338c39a2837750be026fe48f16`)
- Selection/provenance manifest: `/root/workspace/GLX/icassp/MBRS/reports/figure2_matched_psnr_manifest.json`.
- Full validation ranking: `/root/workspace/GLX/icassp/MBRS/reports/figure2_matched_psnr_ranking.csv`.
