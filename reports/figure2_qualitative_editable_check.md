# Figure 2 Editable Qualitative Comparison Check

- Selected validation indices: `20`, `30` mapped to `Sample 1`, `Sample 2`.
- Columns: `Original`, `HiDDeN-64`, `MaskWM-D_64`, `Ours-base`, `Ours`, `Ours + OKLab`.
- `Ours-base` is the saved MBRS Global continuation; `Ours` is the saved Hard Top10 output.
- Each sample has a full-image row and a shared-ROI 4x zoom row; the red ROI rectangles are separate editable vector objects in PPTX.
- Original has no PSNR label; all watermarked columns retain the existing per-image PSNR values below the full-image row.
- PPTX objects: 24 separate image objects for the six-column version, editable text boxes, and editable red ROI shapes.
- No training or checkpoint inference was run; the visual columns are lossless crops from the existing frozen qualitative montages, and only saved MBRS outputs were read for sample selection.

## Outputs

- `/root/workspace/GLX/icassp/MBRS/paper/figures/figure2_qualitative_comparison.png`
- `/root/workspace/GLX/icassp/MBRS/paper/figures/figure2_qualitative_comparison.pdf`
- `/root/workspace/GLX/icassp/MBRS/paper/figures/figure2_qualitative_comparison.pptx`
- `/root/workspace/GLX/icassp/MBRS/paper/figures/figure2_v1_5col.pptx`
- `/root/workspace/GLX/icassp/MBRS/paper/figures/figure2_v2_6col.pptx`
