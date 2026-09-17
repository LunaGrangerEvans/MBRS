# Figure 2 generation notes

## Provenance

- Raw plotted values: `project_test_curve.csv` (project-test rows only).
- Paired cross-check: `operating_curve_summary.csv`.
- Frozen grid: `FROZEN_ALPHA_GRID.txt`.
- Interpolation and interpretation audit: `OPERATING_CURVE_AUDIT.md`.
- Frozen-run metadata: `curve_metadata.json`.
- No file under `paper_bundle/` and no Prism manuscript file was modified.

## Data and transformations

- Methods plotted: `MBRS crop-trained Global` and `Ours = Hard Local-Tail + global OKLab`.
- Metrics plotted exactly: `psnr` on the x-axis; `top25_local_psnr`, `lpips`, and `ber30` on the three y-axes.
- Alpha grid used exactly: `0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.00`.
- Raw measured point count: 9 Global points + 9 Ours points. All raw points remain visible.
- Global measured PSNR range: `36.263172–39.351276 dB`.
- Ours measured PSNR range: `36.854909–39.942909 dB`.
- Common PSNR overlap: `36.854909–39.351276 dB`.
- Matched-PSNR markers shown: yes, hollow method-shaped markers at these five audited points:
  - 36.854909 dB
  - 37.479001 dB
  - 38.103093 dB
  - 38.727185 dB
  - 39.351276 dB
- Interpolation: `numpy.interp` linear interpolation of each measured curve within the common overlap only. No extrapolation, spline, polynomial, regression, or smoothing was used.
- Endpoint precision note: the audit's lower endpoint is printed as `36.854909`, while the raw Ours endpoint is `36.854909401406786`; this 6-decimal display-rounding difference is snapped to the measured endpoint before interpolation. No extrapolation is performed.

## Visual encoding

- Figure layout: three horizontal panels, shared two-method legend, white background, compact conference typography.
- Global: blue circle markers with a solid line; Ours: vermillion square markers with a dashed line. Marker and line differences preserve grayscale readability.
- Alpha=1 natural operating points: larger hollow diamonds over the corresponding raw points for both methods.
- Common-overlap shading: used in all three panels as a subtle light-blue vertical band from `36.854909` to `39.351276` dB; it does not alter data or axes.
- Identical x-axis limits in all panels: `36.10–40.05 dB`.
- Y-axis limits: Top-25 Local PSNR `35.30–39.45 dB`; LPIPS `0.00085–0.00245`; BER@30 `0.11295–0.11370`.
- BER@30 uses normal numeric ordering and four-decimal ticks; no visual inversion, clipping, normalization, or compression was applied.

## Outputs

- PDF: `figure2_operating_curve.pdf` (vector lines/text suitable for LaTeX inclusion).
- SVG: `figure2_operating_curve.svg` (editable vector backup).
- PNG: `figure2_operating_curve.png` at 300 DPI.
- Figure dimensions: `7.16 in × 3.30 in`; PNG raster dimensions `2148 × 990 px`.
- Generation did not retrain, retune, select new operating points, or change any alpha value.
