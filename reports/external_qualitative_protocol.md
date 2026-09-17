# External qualitative visualization protocol

日期：2026-09-13。该 protocol 记录 MaskWM 作为论文第二个 visual external
baseline 的固定展示方案。它不产生 strict cross-method leaderboard。

## Fixed inputs

- Manifest: `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt`
- Manifest SHA-256: `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`
- Samples: validation indices **0, 10, 20, 30, 40**, pre-registered by the existing
  MBRS qualitative protocol.
- Every column uses the same host image and the same fixed 64-bit message for the row.
- MaskWM column is official global `D_64bits`, not ED, because the main figure needs a
  global embedding baseline. ED_64bits is reported separately in the audit/CSV.

## Main native/default figure

Column order:

```text
Original | HiDDeN-64 | MaskWM-D_64 | MBRS Global | Hard Top10 | Hard Top10 + OKLab
```

Rules:

- MaskWM keeps official `μ=1.3`, `blue=True`, 64-bit checkpoint, and official
  512→256→512 inference path.
- The figure uses a common 512×512 display canvas: MaskWM is shown at its native
  output resolution; the 128×128 outputs of the other methods and the host are
  bilinearly upsampled once for display. This is not a claim that the other models
  were trained at 512×512.
- Main figure uses actual RGB outputs and **no residual amplification**.
- Zoom figure uses the same fixed native 32×32 ROI for all six columns, enlarged with
  nearest-neighbor display only.
- ROI is selected once from the existing incumbent Hard Top10 CIEDE2000-tail rule;
  MaskWM is not used to choose its own favorable crop.
- Each full-image panel is labeled with its per-sample PSNR. Original has no score
  label in the corrected renderer.

Generated files:

```text
/mnt/wmcontent/GLX/icassp/MBRS/visualizations/external_qualitative/qualitative_5x6_main.png
/mnt/wmcontent/GLX/icassp/MBRS/visualizations/external_qualitative/qualitative_5x6_zoom4x.png
```

The source renderer is
[experiments/render_external_qualitative.py](../experiments/render_external_qualitative.py).

## Matched-PSNR diagnostic

Matched-PSNR is feasible and is kept separate from the native/default figure. The
target for each fixed sample is the actual Hard Top10 PSNR on the common 512 display
canvas. MaskWM-D_64 is re-inferred over a predeclared JND-factor grid:

```text
μ ∈ {0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
     0.75, 1.0, 1.10, 1.20, 1.30}
```

The closest predefined μ is selected per fixed sample; no decoder, checkpoint, or
training state changes. This is the only embedding-strength sweep. Official MaskWM
`μ=1.3` remains unchanged in the native/default figure and fixed-manifest results.

| Validation index | Hard Top10 target PSNR | Selected MaskWM μ | MaskWM PSNR |
|---:|---:|---:|---:|
| 0 | 42.085095 | 1.10 | 41.940353 |
| 10 | 42.074077 | 1.00 | 42.108093 |
| 20 | 39.577250 | 1.20 | 39.354493 |
| 30 | 40.737163 | 1.00 | 41.004697 |
| 40 | 42.209593 | 1.00 | 42.493283 |

Generated diagnostic:

```text
/mnt/wmcontent/GLX/icassp/MBRS/visualizations/external_qualitative/matched_psnr_maskwm_vs_ours.png
/mnt/wmcontent/GLX/icassp/MBRS/visualizations/external_qualitative/matched_psnr.csv
```

The matched figure is appropriate for a qualified local-artifact comparison near a
common global PSNR. It must not replace the native/default figure or be described as
MaskWM's default operating point.

## Recommendation

Use `MaskWM-D_64` as the second qualitative external baseline beside HiDDeN-64 and
the MBRS methods. It is preferable to TrustMark for the main qualitative figure
because:

- it has an official 64-bit checkpoint with no ECC packet conversion;
- it is not an extreme high-fidelity outlier like TrustMark P/Q in the current setup;
- its official model explicitly handles local masks and localization, which is
  relevant to the MBRS partial-retention discussion.

The caption must still state:

> MaskWM-D_64 is an official external reference evaluated with its native 256/512
> path and displayed on a common 512×512 canvas; this qualitative comparison
> is not a strict same-protocol BER ranking.
