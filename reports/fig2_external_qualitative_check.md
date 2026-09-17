    # Fig. 2 external qualitative comparison check

生成日期：2026-09-13。

## Final outputs

- PDF: `/root/workspace/GLX/icassp/MBRS/paper/figures/fig2_external_qualitative.pdf`
- PNG: `/root/workspace/GLX/icassp/MBRS/paper/figures/fig2_external_qualitative.png`
- Renderer: `/root/workspace/GLX/icassp/MBRS/paper/render_fig2_external_qualitative.py`
- Frozen visual input: `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/external_qualitative/qualitative_5x6_main.png`
- Frozen visual input SHA-256: `4a39a0c4e8559a403b822bb37db11014a82f3deb3d56ef97b7cf5a068aa9ff3e`
- Frozen PSNR artifact: `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/external_qualitative/per_image_psnr.csv`

正文图严格使用 fixed qualitative manifest 的前三个预注册样本：**0, 10, 20**。没有替换、重排或新增样本。

## Column and checkpoint identity

| Column | Identity | Checkpoint / source identity | SHA-256 |
|---|---|---|---|
| Original | Fixed validation host | `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt` | `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2` |
| HiDDeN-64 | 64-bit retrained reimplementation, seed 17, epoch 200 | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt` | `a2181ef05278de48d5af28129474d508a3178b849b9b8655f8a3cb46c671a7e8` |
| MaskWM-D_64 | Official global `D_64bits`, `mu=1.3`, `blue=True`, native 512 display path | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/checkpoints/maskwm/D_64bits.pth` | `0eb1b2bfe37e0479a74483a2d07cd4a28fd113c4d83c4e4396c12fad9e5393b2` |
| MBRS Global | `controlled_seed17_global_continuation`, epoch 20 | `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/controlled_seed17_global_continuation/checkpoint_0020.pth` | `31fddd342f8bd8bd003e4fbc0cf10878035b1cda81d3e10a647833aad84e5248` |
| Hard Top10 | `controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50`, epoch 20 | `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50/checkpoint_0020.pth` | `74fd12a6147b20e68477d9439cf6c416181bc48f07cfa3c3384989f48850f92c` |
| Hard Top10 + OKLab | `seed17_crop_hard16_stride8_top10_global_oklab_g12p5`, epoch 20 | `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g12p5/checkpoint_0020.pth` | `e2847537921565edef747fda7a9eb82e06976ed77624988faa8e7a28b003098c` |

## Panel source paths

正文 renderer 对冻结主图中的 panel 做无损矩形裁取；没有加载模型、重新推理或处理 RGB。下表同时记录每个 panel 的实际视觉 source crop 和其 upstream saved output。crop box 为 PIL 的 `(left, top, right, bottom)`，右/下边界不包含在 crop 中。

| Sample | Column | Frozen visual source and crop box | Upstream saved output |
|---:|---|---|---|
| 0 | Original | `qualitative_5x6_main.png`, `(86, 285, 602, 800)` | `validation_manifest.pt`, `images[0]` |
| 0 | HiDDeN-64 | `qualitative_5x6_main.png`, `(741, 285, 1257, 800)` | content-selector-specific output retained in the frozen visual artifact; checkpoint identity above |
| 0 | MaskWM-D_64 | `qualitative_5x6_main.png`, `(1395, 285, 1911, 800)` | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits/native_512/image_000.png` |
| 0 | MBRS Global | `qualitative_5x6_main.png`, `(2050, 285, 2566, 800)` | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5/controlled_seed17_global_continuation_outputs.pt`, `encoded[0]` |
| 0 | Hard Top10 | `qualitative_5x6_main.png`, `(2704, 285, 3220, 800)` | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5/controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt`, `encoded[0]` |
| 0 | Hard Top10 + OKLab | `qualitative_5x6_main.png`, `(3359, 285, 3875, 800)` | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5/seed17_crop_hard16_stride8_top10_global_oklab_g12p5_outputs.pt`, `encoded[0]` |
| 10 | Original | `qualitative_5x6_main.png`, `(86, 868, 602, 1383)` | `validation_manifest.pt`, `images[10]` |
| 10 | HiDDeN-64 | `qualitative_5x6_main.png`, `(741, 868, 1257, 1383)` | content-selector-specific output retained in the frozen visual artifact; checkpoint identity above |
| 10 | MaskWM-D_64 | `qualitative_5x6_main.png`, `(1395, 868, 1911, 1383)` | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits/native_512/image_010.png` |
| 10 | MBRS Global | `qualitative_5x6_main.png`, `(2050, 868, 2566, 1383)` | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5/controlled_seed17_global_continuation_outputs.pt`, `encoded[10]` |
| 10 | Hard Top10 | `qualitative_5x6_main.png`, `(2704, 868, 3220, 1383)` | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5/controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt`, `encoded[10]` |
| 10 | Hard Top10 + OKLab | `qualitative_5x6_main.png`, `(3359, 868, 3875, 1383)` | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5/seed17_crop_hard16_stride8_top10_global_oklab_g12p5_outputs.pt`, `encoded[10]` |
| 20 | Original | `qualitative_5x6_main.png`, `(86, 1451, 602, 1966)` | `validation_manifest.pt`, `images[20]` |
| 20 | HiDDeN-64 | `qualitative_5x6_main.png`, `(741, 1451, 1257, 1966)` | content-selector-specific output retained in the frozen visual artifact; checkpoint identity above |
| 20 | MaskWM-D_64 | `qualitative_5x6_main.png`, `(1395, 1451, 1911, 1966)` | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits/native_512/image_020.png` |
| 20 | MBRS Global | `qualitative_5x6_main.png`, `(2050, 1451, 2566, 1966)` | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5/controlled_seed17_global_continuation_outputs.pt`, `encoded[20]` |
| 20 | Hard Top10 | `qualitative_5x6_main.png`, `(2704, 1451, 3220, 1966)` | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5/controlled_seed17_hard_patch16_stride8_top10_global_weight50_local_weight50_outputs.pt`, `encoded[20]` |
| 20 | Hard Top10 + OKLab | `qualitative_5x6_main.png`, `(3359, 1451, 3875, 1966)` | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g12p5/seed17_crop_hard16_stride8_top10_global_oklab_g12p5_outputs.pt`, `encoded[20]` |

`validation_manifest.pt` above refers to `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt`. All frozen visual source crops have size 516 x 515 pixels and are placed without changing their aspect ratio.

## Per-sample PSNR

Values below are read verbatim from the existing evaluator artifact `per_image_psnr.csv`; the figure renderer does not recompute PSNR from panel pixels.

| Sample | HiDDeN-64 | MaskWM-D_64 | MBRS Global | Hard Top10 | Hard Top10 + OKLab |
|---:|---:|---:|---:|---:|---:|
| 0 | 32.96 dB | 40.43 dB | 42.01 dB | 42.09 dB | 42.43 dB |
| 10 | 31.64 dB | 39.77 dB | 42.04 dB | 42.07 dB | 42.32 dB |
| 20 | 27.81 dB | 38.62 dB | 39.42 dB | 39.58 dB | 39.79 dB |

Original panels intentionally have no PSNR label.

## Fixed shared ROIs

The same ROI is shown in all six columns of each row. Coordinates use `(x, y, width, height)`.

| Sample | Native 128 x 128 coordinates | Common 512 x 512 display coordinates |
|---:|---|---|
| 0 | `(96, 96, 32, 32)` | `(384, 384, 128, 128)` |
| 10 | `(32, 32, 32, 32)` | `(128, 128, 128, 128)` |
| 20 | `(96, 64, 32, 32)` | `(384, 256, 128, 128)` |

Figure note wording is limited to: “Same host per row; fixed shared ROIs; no residual amplification.” The ROI is used only for visualization and does not alter or trigger metric computation.

## Integrity confirmations

- **Same host per row:** confirmed by the frozen external qualitative protocol and retained montage; all six panels in a row correspond to the same fixed manifest index.
- **Fixed messages:** confirmed; the retained montage was produced with the current protocol's fixed 64-bit message per row. The final renderer neither loads nor generates messages.
- **No residual amplification:** confirmed. Panels are direct lossless crops of the actual-RGB frozen montage; no residual view is computed.
- **No RGB modification:** confirmed. No sharpening, contrast enhancement, color transform, or other pixel adjustment is applied. The existing ROI outline remains part of the frozen panel raster.
- **No cherry-picking:** confirmed. The original manifest order `[0, 10, 20, 30, 40]` is unchanged, and the main-body figure takes its first three entries `[0, 10, 20]` exactly.
- **No training / inference:** confirmed. The final renderer only reads the frozen montage and evaluator CSV.
- **Layout:** confirmed as 3 rows x 6 columns with the requested English column labels, no large title, compact gutters, and a one-line footer.
- **Output format:** PNG is written at 350 dpi; PDF keeps all newly typeset titles, sample labels, PSNR values, and footer as vector text while embedding the unchanged raster panels.
