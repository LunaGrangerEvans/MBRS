# Figure 2 regeneration report

The requested six-column Figure 2 was regenerated from persisted outputs on the fixed 50-image validation manifest. No project-test sample, formal Table I result, or checkpoint was modified.

## Recommendation

Use `figure2_candidate_main` in the main paper. It directly supports the primary local-tail claim with the promoted matched crop-robustness strong-embedding pair; `figure2_candidate_external` is the alternate when external-method contrast is the editorial priority. Both are deliberately metric-selected best-case qualitative evidence and must not be described as representative sampling.

## MaskWM consistency

- Current MaskWM visual asset: **PASS**; see [`figure2_maskwm_consistency_check.md`](figure2_maskwm_consistency_check.md).
- Current provenance and display outputs use the fixed manifest hash `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`.
- Recomputed PSNR annotation error is `2.100e-06 dB`; the existing figure shows PSNR only, not SSIM or LPIPS.
- The legacy `maskwm_results.csv` hash is out of scope/current-manifest mismatched (`36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339`); it was not consumed.

## Fixed protocol

- Manifest: `/mnt/wmcontent/GLX/icassp/MBRS/reports/content_selector/validation_manifest.pt`
- Manifest SHA-256: `60cc8b4d7a6cb9f5c0c77b3f66e02d6e0b739b05867123a6ec246ba6e4992de2`
- Split: validation; 50 images; 64-bit messages; source size 128×128.
- Selection and rendering use saved tensors/PNG outputs only. No project-test samples, attack realizations, manual edits, per-method ROI, or per-method residual normalization were used.

## Checkpoints, configs, and exact outputs

| Method | Checkpoint | Config | Exact visual output | Status |
|---|---|---|---|---|
| HiDDeN-64 (external) | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_64bit_retrained/hidden_64bit_epoch_200.pyt` | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/hidden_64bit_retrained/training_config.json` | `/mnt/wmcontent/GLX/icassp/MBRS/results/fig2_stress/external_baselines/hidden_64bit_validation.pt` | external reference; current-manifest cache |
| MaskWM-D_64 (external) | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/checkpoints/maskwm/D_64bits.pth` | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits/provenance.json` | `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits` and `/mnt/wmcontent/GLX/icassp/MBRS/external_baselines/outputs/maskwm_validation/D_64bits/native_512` | external reference under released checkpoint |
| MBRS crop-trained global | `/mnt/wmcontent/GLX/icassp/MBRS/checkpoints/fig2_stress/promoted_ours-base_wmsg80/checkpoint_0020.pth` | `/mnt/wmcontent/GLX/icassp/MBRS/checkpoints/fig2_stress/promoted_ours-base_wmsg80/config.json` | `/mnt/wmcontent/GLX/icassp/MBRS/results/fig2_stress/promoted_ours-base_wmsg80/outputs.pt` | validation-only strong-embedding visualization |
| Hard Local-Tail (Ours) | `/mnt/wmcontent/GLX/icassp/MBRS/checkpoints/fig2_stress/promoted_ours_wmsg80/checkpoint_0020.pth` | `/mnt/wmcontent/GLX/icassp/MBRS/checkpoints/fig2_stress/promoted_ours_wmsg80/config.json` | `/mnt/wmcontent/GLX/icassp/MBRS/results/fig2_stress/promoted_ours_wmsg80/outputs.pt` | validation-only strong-embedding visualization |
| Hard Local-Tail + OKLab | `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/checkpoint_0020.pth` | `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/config.resolved.json` | `/mnt/wmcontent/GLX/icassp/MBRS/reports/crop_global_oklab/validation_seed17_crop_hard16_stride8_top10_global_oklab_g25/seed17_crop_hard16_stride8_top10_global_oklab_g25_outputs.pt` | validation-only extension |

### Strong-embedding operating point

The two internal comparison columns use the promoted `w_msg=80`, `λ_img=1` pair from the same crop-trained epoch-100 source. This is a validation-only visualization setting, not a replacement for formal Table I.

| Method | BER30 | PSNR (frozen 128 source metric) | Local PSNR | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| MBRS crop-trained global | 0.109937 | 31.486392 | 30.780588 | `e2016823b56b830b8ee108a9fd08a5d3af8e7430dff3d569941b58895b2b11ec` |
| Hard Local-Tail (Ours) | 0.110625 | 32.353024 | 31.610333 | `df4357556a1bb4f2caf84b6603da5905e80347c498b0d42c22e5556059db31a9` |

Hard Local-Tail + OKLab uses the validation-only `g25` run (`λ_OKLab=0.059149764`, checkpoint `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/seed17_crop_hard16_stride8_top10_global_oklab_g25/checkpoint_0020.pth`). It showed the clearest measured color benefit among the available OKLab variants, but failed the complete preservation gate because Gini increased beyond tolerance; it is included only as a validation-only extension.

## Display metrics used in the figure headers

PSNR labels in the PNG/PDF/PPTX are computed on the displayed 512×512 canvas used by every column. This makes the annotation correspond to the rendered panel; the strong-pair frozen 128-source metrics are recorded above.

| Column | Display PSNR | Display SSIM | BER30 annotation |
|---|---:|---:|---:|
| Original | reference | reference | — |
| HiDDeN-64 (external) | 30.141 dB | 0.91798 | — |
| MaskWM-D_64 (external) | 38.988 dB | 0.96969 | — |
| MBRS crop-trained global | 35.283 dB | 0.93086 | 0.110 |
| Hard Local-Tail (Ours) | 36.823 dB | 0.93833 | 0.111 |
| Hard Local-Tail + OKLab | 40.666 dB | 0.97483 | validation-only |

### Candidate_main

| Sample | Validation index | ROI `(x,y,w,h)` | Δ local PSNR (dB) | Δ P95 MSE | ROI residual-energy reduction | Structured-TV reduction | OKLab CIEDE2000 reduction | Selection rationale |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `0822` | 21 | `(68,84,32,32)` | +1.448 | +2.832e-04 | +5.139e-03 | +7.294e-03 | +2.256 | largest internal local-tail gain with positive P95 reduction and reduced structured residual proxy |
| `0823` | 22 | `(88,36,32,32)` | +1.372 | +2.420e-04 | +4.984e-03 | +8.036e-03 | +2.030 | largest internal local-tail gain with positive P95 reduction and reduced structured residual proxy |

The complete main and external rankings are persisted in `reports/figure2_candidate_main_ranking.csv` and `reports/figure2_candidate_external_ranking.csv`. Main ranking priority was local PSNR improvement, then P95 local patch-MSE reduction, ROI residual-energy reduction, and a structured-residual TV proxy.

### Candidate_external

| Sample | Validation index | ROI `(x,y,w,h)` | Δ local PSNR (dB) | Δ P95 MSE | ROI residual-energy reduction | Structured-TV reduction | OKLab CIEDE2000 reduction | Selection rationale |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `0803` | 2 | `(48,28,32,32)` | +0.807 | +2.889e-04 | +4.997e-03 | +8.204e-03 | +1.168 | strong external-method disagreement plus lower Hard Local-Tail local error; retained as a reference contrast |
| `0821` | 20 | `(8,28,32,32)` | +0.253 | +6.611e-05 | +2.497e-03 | +3.669e-03 | +2.535 | strong external-method disagreement plus lower Hard Local-Tail local error; retained as a reference contrast |

Candidate_external prioritizes external local-error disagreement and the gap between the external references and Hard Local-Tail, while retaining the internal ROI gain as a secondary term. It uses the same fixed manifest, shared host, shared ROI per sample, and same normalization policy.

## Generated files

- [`visualizations/figure2/figure2_candidate_main.png`](/root/workspace/GLX/icassp/MBRS/visualizations/figure2/figure2_candidate_main.png) — SHA-256 `8894a13e689792b8b96a441f407005ce75c9950873a128f9a31e058a1d17606a`
- [`visualizations/figure2/figure2_candidate_main.pdf`](/root/workspace/GLX/icassp/MBRS/visualizations/figure2/figure2_candidate_main.pdf) — SHA-256 `b16732e0e77f1ab6fd4fcbde6e64fe11e9381cc1b7d0045dc5df13d6098c113c`
- [`visualizations/figure2/figure2_candidate_main_pptx.pptx`](/root/workspace/GLX/icassp/MBRS/visualizations/figure2/figure2_candidate_main_pptx.pptx) — SHA-256 `8253f336b5db44cf5bdc1499bccae8d929ba8ade796d67445a9a15f79aaf7689`
- [`visualizations/figure2/figure2_candidate_external.png`](/root/workspace/GLX/icassp/MBRS/visualizations/figure2/figure2_candidate_external.png) — SHA-256 `31510a3acac8ffb31fd3e55ad03a95d5864f02107a9118e51b1252b5ecb3c109`
- [`visualizations/figure2/figure2_candidate_external.pdf`](/root/workspace/GLX/icassp/MBRS/visualizations/figure2/figure2_candidate_external.pdf) — SHA-256 `bd39dec237e703010f82761851982b15dd07e0549d3fc9cb4baef247aa667add`
- [`visualizations/figure2/figure2_candidate_external_pptx.pptx`](/root/workspace/GLX/icassp/MBRS/visualizations/figure2/figure2_candidate_external_pptx.pptx) — SHA-256 `0705c6f3820b9b27f051da8737f50ddc8a53179cb952d072acd06dc376f383b5`
- `reports/figure2_candidate_main_ranking.csv` and `reports/figure2_candidate_external_ranking.csv` — complete selection rankings

## Figure annotations and honesty boundary

- Exact method names in all generated files are: Original; HiDDeN-64 (external); MaskWM-D_64 (external); MBRS crop-trained global; Hard Local-Tail (Ours); Hard Local-Tail + OKLab.
- Full rows show the same host canvas and a shared red ROI rectangle. Zoom rows use the same 8× source-pixel scale; smaller ROIs are centered on a common canvas.
- Residual rows show mean absolute RGB residual after normalization, multiplied by ×10 for visualization only. Each sample has one shared magma range across all six methods.
- External methods are reference baselines under released checkpoints, not strictly matched training baselines. OKLab is validation-only. Selection is disclosed as best-case and non-representative.
