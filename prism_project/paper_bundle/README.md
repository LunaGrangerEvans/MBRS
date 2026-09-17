# Prism-ready paper bundle

This directory is a curated bundle of authoritative materials for the current ICASSP watermarking paper. It contains copied reports, the selected final figures, concise LaTeX table fragments, a minimal paper skeleton, and drafting notes. It does not contain training code, checkpoints, datasets, raw output tensors, obsolete exploratory figures, or historical `w_msg=80` stress figures.

Read these files before drafting with Prism or another writing assistant:

1. `README.md`
2. `notes/paper_story.md`
3. `notes/claims_and_limits.md`

## Evidence boundaries

Formal natural frozen source-resolution project-test evidence:

- `evidence/final_method_frozen_config.md`
- `evidence/controlled_continuation_protocol.md`
- `evidence/final_ours_project_test_report.md`
- `tables/table1_main_ablation.tex`

Internal source-resolution matched-PSNR project-test analysis:

- `evidence/matched_psnr_alpha_residual.md`
- `tables/table2_matched_psnr.tex`

External qualitative/display-space evidence:

- `evidence/external_matched_psnr_feasibility.md`
- `evidence/final_external_baseline_audit.md`
- `evidence/figure2_matched_psnr_selection.md`
- `evidence/figure2_matched_psnr_manifest.json`
- `figures/figure2_matched_psnr.pdf`
- `figures/figure2_matched_psnr.png`

These three evaluation spaces must remain separate:

1. Natural frozen source-resolution project-test: Global PSNR `36.263172`, Ours PSNR `36.854909`.
2. Internal source-resolution matched-PSNR project-test: Global `36.910772`, Ours `36.906757`.
3. External qualitative display-space matched-PSNR: mean validation display PSNR `40.72 dB` on a common `512x512` canvas.

Do not treat the external display-space comparison as a formal quantitative benchmark, and do not rank external BER. Historical exploratory `w_msg=80` runs are excluded from the formal paper bundle and must not be used as formal results.

## Figure selection

- `figures/figure1_framework.{pdf,png}` is copied from the newer `paper/figures/figure1_final_method.{pdf,png}`. Its check confirms `Ours = Hard Local-Tail + global OKLab`, frozen `lambda_OK = 0.059149764`, no validation-only wording, no Gini key-effect claim, and unchanged inference architecture.
- `figures/figure2_matched_psnr.{pdf,png}` is copied from `paper/figures_external_matched_psnr/`, the latest all-method matched-PSNR output. No obsolete Figure 2 variants are included.

## Paper template status

No project-local ICASSP `main.tex`, class/style, bibliography style, or `references.bib` was present. Therefore `paper/main.tex` is a minimal IEEEtran-compatible skeleton with placeholders only, and `paper/references.bib` contains a TODO comment. No references have been invented.

Missing requested authoritative reports: none; all eight listed reports/manifests were present when this bundle was assembled.
