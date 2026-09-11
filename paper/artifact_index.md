# Draft v1 artifact index

## Manuscript and tables

- [draft_v1.md](draft_v1.md)
- [main_table.md](main_table.md)
- [main_table.csv](main_table.csv)
- [ablation_table.md](ablation_table.md)
- [external_reference_table.md](external_reference_table.md)
- [external_reference_table.csv](external_reference_table.csv)
- [figure_plan.md](figure_plan.md) — individual PNG/PDF paths and final captions
- [figure_table_plan_4page.md](figure_table_plan_4page.md) — frozen two-figure/two-table main-body set
- [four_page_main_layout.md](four_page_main_layout.md) — captions, layout, and provenance boundary for the compact body
- [table1_main_results.tex](table1_main_results.tex)
- [table2_color_extension.tex](table2_color_extension.tex)
- [evidence_manifest.json](evidence_manifest.json)
- [paired_diagnostics.json](paired_diagnostics.json)
- [paper_evidence_freeze_checklist.md](../reports/paper_evidence_freeze_checklist.md)
- [artifact_index.md](artifact_index.md)

## Reproduction code

- [freeze_paper_evidence.py](../experiments/freeze_paper_evidence.py)
- [render_paper_freeze.py](../experiments/render_paper_freeze.py)
- [verify_paper_freeze.py](../experiments/verify_paper_freeze.py)

## Generated numerical artifacts (mounted storage)

- [per_image.csv](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/per_image.csv)
- [00_outputs.pt — Global](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/00_outputs.pt)
- [01_outputs.pt — non-overlap Hard](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/01_outputs.pt)
- [02_outputs.pt — overlapping Top25](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/02_outputs.pt)
- [03_outputs.pt — overlapping Top10](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/03_outputs.pt)
- [04_outputs.pt — Soft T0.25](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/04_outputs.pt)
- [07_outputs.pt — Multi-scale](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/07_outputs.pt)
- [08_outputs.pt — Excess](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/08_outputs.pt)
- [09_outputs.pt — Gradient-aware](/mnt/wmcontent/GLX/icassp/MBRS/reports/paper_freeze_v1/09_outputs.pt)

These files contain outputs of existing checkpoints, not newly trained weights.

## Refreshed selected figures (mounted storage)

- Fig1 method: [PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig01_method.png), [PDF](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig01_method.pdf)
- Fig3 quality/robustness: [PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig03_quality_robustness.png), [PDF](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig03_quality_robustness.pdf)
- Fig4 concentration: [PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig04_concentration.png), [PDF](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig04_concentration.pdf)
- Fig5 pixel/perceptual tails: [PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig05_pixel_perceptual.png), [PDF](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/fig05_pixel_perceptual.pdf)
- FigS1 external reference: [PNG](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/figS1_external_reference.png), [PDF](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/figS1_external_reference.pdf)
- [render provenance](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/paper_freeze_v1/provenance.json)

## Compact two-figure main-body assets

- Figure 1 method framework: [PNG](figures/fig01_method_framework.png), [PDF](figures/fig01_method_framework.pdf)
- Figure 2 qualitative comparison: [PNG](figures/fig02_qualitative_comparison.png), [PDF](figures/fig02_qualitative_comparison.pdf)
- [Local render provenance](figures/provenance.json)

Fig2 reuses the existing predetermined RGB panels; it was not regenerated. Its source paths are listed in [figure_plan.md](figure_plan.md).
