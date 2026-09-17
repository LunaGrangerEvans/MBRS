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

## Superseded compact two-figure assets

- Figure 1 method framework: [PNG](figures/fig01_method_framework.png), [PDF](figures/fig01_method_framework.pdf)
- Figure 2 qualitative comparison: [PNG](figures/fig02_qualitative_comparison.png), [PDF](figures/fig02_qualitative_comparison.pdf)
- [Local render provenance](figures/provenance.json)

These are historical pre-freeze assets. The current camera-ready assets are listed below and are generated from the frozen final-method outputs.

## Frozen final-method assets

- [Frozen configuration](../reports/final_method_frozen_config.md)
- [One-pass formal project-test report](../reports/final_ours_project_test_report.md)
- [Expanded formal per-image CSV](../reports/final_ours_project_test_per_image.csv)
- [External-baseline audit](../reports/final_external_baseline_audit.md)
- Figure 1 final method: [PNG](figures/figure1_final_method.png), [PDF](figures/figure1_final_method.pdf)
- Figure 2 final comparison: [PNG](figures/figure2_final.png), [PDF](figures/figure2_final.pdf), [editable PPTX](figures/figure2_final.pptx)
- Final table fragments: [ablation](tables/final_method_ablation.tex), [color](tables/final_method_color.tex)
