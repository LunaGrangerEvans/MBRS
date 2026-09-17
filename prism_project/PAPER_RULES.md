# Paper Workspace Rules

These rules are permanent for all remaining paper-writing work.

## Authoritative evidence

- `paper_bundle/` is the only authoritative experimental evidence source.
- `paper/paper_evidence_audit.md` is the authoritative paper-level evidence summary.
- Do not use `experiments/`, `results/`, `reports/`, `visualizations/`, historical paper files, old figures, old tables, filenames, timestamps, or git history as evidence unless `paper_bundle/` explicitly references them for provenance.
- If any historical file conflicts with `paper_bundle/`, `paper_bundle/` wins.

## Frozen method

- Ours = Hard Local-Tail + global OKLab regularization.
- `lambda_OK = 0.059149764`.
- `w_msg = 10`.
- RGB/global weight = `0.5`.
- Hard local-tail weight = `0.5`.
- Training selector = P16 / S8 / hard Top-10%.
- 225 overlapping candidates, 23 selected patches.
- Hard Local-Tail is the intermediate ablation.
- Local-tail and OKLab are training-only losses.
- No additional inference module is introduced.

## Metric-space separation

Never mix these three spaces:

### A. Natural frozen 128×128 project-test

- formal main-result space

### B. Internal matched-PSNR 128×128

- controlled fairness analysis only

### C. Figure 2 matched display-PSNR at 512×512

- qualitative external-comparison space only
- mean display PSNR = `40.72 dB`

Never report the `40.72 dB` display value as a formal 128×128 result.

## Claims

Allowed:

- improves absolute local-tail fidelity
- improves measured perceptual/color fidelity
- approximately preserves crop robustness
- representative/contextual external references
- matched-PSNR qualitative comparison

Not allowed:

- state-of-the-art
- universal superiority over HiDDeN or MaskWM
- significantly stronger crop robustness
- uniformly distributed residuals
- human perceptual superiority
- external BER ranking
- Gini as a primary objective

## Historical exclusions

Do not use in the main paper:

- `w_msg=80` stress experiments
- Ours-base naming
- obsolete six-column Figure 2
- old validation-only OKLab interpretation
- outdated historical tables
- Hard Top-25% lineage as the final method

The existing `paper/figures/` symlink and `paper/tables/` contents are preserved as workspace state. Do not import or treat their contents as evidence unless the material is explicitly copied from `paper_bundle/` under these rules.
