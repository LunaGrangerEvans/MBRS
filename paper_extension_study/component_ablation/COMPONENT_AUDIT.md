# Component Ablation Audit

This audit uses the authoritative frozen rows from `paper_bundle/` and the two registered seed17 control runs under `paper_extension_study/component_ablation/`. It does not modify the frozen final method, paper bundle, or Prism manuscript.

## Rows and provenance

- **MBRS crop-trained Global**, **Hard Local-Tail**, and **Ours**: formal natural project-test rows from `paper_bundle/evidence/final_ours_project_test_report.md` and `paper/paper_evidence_audit.md`; shared controlled source/protocol identity is defined by the bundled frozen-method and continuation reports.
- **Global + OKLab only**: `global_oklab_only/`, registered objective `10 L_msg + 0.5 L_RGB + 0.059149764 L_OKLab`, no local-tail term.
- **Global-RGB0.5-Control**: `rgb05_control/`, registered objective `10 L_msg + 0.5 L_RGB`, no local-tail or OKLab term.
- The formal frozen rows are used as the paper-authoritative comparison. The two new rows are extension evidence and cannot replace Ours.
- Shared source checkpoint: `/mnt/wmcontent/GLX/icassp/MBRS/experiments/runs/optimization_global_seed17_128_m64_crop/checkpoint_0100.pth`, SHA-256 `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`; new branch configs, checkpoints, logs, per-image rows, and split summaries are stored inside the corresponding extension directories.

## Protocol parity

| Field | Frozen Global / Hard / Ours | Global + OKLab only | Global-RGB0.5-Control | Difference |
|---|---|---|---|---|
| Source checkpoint | Same seed17 epoch-100 crop-trained Global source under bundled protocol | Same path/hash; restored | Same path/hash; restored | None |
| Continuation | 20 epochs | 20 epochs | 20 epochs | None |
| Learning rate | 1e-4 | 1e-4 | 1e-4 | None |
| Batch / workers | 16 / 0 | 16 / 0 | 16 / 0 | None |
| Crop augmentation | RandomCrop(0.3, 1.0) | Same | Same | None |
| Seed | 17 | 17 | 17 | None |
| Model / BatchNorm / Adam restoration | Restored under bundled protocol | Restored | Restored | None |
| Validation/project-test identity | Fixed messages and fixed crop masks | Same fixed evaluator/manifests | Same fixed evaluator/manifests | None |
| Objective | Global, Hard, or frozen Ours term set | Tail off; OKLab on | Tail off; OKLab off; RGB weight 0.5 | Objective terms only |
The bundled continuation report also contains a Hard Top-25% lineage branch. That branch is not substituted for the formal frozen Hard/Ours rows here; the final method remains P16/S8 hard Top-10% plus global OKLab.

## Project-test component summary

The `component_summary.csv` file contains the requested metrics and deltas relative to MBRS crop-trained Global. Lower is better for P95, P99, LPIPS, CIEDE2000, Gini, and BER30; higher is better for PSNR and Top-25 Local PSNR.

| Method | PSNR | Top-25 Local PSNR | P95 MSE | P99 MSE | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MBRS crop-trained Global | 36.263172 | 35.522213 | 3.016531971e-04 | 3.199585360e-04 | 0.00234960 | 3.535857 | 5.159104 | 0.095007 | 0.11312500 |
| Global-RGB0.5-Control | 35.164486 | 34.447721 | 3.813118118e-04 | 4.026092408e-04 | 0.00261828 | 4.027174 | 5.858652 | 0.088316 | 0.11306250 |
| Global + OKLab only | 35.943842 | 35.207559 | 3.227863646e-04 | 3.422455385e-04 | 0.00236554 | 3.577205 | 5.215556 | 0.093363 | 0.11325000 |
| Hard Local-Tail | 36.442300 | 35.754376 | 2.844551472e-04 | 3.010726967e-04 | 0.00221518 | 3.484790 | 5.075666 | 0.086897 | 0.11293750 |
| Ours = Hard Local-Tail + global OKLab | 36.854909 | 36.141895 | 2.619925590e-04 | 2.782322409e-04 | 0.00191711 | 3.254091 | 4.772229 | 0.092130 | 0.11312500 |

## Answers to the registered questions

### A. Does RGB0.5-Control reproduce most of the Hard/Ours gain?

No. Relative to the frozen Global row, RGB0.5-Control is lower in PSNR and local PSNR and higher in P95, P99, LPIPS, and CIEDE2000. BER30 is close (`0.1130625` versus `0.1131250`) but does not rescue the quality result. A simple global RGB-weight reduction does not reproduce the Hard/Ours pattern.

### B. Does Global+OKLab-only improve color/perceptual fidelity without reproducing the full local-tail behavior?

Not relative to the frozen Global baseline in this run. Global+OKLab-only is lower in PSNR/local PSNR and has higher P95, P99, LPIPS, and CIEDE2000 than Global. It is better than RGB0.5-Control on these metrics, but that comparison is not evidence of improvement over the frozen Global baseline. The standalone OKLab control therefore does not show an independent color/perceptual gain under this continuation.

### C. Does Hard Local-Tail provide a distinct local-tail/concentration contribution?

Yes, descriptively. Hard Local-Tail improves Top-25 Local PSNR by `+0.232163 dB`, lowers P95 by `1.719804989e-05`, lowers P99 by `1.888583935e-05`, and lowers Gini by `0.008109840` relative to Global. It also improves LPIPS/CIEDE2000, but the clearest component-specific evidence is the local-tail and concentration movement.

### D. Does Ours show complementary effects from Tail + OKLab?

The measured results support cautious language about complementary effects, not synergy. Relative to Hard Local-Tail, Ours improves PSNR, local PSNR, P95, P99, LPIPS, and both CIEDE2000 metrics, while BER30 remains effectively tied. Ours has higher Gini than Hard, so the complement is not uniformly favorable on every diagnostic.

### E. Are any component results negative or mixed?

Yes. Both Tail-off controls are negative relative to the frozen Global quality row. Ours versus Hard has a Gini increase, and BER30 differences are small/mixed. These outcomes remain visible and are not converted into a universal or causal superiority claim.

### F. Is the current final method interpretation still supported?

Yes, with narrower wording. The formal frozen Ours row still supports the interpretation that Hard Local-Tail supplies explicit absolute local-tail control and the combined Ours objective improves measured perceptual/color fidelity under the frozen protocol. The new component controls do not justify claiming that OKLab alone improves the frozen Global baseline, nor do they establish statistical interaction or “synergy.”

## Boundary

All rows in this audit are natural source-resolution project-test results; they must not be mixed with the internal matched-PSNR or external 512×512 display-space results. Gini remains diagnostic only. The extension rows are supplementary and do not retune or replace the frozen paper evidence.
