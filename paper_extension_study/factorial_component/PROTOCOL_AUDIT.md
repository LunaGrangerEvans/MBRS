# Factorial Component Protocol Audit

## Scope

This factorial uses only the four seed17 continuation branches with RGB/global weight `0.5`:

```text
                 OKLab OFF              OKLab ON
Tail OFF         RGB0.5-Control         Global + OKLab only
Tail ON          Hard Local-Tail        Ours
```

The frozen Global paper row with RGB/global weight `1.0` is intentionally excluded from the factorial design. It remains the primary paper baseline outside the factorial.

## Row provenance

- `RGB0.5-Control`: `paper_extension_study/component_ablation/rgb05_control/`, registered config and epoch-20 checkpoint.
- `Global+OKLab-only`: `paper_extension_study/component_ablation/global_oklab_only/`, registered config and epoch-20 checkpoint.
- `Hard Local-Tail` and `Ours`: formal natural project-test rows from `paper_bundle/evidence/final_ours_project_test_report.md` and `paper/paper_evidence_audit.md`; their paired per-image records are from the bundle-referenced frozen per-image output.
- The two new control runs use the same source/checkpoint identity recorded in their provenance files and the same source hash as the bundled frozen protocol: `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`.

## Parity checks

| Protocol field | RGB0.5-Control | Global+OKLab-only | Hard Local-Tail | Ours | Parity result |
|---|---|---|---|---|---|
| Source | seed17 epoch-100 crop-trained Global | same | same frozen source policy | same frozen source policy | PASS |
| Source SHA-256 | `1a82ec4f...cce2907` | same | same bundled source | same bundled source | PASS |
| Continuation | 20 epochs | 20 epochs | 20 epochs | 20 epochs | PASS |
| Learning rate | `1e-4` | `1e-4` | `1e-4` | `1e-4` | PASS |
| Batch / workers | 16 / 0 | 16 / 0 | 16 / 0 | 16 / 0 | PASS |
| Augmentation | `RandomCrop(0.3, 1.0)` | same | same | same | PASS |
| Seed | 17 | 17 | 17 | 17 | PASS |
| Model restoration | same MBRS model state | same | same | same | PASS |
| BatchNorm restoration | restored | restored | restored | restored | PASS |
| Adam restoration | restored | restored | restored | restored | PASS |
| Project-test manifest | fixed 50-image source-resolution manifest | same | same | same | PASS |
| Messages / crop masks | fixed messages and fixed masks | same | same | same | PASS |
| Metric implementation | fixed clipped-RGB evaluator, native 32×32 tails, image-level BER | same evaluator family | bundled formal evaluator | bundled formal evaluator | PASS |
| Metric space | natural source-resolution 128×128 project-test | same | same | same | PASS |

The objective is the intended factorial difference:

- `M00` RGB0.5-Control: `10 L_msg + 0.5 L_RGB`;
- `M01` Global+OKLab-only: `10 L_msg + 0.5 L_RGB + 0.059149764 L_OKLab`;
- `M10` Hard Local-Tail: `10 L_msg + 0.5 L_RGB + 0.5 L_tail`;
- `M11` Ours: `10 L_msg + 0.5 L_RGB + 0.5 L_tail + 0.059149764 L_OKLab`.

No scientifically relevant parity failure was found. The only intentional difference among cells is the presence/absence of the Tail and OKLab terms. The bundled Hard Top-25% lineage branch is not part of this factorial and is not used to define the final method.

## Metric and bootstrap audit

- PSNR uses the frozen paper estimator `10 log10(1 / mean_i(global_mse_i))` for point estimates and every bootstrap replicate.
- Top-25 Local PSNR is the documented image-level local-tail metric.
- P95/P99, LPIPS, CIEDE2000, and Gini use the documented image-level aggregation.
- BER30 uses per-image BER contributions; individual bits are not IID bootstrap units.
- Bootstrap uses 20,000 paired image-index resamples with analysis seed `20260916`.
- The factorial summary and bootstrap files are under this directory; no project-test result was used for configuration selection.

## Verdict

**PROTOCOL AUDIT: PASS.** The four rows are suitable for descriptive fixed-RGB-weight factorial analysis. The factorial does not replace the frozen paper baseline or authorize a new final method.
