# Reviewer-risk analysis

| Likely reviewer concern | Current evidence | Missing evidence | Best response | Required experiment |
|---|---|---|---|---|
| Improvement is only 0.233 dB | Fixed controlled result and P95/Top25 reductions | More task-specific effect-size context | Emphasize tail risk and negligible BER cost, not PSNR superiority | None immediately; improve analysis |
| Worst-patch metric is method-shaped | Primary metric is defined independently at 32×32 | Perceptual alignment is weak | Report multiple scales, SSIM, LPIPS, and P95/P99 | Analysis-only scale/perceptual report |
| Why only seed17? | Teacher-defined main protocol | Broader robustness evidence | State seed17 is a controlled engineering study, not a significance claim | No seed29/41 under current instruction |
| Crop training does not increase concentration | Tail ratio actually decreases | Why local tail matters despite this | Reframe as implicit global objective, not concentration claim | Motivation figure and distribution table |
| Gain may be global PSNR, not local | Local-specific gain only +0.050 dB | Percentile improvement curve | Report ΔPSNR, ΔWorst, and percentile curves together | Analysis-only percentile figure |
| Overlap only adds computation | 225 candidates, 38.9% union coverage | Runtime/memory accounting | Quantify coverage and cost; do not claim free gain | One profiling run or report theoretical cost |
| Why Top25? | Historical screen and controlled reference | Selectivity/coverage sensitivity under overlap | State Top25 is a fixed engineering choice, not theoretically optimal | One preapproved ratio refinement if needed |
| MSE may not equal visual artifact | MSE/LPIPS rank alignment near zero | Content/activity confound | Treat MSE tail as a proxy and disclose limitation | Content-normalization diagnostic |
| BER is not fully unaffected | BER30 changes within 0.00025 in controlled results | More attacks/datasets | Report full BER curve and exact crop protocol | Existing fixed attacks are sufficient for first paper |
| No other attacks | Current task is crop robustness | Generalization to blur/JPEG/noise | Narrow claim to crop robustness | Optional supplementary attacks |
| Qualitative cherry-picking | One aligned image index exists | Predeclared example rule | Use fixed manifest and selection rule based on baseline tail | Generate multi-example panel |
| Excess failure was implementation-specific | Initial gradient matched but final local loss collapsed | Active-region trajectory | Explain dynamic collapse and reject formulation | Existing logs plus analysis; no sweep |
| Multi-scale failure is under-tuned | Fixed 0.7/0.3 was predeclared | Whether another mix works | Report it as a negative minimal test, not exhaustive claim | No grid |
| Why not perceptual training? | LPIPS at 16×16 is invalid and rank alignment is weak | Larger-region perceptual objective | Make metric alignment a limitation/future direction | Analysis-only 32×32 study first |
| Contribution may be too narrow for one MBRS setting | Controlled result is clean but narrow | Other payload/resolution/data evidence | Call it a controlled local-tail study and avoid broad robustness claims | New setting only if effect-size justification is needed |
