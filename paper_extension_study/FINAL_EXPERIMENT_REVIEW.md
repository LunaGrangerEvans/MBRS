# Final Experimental Review

This review covers the final factorial, overhead, and continuation-seed stages. It does not modify the frozen paper evidence, `paper_bundle/`, or the Prism manuscript.

## 1. Factorial component evidence

The fixed-RGB-weight factorial uses only four seed17 natural 128×128 project-test cells:

```text
M00 RGB0.5-Control       M01 Global+OKLab-only
M10 Hard Local-Tail      M11 Ours
```

The frozen Global RGB-weight-1.0 row is correctly excluded from the factorial design. `factorial_component/PROTOCOL_AUDIT.md` reports parity across source, state restoration, training schedule, fixed masks/messages, evaluator, and metric space.

## 2. Simple effects

- OKLab without Tail (`M01−M00`) improves PSNR, local PSNR, P95/P99, LPIPS, and CIEDE2000, but increases Gini; BER30 is inconclusive.
- OKLab with Tail (`M11−M10`) improves the same fidelity/color metrics, increases Gini, and has inconclusive BER30.
- Tail without OKLab (`M10−M00`) provides the strongest local-tail effect: PSNR `+1.277815 dB`, local PSNR `+1.306655 dB`, P95 `−9.685666e-05`, and P99 `−1.015365e-04`; Gini and BER30 are inconclusive under bootstrap.
- Tail with OKLab (`M11−M01`) remains favorable on fidelity/color metrics and has inconclusive Gini/BER30.

The pure OKLab comparison is `Global+OKLab-only vs RGB0.5-Control`; `Global+OKLab-only vs frozen Global` is not pure because frozen Global uses RGB weight 1.0.

## 3. Interaction contrasts

The interaction contrast is `M11−M10−M01+M00`.

| Metric | Interaction | 95% interval | Interpretation |
|---|---:|---:|---|
| PSNR | `−0.366748 dB` | `[-0.380423, −0.351979]` | diminishing returns |
| Top-25 Local PSNR | `−0.372320 dB` | `[-0.390011, −0.353527]` | diminishing returns |
| P95 MSE | `+3.606286e-05` | `[+3.413000e-05, +3.828961e-05]` | diminishing favorable reduction |
| P99 MSE | `+3.752325e-05` | `[+3.526294e-05, +3.996619e-05]` | diminishing favorable reduction |
| LPIPS | `−4.532402e-05` | `[-1.119358e-04, +1.866620e-05]` | interval includes zero |
| Global CIEDE2000 | `+0.219270` | `[+0.206887, +0.231136]` | diminishing favorable reduction |
| Top10 CIEDE2000 | `+0.339659` | `[+0.322037, +0.355784]` | diminishing favorable reduction |
| Gini | `+0.000186` | `[-0.000648, +0.001019]` | interval includes zero |
| BER30 | approximately `0` | `[-0.000813, +0.000813]` | interval includes zero |

The results support “complementary effects” with diminishing returns in several metrics. They do not establish synergy.

## 4. Factorial bootstrap uncertainty

The factorial bootstrap uses 20,000 paired image-level resamples with analysis seed `20260916`. Every PSNR replicate recomputes aggregate mean MSE before conversion to PSNR. BER30 uses per-image contributions, not individual bits.

Intervals exclude zero for the main PSNR/local/P95/P99/LPIPS/CIEDE simple effects. Tail Gini effects and all BER30 effects include zero. Full results are under `factorial_component/bootstrap/`.

## 5. Practical overhead

The benchmark used 20 warm-ups and 100 synchronized FP32 iterations on an A800 with batch `16×3×128×128` and 64-bit payloads.

- Training-step mean: Global `105.7795 ms`, Ours `107.2207 ms`; measured Ours overhead `+1.36%`.
- Training peak allocated memory: Global `6975.47 MiB`, Ours `6976.96 MiB`; reserved memory was `7990 MiB` for both.
- Inference parameter count: `20,805,391` total and trainable for both.
- Inference peak allocated/reserved memory: `903.29/7990 MiB` for both.
- Inference mean latency: Global `22.7025 ms`, Ours `21.7693 ms`; the `−4.11%` difference is noisy timing variation, not an architectural speedup claim.

The architectural conclusion is: **no additional inference module or parameter is introduced.**

## 6. Continuation-seed sensitivity

The exact same seed17 epoch-100 crop-trained Global source was used for continuation seeds `17, 23, 42`; seed17 reused its valid frozen endpoint. This is stochastic continuation-seed sensitivity, not full end-to-end multi-seed reproducibility.

| Method | PSNR mean±SD | Local PSNR mean±SD | P95 mean±SD | LPIPS mean±SD | Global CIEDE mean±SD | BER30 mean±SD |
|---|---:|---:|---:|---:|---:|---:|
| Global | 36.369326 ± 0.095698 | 35.639303 ± 0.101442 | 2.940083e-04 ± 6.629717e-06 | 0.00229671 ± 0.00008780 | 3.542748 ± 0.007896 | 0.114167 ± 0.001031 |
| Hard Local-Tail | 36.561878 ± 0.136319 | 35.880872 ± 0.126885 | 2.771600e-04 ± 7.536177e-06 | 0.00210872 ± 0.00014343 | 3.477169 ± 0.027968 | 0.114042 ± 0.000961 |
| Ours | 37.042103 ± 0.201670 | 36.330960 ± 0.192221 | 2.523028e-04 ± 1.022421e-05 | 0.00192414 ± 0.00009657 | 3.205831 ± 0.050039 | 0.114125 ± 0.000886 |

Ours−Global differences by seed are:

| Seed | ΔPSNR | ΔLocal PSNR | ΔP95 | ΔP99 | ΔLPIPS | ΔGlobal CIEDE | ΔBER30 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | `+0.591737` | `+0.619682` | `−3.966064e-05` | `−4.172632e-05` | `−0.000432485` | `−0.281766` | approximately `0` |
| 23 | `+0.806668` | `+0.825547` | `−4.822228e-05` | `−4.978691e-05` | `−0.000321149` | `−0.397179` | `+0.000250` |
| 42 | `+0.619927` | `+0.629741` | `−3.723369e-05` | `−3.896048e-05` | `−0.000364088` | `−0.331806` | `−0.000375` |

The primary fidelity/color direction persists across all three continuation seeds. Gini is mixed for Ours−Global and BER30 is mixed; neither supports a directional claim. Full individual rows and Hard/Ours deltas are in `continuation_seed/CONTINUATION_SEED_REPORT.md`.

## 7. Negative or mixed findings

- Global+OKLab-only is not better than frozen Global in this controlled continuation.
- RGB0.5-Control does not reproduce the Hard/Ours gain.
- Factorial interaction is diminishing rather than synergistic for several metrics.
- BER30 is mixed across operating points, factorial effects, and continuation seeds.
- Gini is diagnostic and mixed across the final Ours-vs-Global continuation comparison; it increases relative to Hard in all three seeds.
- Continuation sensitivity is not full end-to-end reproducibility because the source model is held fixed.
- The original non-paper-estimator PSNR bootstrap remains a distinct estimand; the corrected aggregate-MSE bootstrap must be used for paper PSNR.

## 8. Remaining limitations

- No end-to-end source-training seed sweep was run.
- External Figure 2 remains qualitative/display-space evidence and external BER is not comparable.
- The new factorial does not prove formal causal interaction beyond its declared contrasts.
- The overhead benchmark is hardware- and implementation-specific; measured inference latency differences are noisy.
- Verified bibliography entries are still missing.

## 9. Evidence suitable for the main paper

- Keep the frozen natural project-test Table I and method definition unchanged.
- If space permits, add one sentence that paired image-level bootstrap intervals exclude zero for the main fidelity/tail metrics while no directional BER improvement is established.
- Add a short statement that the fidelity trend persists across stochastic continuation seeds from a common frozen source checkpoint, with no claim of full reproducibility.
- A short training-overhead sentence is suitable if practical cost is discussed: approximately `+1.36%` mean training-step time in this benchmark, with no inference module or parameter increase.

## 10. Evidence suitable only for supplementary discussion

- Full factorial table, simple effects, interaction contrasts, and bootstrap distributions.
- Full overhead protocol/results.
- Per-seed continuation tables, checkpoints, and RNG audits.
- Complete operating curves and interpolation diagnostics.
- Upper-tail-risk interpretation and Figure 1 redesign notes.
- Reference-gap audit and blocked/negative findings.

## 11. Evidence that should NOT be used

- Old or corrected-away Top10 CIEDE pixel-tail values.
- Mean-per-image PSNR bootstrap values as if they were paper-estimator PSNR.
- BER30 as equivalent or superior across methods.
- Gini as the primary objective or proof of uniform residuals.
- “Synergy,” SOTA, universal external superiority, or full end-to-end reproducibility.
- Historical `w_msg=80`, Ours-base, obsolete figures/tables, or project-test-tuned configurations.

## 12. Recommended final claim wording

> Under the frozen natural project-test protocol, Ours improves global and local fidelity and measured perceptual/color metrics while preserving BER30. Paired image-level bootstrap intervals exclude zero for the fidelity/tail metrics, whereas no directional BER improvement is established.

> The fidelity trend persists across stochastic continuation seeds from a common frozen source checkpoint. This is continuation-seed sensitivity evidence, not full end-to-end reproducibility.

> Hard Local-Tail and global OKLab have complementary measured effects with diminishing returns in several factorial interaction contrasts; no synergy claim is made.

## 13. Whether additional experiments are justified

No additional experiments are justified after this stage. The remaining weaknesses are limitations to disclose, not reasons to reopen hyperparameter, Top-k, patch-size, or external-baseline searches. The only non-experimental next work is reference verification and paper integration after approval.

## Final decision

**STOP EXPERIMENTS: YES**

The current frozen final method remains supported. All extension evidence stays supplementary until explicitly approved for integration.
