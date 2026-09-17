# Next Experimental Phase Decision Memo

This memo is based only on the frozen paper authority and the completed `paper_extension_study/` outputs. It does not launch experiments, modify `paper_bundle/`, change the frozen method, or edit the Prism manuscript.

## 1. PSNR estimator audit result

The original bootstrap used the mean of per-image PSNR differences. The frozen paper uses PSNR of the aggregate mean clipped-RGB MSE:

```text
PSNR_paper = 10 log10(1 / mean_i(global_mse_i))
```

The estimators differ because PSNR is nonlinear in MSE. Clipping, normalized-tensor conversion, image dimensions, pixel weighting, and Global/Ours pairing are otherwise aligned. The original PSNR interval is valid for a different estimand but is not compatible with the frozen paper estimator.

Corrected result:

- observed `Ours − Global`: `+0.591736509 dB`;
- 95% paired image-level bootstrap interval: `[+0.552852365, +0.635779193] dB`;
- bootstrap standard error: `0.021363448 dB`;
- interval includes zero: `No`.

Corrected files are separate under `bootstrap/`; the original non-PSNR bootstrap files were not overwritten.

## 2. Bootstrap evidence

Using 20,000 paired image-index resamples with analysis seed `20260916`, the Global-versus-Ours interval excludes zero for PSNR, Top-25 Local PSNR, P95, P99, LPIPS, Global CIEDE2000, Top10 CIEDE2000, and Gini. The BER30 interval includes zero. The full interpretation is in `bootstrap/bootstrap_interpretation.md`.

The supported wording is:

> Fidelity improvements are stable under paired image-level resampling, whereas no directional BER improvement is established.

This wording does not call the results statistically significant and does not interpret BER interval inclusion as equivalence.

## 3. Component-control conclusions

- **RGB0.5-Control:** does not reproduce most of the Hard/Ours gain. It is worse than frozen Global on PSNR, local PSNR, P95, P99, LPIPS, and CIEDE2000; BER30 is close but not explanatory.
- **Global+OKLab-only:** does not improve the frozen Global baseline in this continuation. It is better than RGB0.5-Control, but that comparison does not establish an independent OKLab gain over Global.
- **Hard Local-Tail:** provides a distinct local-tail/concentration contribution: relative to Global, local PSNR is `+0.232163 dB`, P95 and P99 decrease, and Gini decreases.
- **Ours:** relative to Hard, improves PSNR, local PSNR, P95, P99, LPIPS, and CIEDE2000 while BER30 is effectively tied; Gini increases relative to Hard.

The evidence supports “complementary effects” as cautious wording. It does not establish synergy or an interaction beyond additive component effects.

## 4. Operating-curve conclusions

The same pre-registered alpha grid `[0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.00]` was used for Global and Ours, with validation first and project-test second. No alpha exceeds 1 and no per-image tuning occurred.

The measured PSNR overlap is `36.854909–39.351276 dB`. Linear interpolation within that overlap at five evenly spaced points gives:

- Ours better Top-25 Local PSNR: `5/5`;
- Ours lower P95: `5/5`;
- Ours lower LPIPS: `5/5`;
- Ours lower Global CIEDE2000: `5/5`;
- BER30: better `0/5`, tied `1/5`, worse `4/5`, with ties classified using `1e-9` tolerance.

Therefore the local/perceptual advantage persists across the overlapping operating range under the declared interpolation diagnostic, but BER30 is mixed and no total curve dominance is supported.

## 5. Remaining weaknesses

- Training stochasticity from a common frozen source is not yet tested for continuation seeds23/42.
- The new Global+OKLab-only control is negative relative to Global, so the paper should not claim a standalone OKLab improvement from this control.
- BER intervals include zero and the operating curve shows mixed BER30; crop-robustness wording must remain approximate.
- Gini is not aligned with every component comparison and must remain diagnostic.
- External Figure 2 remains qualitative/display-space evidence, not a strict quantitative benchmark.
- The reference bibliography still has no verified entries in the authoritative source set.

## 6. Experiments that should NOT be run

- Further hyperparameter search for `lambda_OK`, Top-k fraction, patch size, stride, message weight, or continuation length: the frozen method is already defined and project-test retuning is prohibited.
- More external-baseline tuning: it would expand the qualitative comparison and introduce protocol ambiguity without addressing the main internal causal question.
- Additional Top-k/patch-size sweeps: existing Hard Local-Tail evidence plus the component controls already answer the local-tail contribution question; more sweeps risk multiplicity and method drift.
- Full end-to-end multi-seed retraining: it is not the registered next question and would conflate source-training variance with continuation variance.

## 7. Recommended next experiment: continuation-seed sensitivity

This is worth running after explicit approval, but not in the current audit turn.

Definition: use the same frozen seed17 epoch-100 source checkpoint for Global, Hard Local-Tail, and Ours, with continuation seeds `17, 23, 42`. Align data order where appropriate, message RNG, crop RNG, training protocol, and evaluation protocol. Call it **continuation-seed sensitivity**, not multi-seed reproducibility.

Why it is valuable: the bootstrap establishes image-level stability but cannot test optimization stochasticity. This experiment directly addresses whether the observed continuation effect persists from one common source under different random streams.

Protocol risk: using one source for all seeds is intentionally a different question from end-to-end reproducibility. The reports must label it precisely and must not replace the frozen seed17 result. If RNG alignment or source reuse cannot be implemented exactly, stop rather than guess.

## 8. Second-priority experiment: training/inference overhead measurement

This is the safest low-cost addition. Measure training iteration time, peak GPU memory, inference latency, parameter count, and confirm unchanged inference architecture for the frozen Global and Ours checkpoints. Use a fixed warm-up/repetition protocol and report hardware/software metadata. No new training or project-test selection is needed.

Why second: it adds reviewer value about practicality and reinforces that the method adds training losses rather than an inference module, with low scientific ambiguity and low compute cost.

## 9. Optional low-cost experiment: frozen cross-dataset evaluation

A frozen cross-dataset evaluation could add external-validity value without retraining, but it requires a pre-specified, legally available, resolution-compatible dataset and a fixed no-development-selection protocol. Without that choice already frozen, the interpretation risk is higher than the overhead measurement. Run only if the dataset and metric protocol can be registered before inspection; otherwise defer.

## 10. Stop condition for experimental work

Stop experimental work after continuation-seed sensitivity and overhead measurement are complete, or sooner if the exact registered protocol cannot be maintained. Do not launch another search, modify the frozen method, update `paper_bundle/`, or integrate any extension result into the paper without explicit approval. Any approved new result must remain supplementary until separately reviewed.

## Decision table

| Experiment | Expected reviewer value | Compute cost | Scientific risk | Recommendation | Reason |
|---|---|---:|---|---|---|
| Continuation-seed sensitivity from common frozen source | High | High | Medium | **Run first after approval** | Directly tests optimization stochasticity left open by image bootstrap; must be labeled separately from reproducibility. |
| Training/inference overhead measurement | Medium | Low | Low | **Run second** | Low-risk practical evidence; reinforces unchanged inference architecture. |
| Frozen cross-dataset evaluation | Medium–high | Low–medium | Medium | **Optional/defer** | Valuable only with a pre-specified dataset and no development selection; otherwise adds distribution/protocol ambiguity. |
| Further hyperparameter search | Low for current question | High | High | **Do not run** | Violates frozen-method/no-retuning boundary and increases multiplicity. |
| More external-baseline tuning | Low–medium | Medium–high | High | **Do not run** | Figure 2 is already qualitative/contextual; more tuning would not strengthen internal causal evidence. |
| More Top-k/patch-size sweeps | Low–medium | Medium–high | Medium–high | **Do not run** | Existing ablation and controls sufficiently address the question; extra sweeps risk method drift. |

## Final decision

The current frozen final method remains supported. The most valuable unresolved question is continuation-seed sensitivity from a common frozen source, and it is worth the compute cost only after explicit approval. No new training is launched by this memo.
