# Full experiment lineage

## Scope rules

The formal project conclusion uses only seed17 deterministic controlled continuation from the same crop-trained Global epoch-100 checkpoint. Historical DataParallel, non-deterministic, duplicated, and full-scratch runs are retained as exploratory context only.

The machine-readable version is [full_experiment_lineage.csv](full_experiment_lineage.csv). The formal numerical source is [current_controlled_master_table.csv](current_controlled_master_table.csv).

## A. Motivation experiments

- No-crop encoder-decoder with identity noise: establishes that image quality can be high when crop robustness is not trained, but BER under crop is unacceptable.
- Crop-trained Global image MSE: establishes the robustness/visual-quality baseline.
- Patch distortion analysis: shows that crop training increases absolute distortion while not increasing the top25/global concentration ratio.

These runs answer the motivation question but are not a clean causal comparison of local losses.

## B. Historical exploratory experiments

The historical matrix varied patch size, top ratio, and global/local weight under full-scratch or mixed-topology execution. It includes Patch8, Patch16, Patch32, Patch64, PatchMean, no-crop, and local-weight variants. The results show a strong Patch16 signal and failures for coarse Patch32/64 and unstable Patch8, but these runs mix DataParallel, single GPU, file-order, and determinism conditions.

They are useful for hypothesis generation only. They must not be used as evidence that one configuration is statistically superior to another.

## C. Controlled deterministic experiments

Every controlled branch uses:

- seed17;
- the same source checkpoint and restored Adam/BatchNorm state;
- 20 continuation epochs at `lr=1e-4`;
- batch size 16 on one visible GPU;
- deterministic data order and crop RNG;
- the same fixed evaluation manifest and five attack repeats.

The controlled sequence is:

1. Global continuation reference.
2. Hard Patch16, stride16, top25%, global/local 0.5/0.5.
3. Detached Soft-tail Patch16 with temperatures 0.25, 0.5, and 1.0.
4. Hard Patch16, stride8, top25%, global/local 0.5/0.5.
5. Multi-scale `0.7 × Patch16 Top25 + 0.3 × Patch32 Top25`.
6. Excess Distortion Loss with threshold 1 and fixed scale 3.

Only the stride8 overlap branch exceeds the previous controlled Hard reference; its gain remains modest at `+0.233 dB Worst PSNR`.

## D. Failed diagnostic experiments

- PatchMean is mathematically the same image objective as Global MSE; historical differences are execution artifacts, not a new method effect.
- Patch32/64 hard selection concentrates local gradients into too few regions.
- Patch8 produces unstable, high-variance outcomes under historical execution.
- Soft T=1 becomes too close to uniform weighting.
- Excess loss has matched initial gradient scale but its active local magnitude collapses during training and degrades both global and tail quality.

## E. Current method candidates

1. Current best: seed17 Hard Patch16, stride8, top25%, global/local 0.5/0.5.
2. Reference: seed17 Hard Patch16, stride16, top25%, global/local 0.5/0.5.
3. Soft T=0.25: retained as an ablation, not the main method.
4. Multi-scale and Excess: rejected as current main methods.

No new formal training should start until the research review has selected the next question.
