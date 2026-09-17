# Figure 2 Candidate Summary

All selection uses the fixed 50-image validation manifest. No project-test sample, encoder inference, retraining, or formal Table I result is used or changed.

**Selection caveat:** both the two samples and their ROIs are selected for the largest measured Ours-base → Ours local improvement. This is a deliberate double best-case qualitative search and is not representative-sample evidence.

| Candidate | Scientific honesty /5 | Visual difference /5 | Direct support /5 | Layout clarity /5 | ICASSP suitability /5 | Decision |
|---|---:|---:|---:|---:|---:|---|
| A. Native zoom | 5 | 3 | 5 | 5 | 5 | MAIN PAPER: direct evidence with native outputs and explicit best-case disclosure. |
| B. Residual-strength stress | 4 | 4 | 3 | 5 | 3 | Stress-only candidate; never label as native output. |
| C. Residual visualization | 5 | 5 | 5 | 4 | 4 | SUPPLEMENTARY: strongest diagnostic explanation; amplification is labelled. |
| D. Patch-error heatmap | 5 | 5 | 5 | 4 | 4 | Supplementary alternative with a shared color scale. |
| E. Color-tail visualization | 4 | 3 | 3 | 5 | 3 | Validation-only extension; incomplete preservation gate prevents a main claim. |
| F. Attack stress | 5 | 2 | 2 | 3 | 2 | Metrics-only exploration; no attack image promoted into Figure 2. |

## Recommendation

- Main paper: **fig2_final** combines unmodified native full/zoom panels with a clearly separated residual ×10 diagnostic row, because the native difference alone is subtle.
- Supplementary: **Figure2-C-residual**. It makes the local residual redistribution inspectable while explicitly labelling ×10 as visualization-only.

## Stress and attack status

- Figure2-B uses equal α=1.5 after RGB[0,1] clipping; per-method clip fractions and metrics are in `fig2_strength_sweep.csv`.
- BER@30/40/50 is real method-specific decoder inference on the fixed crop-mask repeats, including every alpha after clipping.
- Crop, fixed-seed Gaussian noise, and real JPEG attack BER were run for all three saved methods. They remain exploratory and are not promoted to the main qualitative figure.
- Ours + OKLab is a validation-only global extension and did not pass the complete predeclared preservation gate.
