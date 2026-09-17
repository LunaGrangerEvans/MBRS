# Figure 2 Final Selection

Selected the combined native + explicitly labelled residual diagnostic as `fig2_final.{png,pdf,pptx}` and `fig2_final_editable.pptx`.

- Samples: 0805, 0844 (validation indices 4, 43).
- Quantitative basis: native32 Top-25 local PSNR improvements are +0.524719 dB and +0.513161 dB; P95-MSE reductions are +2.701124686e-05 and +4.164558777e-05.
- Shared ROIs: 0805 uses (x=36, y=52, size=32); 0844 uses (x=28, y=24, size=32).
- ROI rule: among 32/40/48/64-pixel candidates on a stride-4 scan, maximize mean local MSE(Ours-base) minus mean local MSE(Ours), subject only to a border margin and original-image texture eligibility. One ROI is shared by every method.
- Sample and ROI selection are both best-case selections for Ours-base → Ours and are not representative.
- Native panels come directly from saved float tensors. No enhancement, sharpening, saturation, contrast, or per-method scaling is applied.
- Residual-strength scaling in the final figure: no (alpha = 1.0 native output). The separate Figure2-B stress candidate uses the same alpha = 1.5 for every method and is not native output.
- Residual amplification in the final figure: yes, ×10, in rows explicitly labelled `visualization only`; the full and zoom rows remain native.
- Attack in the final figure: no. Crop/noise/JPEG stress was measured on validation only and not selected.
- Checkpoint modification: none; checkpoints were read only, and decoders were used only for BER evaluation.
- Formal-test/Table I modification: none; no project-test manifest or formal result was used for selection.
- PPTX contains separate raster panels plus editable text, method names, captions, red ROI rectangles, and neutral panel-border vector shapes. No arrows are used.
- Supplementary recommendation: `fig2_C_residual`.
- Ours + OKLab remains validation-only and did not pass the complete validation gate.
