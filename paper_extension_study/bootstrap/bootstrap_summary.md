# Paired Image-Level Bootstrap Summary

This supplementary analysis reuses the frozen 50-image project-test per-image output explicitly referenced by the authoritative bundle. It does not rerun training or select any hyperparameter. Each bootstrap resample samples image indices with replacement and preserves method pairing; individual bits are never treated as IID bootstrap units.

- Resamples: `20,000`
- Analysis seed: `20260916`
- Difference convention: `second method − first method`.
- For higher-is-better metrics, positive differences favor the second method; for lower-is-better metrics, negative differences favor the second method.
- The interval wording below is descriptive: an interval either excludes or includes zero. No p-value or generic statistical-significance claim is made.

## Results

| Comparison | Metric | Observed difference | 2.5 percentile | 97.5 percentile | Bootstrap SE | CI crosses zero | Direction |
|---|---|---:|---:|---:|---:|:---:|---|
| global_vs_ours | PSNR | 0.596953233 | 0.558864805 | 0.640176282 | 0.0207791777 | no | higher_is_better |
| global_vs_ours | Top-25 Local PSNR | 0.61968201 | 0.57913228 | 0.664227127 | 0.0218986815 | no | higher_is_better |
| global_vs_ours | P95 patch MSE | -3.96606381e-05 | -4.33451508e-05 | -3.62061527e-05 | 1.83097228e-06 | no | lower_is_better |
| global_vs_ours | P99 patch MSE | -4.17262951e-05 | -4.58671857e-05 | -3.78152806e-05 | 2.0573674e-06 | no | lower_is_better |
| global_vs_ours | LPIPS | -0.000432485252 | -0.000557054864 | -0.000341752803 | 5.56899209e-05 | no | lower_is_better |
| global_vs_ours | Global CIEDE2000 | -0.281766059 | -0.305251388 | -0.261962791 | 0.0111090178 | no | lower_is_better |
| global_vs_ours | Top10 CIEDE2000 | -0.386875057 | -0.415715378 | -0.361141439 | 0.0139336785 | no | lower_is_better |
| global_vs_ours | Gini | -0.0028765271 | -0.00441913123 | -0.00128797928 | 0.000802387121 | no | lower_is_better |
| global_vs_ours | BER100 | 0 | 0 | 0 | 0 | yes | lower_is_better |
| global_vs_ours | BER70 | 0 | 0 | 0 | 0 | yes | lower_is_better |
| global_vs_ours | BER50 | 6.25e-05 | 0 | 0.0001875 | 6.16851773e-05 | yes | lower_is_better |
| global_vs_ours | BER40 | 6.25e-05 | -0.0003125 | 0.0004375 | 0.000186559699 | yes | lower_is_better |
| global_vs_ours | BER30 | -1.38777878e-18 | -0.0005 | 0.0005625 | 0.000266347232 | yes | lower_is_better |
| global_vs_hard | PSNR | 0.165960639 | 0.124230161 | 0.21254858 | 0.0225870979 | no | higher_is_better |
| global_vs_hard | Top-25 Local PSNR | 0.232163085 | 0.189527141 | 0.278355095 | 0.0225603697 | no | higher_is_better |
| global_vs_hard | P95 patch MSE | -1.71980499e-05 | -2.09462756e-05 | -1.36955688e-05 | 1.85330214e-06 | no | lower_is_better |
| global_vs_hard | P99 patch MSE | -1.88858394e-05 | -2.30699737e-05 | -1.49666192e-05 | 2.07336331e-06 | no | lower_is_better |
| global_vs_hard | LPIPS | -0.0001344208 | -0.0002265356 | -6.09110947e-05 | 4.23022525e-05 | no | lower_is_better |
| global_vs_hard | Global CIEDE2000 | -0.0510669923 | -0.0742616588 | -0.0317837107 | 0.010948674 | no | lower_is_better |
| global_vs_hard | Top10 CIEDE2000 | -0.0834379721 | -0.113206294 | -0.0579387581 | 0.0140635242 | no | lower_is_better |
| global_vs_hard | Gini | -0.00810984028 | -0.0096820922 | -0.00660990977 | 0.00079133113 | no | lower_is_better |
| global_vs_hard | BER100 | 0 | 0 | 0 | 0 | yes | lower_is_better |
| global_vs_hard | BER70 | 0 | 0 | 0 | 0 | yes | lower_is_better |
| global_vs_hard | BER50 | 6.25e-05 | 0 | 0.0001875 | 6.1430412e-05 | yes | lower_is_better |
| global_vs_hard | BER40 | 0.0001875 | -0.0001875 | 0.0005625 | 0.000186220823 | yes | lower_is_better |
| global_vs_hard | BER30 | -0.0001875 | -0.00075 | 0.000375 | 0.000285960844 | yes | lower_is_better |
| hard_vs_ours | PSNR | 0.430992595 | 0.408383581 | 0.452152449 | 0.0111214265 | no | higher_is_better |
| hard_vs_ours | Top-25 Local PSNR | 0.387518925 | 0.362724942 | 0.41070982 | 0.0122648261 | no | higher_is_better |
| hard_vs_ours | P95 patch MSE | -2.24625882e-05 | -2.31597902e-05 | -2.17816717e-05 | 3.54649194e-07 | no | lower_is_better |
| hard_vs_ours | P99 patch MSE | -2.28404557e-05 | -2.36262715e-05 | -2.20628717e-05 | 3.99282773e-07 | no | lower_is_better |
| hard_vs_ours | LPIPS | -0.000298064452 | -0.00036160571 | -0.00023713946 | 3.17614745e-05 | no | lower_is_better |
| hard_vs_ours | Global CIEDE2000 | -0.230699067 | -0.245431625 | -0.215594789 | 0.00763139472 | no | lower_is_better |
| hard_vs_ours | Top10 CIEDE2000 | -0.303437085 | -0.320117986 | -0.285144062 | 0.00898228157 | no | lower_is_better |
| hard_vs_ours | Gini | 0.00523331318 | 0.00407065444 | 0.00656331145 | 0.000631355663 | no | lower_is_better |
| hard_vs_ours | BER100 | 0 | 0 | 0 | 0 | yes | lower_is_better |
| hard_vs_ours | BER70 | 0 | 0 | 0 | 0 | yes | lower_is_better |
| hard_vs_ours | BER50 | 0 | -0.0001875 | 0.0001875 | 8.80587338e-05 | yes | lower_is_better |
| hard_vs_ours | BER40 | -0.000125 | -0.0005625 | 0.000375 | 0.000234269913 | yes | lower_is_better |
| hard_vs_ours | BER30 | 0.0001875 | -0.0004375 | 0.0008125 | 0.000310357129 | yes | lower_is_better |

## Interpretation boundary

The observed differences and paired bootstrap intervals quantify image-level stability of the frozen comparisons. They do not alter the frozen method, replace the formal project-test table, establish universal superiority, or authorize retuning. Use the exact metric-space labels from the paper evidence audit when citing these results.
