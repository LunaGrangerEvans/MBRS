# Factorial Component Bootstrap Summary

The four fixed-RGB-weight cells are bootstrapped at the paired image level using the corrected paper PSNR estimator. For PSNR, each replicate recomputes aggregate mean MSE per cell and then converts it to PSNR. Other metrics use their documented image-level values; BER30 resamples per-image BER contributions.

- Resamples: `20,000`
- Analysis seed: `20260916`
- Effects: OKLab without Tail, OKLab with Tail, Tail without OKLab, Tail with OKLab, and the interaction contrast.
- Interval language is limited to “interval excludes/includes zero.”

| Effect | Metric | Observed | 2.5 percentile | 97.5 percentile | Bootstrap SE | Interval includes zero? |
|---|---|---:|---:|---:|---:|:---:|
| OKLab_without_Tail | PSNR | 0.779356563 | 0.744868811 | 0.813428937 | 0.0174693938 | No |
| OKLab_with_Tail | PSNR | 0.412608788 | 0.385126924 | 0.439821251 | 0.0140061293 | No |
| Tail_without_OKLab | PSNR | 1.27781453 | 1.2272234 | 1.33121951 | 0.0266967026 | No |
| Tail_with_OKLab | PSNR | 0.911066758 | 0.866084715 | 0.960321738 | 0.0241825005 | No |
| interaction | PSNR | -0.366747775 | -0.380422892 | -0.351978999 | 0.00726407748 | No |
| OKLab_without_Tail | Top-25 Local PSNR | 0.759838602 | 0.723384231 | 0.794625183 | 0.0181107484 | No |
| OKLab_with_Tail | Top-25 Local PSNR | 0.387518925 | 0.362724942 | 0.41070982 | 0.0122648261 | No |
| Tail_without_OKLab | Top-25 Local PSNR | 1.30665479 | 1.2494542 | 1.36360665 | 0.0293320758 | No |
| Tail_with_OKLab | Top-25 Local PSNR | 0.934335114 | 0.886093371 | 0.98383373 | 0.0251874636 | No |
| interaction | Top-25 Local PSNR | -0.372319677 | -0.390011316 | -0.353527021 | 0.00927086952 | No |
| OKLab_without_Tail | P95 MSE | -5.85254472e-05 | -6.06764187e-05 | -5.6611518e-05 | 1.03219499e-06 | No |
| OKLab_with_Tail | P95 MSE | -2.24625882e-05 | -2.31597902e-05 | -2.17816717e-05 | 3.54649194e-07 | No |
| Tail_without_OKLab | P95 MSE | -9.68566646e-05 | -0.000102587356 | -9.162733e-05 | 2.79755004e-06 | No |
| Tail_with_OKLab | P95 MSE | -6.07938055e-05 | -6.49305788e-05 | -5.68749229e-05 | 2.04261689e-06 | No |
| interaction | P95 MSE | 3.60628591e-05 | 3.41299992e-05 | 3.82896059e-05 | 1.06827434e-06 | No |
| OKLab_without_Tail | P99 MSE | -6.03637023e-05 | -6.28889411e-05 | -5.8053192e-05 | 1.23644814e-06 | No |
| OKLab_with_Tail | P99 MSE | -2.28404557e-05 | -2.36262715e-05 | -2.20628717e-05 | 3.99282773e-07 | No |
| Tail_without_OKLab | P99 MSE | -0.000101536544 | -0.000108451097 | -9.52018528e-05 | 3.37666544e-06 | No |
| Tail_with_OKLab | P99 MSE | -6.40132975e-05 | -6.89674172e-05 | -5.94185961e-05 | 2.41707553e-06 | No |
| interaction | P99 MSE | 3.75232466e-05 | 3.52629443e-05 | 3.99661908e-05 | 1.20542938e-06 | No |
| OKLab_without_Tail | LPIPS | -0.000252740434 | -0.000311958597 | -0.00019508109 | 2.98485298e-05 | No |
| OKLab_with_Tail | LPIPS | -0.000298064452 | -0.00036160571 | -0.00023713946 | 3.17614745e-05 | No |
| Tail_without_OKLab | LPIPS | -0.000403105833 | -0.000525796248 | -0.00029096405 | 6.00304562e-05 | No |
| Tail_with_OKLab | LPIPS | -0.000448429851 | -0.000558019135 | -0.000359235752 | 5.10224355e-05 | No |
| interaction | LPIPS | -4.5324018e-05 | -0.000111935754 | 1.86662028e-05 | 3.34679387e-05 | Yes |
| OKLab_without_Tail | Global CIEDE2000 | -0.449969101 | -0.472308781 | -0.426294972 | 0.0117462867 | No |
| OKLab_with_Tail | Global CIEDE2000 | -0.230699067 | -0.245431625 | -0.215594789 | 0.00763139472 | No |
| Tail_without_OKLab | Global CIEDE2000 | -0.542383244 | -0.568679212 | -0.516095573 | 0.0135187409 | No |
| Tail_with_OKLab | Global CIEDE2000 | -0.323113211 | -0.347283521 | -0.302512524 | 0.0115009914 | No |
| interaction | Global CIEDE2000 | 0.219270033 | 0.206887219 | 0.231136041 | 0.00622912621 | No |
| OKLab_without_Tail | Top10 CIEDE2000 | -0.64309626 | -0.672459404 | -0.610460937 | 0.0157510318 | No |
| OKLab_with_Tail | Top10 CIEDE2000 | -0.303437085 | -0.320117986 | -0.285144062 | 0.00898228157 | No |
| Tail_without_OKLab | Top10 CIEDE2000 | -0.782986015 | -0.813679988 | -0.751253791 | 0.0159468171 | No |
| Tail_with_OKLab | Top10 CIEDE2000 | -0.44332684 | -0.47325582 | -0.416347464 | 0.0144941273 | No |
| interaction | Top10 CIEDE2000 | 0.339659175 | 0.322036982 | 0.355783819 | 0.00863913416 | No |
| OKLab_without_Tail | Gini | 0.00504779935 | 0.00362234794 | 0.00654114336 | 0.000746379525 | No |
| OKLab_with_Tail | Gini | 0.00523331318 | 0.00407065444 | 0.00656331145 | 0.000631355663 | No |
| Tail_without_OKLab | Gini | -0.0014186071 | -0.00362906005 | 0.000874384968 | 0.00114520998 | Yes |
| Tail_with_OKLab | Gini | -0.00123309327 | -0.003070065 | 0.000669099891 | 0.000948950042 | Yes |
| interaction | Gini | 0.000185513824 | -0.000647616276 | 0.001018819 | 0.000425596426 | Yes |
| OKLab_without_Tail | BER30 | 0.0001875 | -0.0005 | 0.0009375 | 0.000364602431 | Yes |
| OKLab_with_Tail | BER30 | 0.0001875 | -0.0004375 | 0.0008125 | 0.000310357129 | Yes |
| Tail_without_OKLab | BER30 | -0.000125 | -0.0006875 | 0.0005 | 0.000304652777 | Yes |
| Tail_with_OKLab | BER30 | -0.000125 | -0.00075 | 0.0005 | 0.000330348051 | Yes |
| interaction | BER30 | -2.77555756e-17 | -0.0008125 | 0.0008125 | 0.000420857314 | Yes |

The factorial bootstrap quantifies uncertainty in cell contrasts; it does not change the frozen method, establish universal superiority, or justify the word “synergy” by itself.
