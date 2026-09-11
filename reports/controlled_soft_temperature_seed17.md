# Controlled soft-tail temperature screen: seed17

All soft branches start from the same Global checkpoint and use the same 20-epoch continuation protocol. Selection is restricted to T=0.25, 0.5, and 1.0.

| Branch | PSNR | ΔPSNR | Worst PSNR | ΔWorst | Top25 MSE | ΔTop25 | P95 MSE | ΔP95 | BER30 | ΔBER30 | Soft max weight | N_eff | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T=0.25 | 36.314 | +0.086 | 35.451 | +0.136 | 0.001140 | -0.000036 | 0.001551 | -0.000062 | 0.11338 | +0.00025 | 0.16483 | 28.00 | PASS |
| T=0.5 | 36.305 | +0.077 | 35.420 | +0.104 | 0.001148 | -0.000028 | 0.001573 | -0.000040 | 0.11300 | -0.00013 | 0.07150 | 46.05 | PASS |
| T=1 | 36.250 | +0.023 | 35.353 | +0.037 | 0.001166 | -0.000010 | 0.001610 | -0.000003 | 0.11319 | +0.00006 | 0.03489 | 58.19 | FAIL |

Selected final Soft temperature: **0.25**.

Selection rule: among temperatures passing the seed17 quality/BER/oscillation gate, choose the lowest top25 patch MSE. No additional temperatures are considered.

Figures:

- `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/controlled_soft_temperature_seed17/temperature_percentile_curve.png`
- `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/controlled_soft_temperature_seed17/temperature_tail_global_cdf.png`
