# Stochastic Continuation-Seed Sensitivity Report

This report measures stochastic continuation-seed sensitivity from the same frozen seed17 epoch-100 crop-trained Global source checkpoint. It is not full end-to-end multi-seed reproducibility. Seed17 reuses the valid frozen endpoint; seeds23 and42 are new continuations from the identical source.

- Seeds: `17, 23, 42`.
- Source SHA-256: `1a82ec4f9559c5861fdcbd51ddd76b4ecce2507e7ecf8c1a7e63a9ea6cce2907`.
- All methods: 20 epochs, `lr=1e-4`, batch16, workers0, `RandomCrop(0.3,1.0)`, restored model/BatchNorm/Adam state, deterministic settings.
- No seed, checkpoint, or hyperparameter was selected using project-test performance.
- All values below are natural source-resolution 128×128 project-test metrics.

## Individual seed values

| Method | Seed | PSNR | Top-25 Local PSNR | P95 | P99 | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Global | 17 | 36.263172 | 35.522213 | 3.016532060e-04 | 3.199585441e-04 | 0.00234960 | 3.535857 | 5.159104 | 0.095007 | 0.11312500 |
| Global | 23 | 36.448988 | 35.700641 | 2.898393316e-04 | 3.077535522e-04 | 0.00234518 | 3.551365 | 5.200859 | 0.096139 | 0.11418750 |
| Global | 42 | 36.395817 | 35.695057 | 2.905324678e-04 | 3.085268830e-04 | 0.00219537 | 3.541021 | 5.173080 | 0.092374 | 0.11518750 |
| Hard Local-Tail | 17 | 36.442300 | 35.754376 | 2.844551340e-04 | 3.010726963e-04 | 0.00221518 | 3.484790 | 5.075666 | 0.086897 | 0.11293750 |
| Hard Local-Tail | 23 | 36.710316 | 36.008143 | 2.694039427e-04 | 2.861353528e-04 | 0.00216537 | 3.446180 | 5.031469 | 0.090237 | 0.11450000 |
| Hard Local-Tail | 42 | 36.533018 | 35.880099 | 2.776209903e-04 | 2.944954163e-04 | 0.00194561 | 3.500537 | 5.116009 | 0.085759 | 0.11468750 |
| Ours | 17 | 36.854909 | 36.141895 | 2.619925669e-04 | 2.782322237e-04 | 0.00191711 | 3.254091 | 4.772229 | 0.092130 | 0.11312500 |
| Ours | 23 | 37.255656 | 36.526187 | 2.416170492e-04 | 2.579666411e-04 | 0.00202403 | 3.154185 | 4.635024 | 0.096904 | 0.11443750 |
| Ours | 42 | 37.015743 | 36.324798 | 2.532987784e-04 | 2.695664002e-04 | 0.00183128 | 3.209215 | 4.704172 | 0.093529 | 0.11481250 |

## Mean ± standard deviation across continuation seeds

| Method | PSNR | Top-25 Local PSNR | P95 | P99 | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Global | 36.369326 ± 0.095698 | 35.639303 ± 0.101442 | 2.940083352e-04 ± 6.629717004e-06 | 3.120796598e-04 ± 6.834261009e-06 | 0.00229671 ± 0.00008780 | 3.542748 ± 0.007896 | 5.177681 ± 0.021255 | 0.094507 ± 0.001931 | 0.11416667 ± 0.00103141 |
| Hard Local-Tail | 36.561878 ± 0.136319 | 35.880872 ± 0.126885 | 2.771600223e-04 ± 7.536176611e-06 | 2.939011552e-04 ± 7.486382131e-06 | 0.00210872 ± 0.00014343 | 3.477169 ± 0.027968 | 5.074382 ± 0.042285 | 0.087631 ± 0.002327 | 0.11404167 ± 0.00096082 |
| Ours | 37.042103 ± 0.201670 | 36.330960 ± 0.192221 | 2.523027982e-04 ± 1.022420720e-05 | 2.685884216e-04 ± 1.016812622e-05 | 0.00192414 ± 0.00009657 | 3.205831 ± 0.050039 | 4.703809 ± 0.068603 | 0.094188 ± 0.002454 | 0.11412500 ± 0.00088609 |

## Paired method differences by continuation seed

Differences are computed within each continuation seed. For PSNR and local PSNR, positive favors the first method; for lower-is-better metrics, negative favors the first method.

### Ours - Global

| Seed | PSNR | Top-25 Local PSNR | P95 | P99 | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.591737 | 0.619682 | -3.966063916e-05 | -4.172632043e-05 | -0.00043249 | -0.281766 | -0.386875 | -0.002877 | -0.00000000 |
| 23 | 0.806668 | 0.825547 | -4.822228242e-05 | -4.978691116e-05 | -0.00032115 | -0.397179 | -0.565835 | 0.000765 | 0.00025000 |
| 42 | 0.619927 | 0.629741 | -3.723368944e-05 | -3.896048285e-05 | -0.00036409 | -0.331806 | -0.468908 | 0.001155 | -0.00037500 |

### Hard - Global

| Seed | PSNR | Top-25 Local PSNR | P95 | P99 | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.179128 | 0.232163 | -1.719807209e-05 | -1.888584775e-05 | -0.00013442 | -0.051067 | -0.083438 | -0.008110 | -0.00018750 |
| 23 | 0.261329 | 0.307502 | -2.043538887e-05 | -2.161819939e-05 | -0.00017981 | -0.105184 | -0.169391 | -0.005902 | 0.00031250 |
| 42 | 0.137202 | 0.185042 | -1.291147753e-05 | -1.403146674e-05 | -0.00024975 | -0.040485 | -0.057071 | -0.006615 | -0.00050000 |

### Ours - Hard

| Seed | PSNR | Top-25 Local PSNR | P95 | P99 | LPIPS | Global CIEDE2000 | Top10 CIEDE2000 | Gini | BER30 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.412609 | 0.387519 | -2.246256707e-05 | -2.284047268e-05 | -0.00029806 | -0.230699 | -0.303437 | 0.005233 | 0.00018750 |
| 23 | 0.545340 | 0.518045 | -2.778689355e-05 | -2.816871177e-05 | -0.00014134 | -0.291995 | -0.396445 | 0.006667 | -0.00006250 |
| 42 | 0.482725 | 0.444699 | -2.432221190e-05 | -2.492901612e-05 | -0.00011433 | -0.291321 | -0.411837 | 0.007770 | 0.00012500 |

For `Ours - Global`, the favorable direction is consistent across all three continuation seeds for: PSNR, Top-25 Local PSNR, P95, P99, LPIPS, Global CIEDE2000, Top10 CIEDE2000.

For `Hard - Global`, the favorable direction is consistent across all three continuation seeds for: PSNR, Top-25 Local PSNR, P95, P99, LPIPS, Global CIEDE2000, Top10 CIEDE2000.

For `Ours - Hard`, the favorable direction is consistent across all three continuation seeds for: PSNR, Top-25 Local PSNR, P95, P99, LPIPS, Global CIEDE2000, Top10 CIEDE2000.

## Interpretation

The primary fidelity trend for Ours versus Global persists across all three stochastic continuation seeds from a common frozen source checkpoint: PSNR and Top-25 Local PSNR are higher, while P95, P99, LPIPS, and CIEDE2000 are lower for Ours in every seed. This supports the cautious sentence: **“The fidelity trend persists across stochastic continuation seeds from a common frozen source checkpoint.”**

This consistency does not extend to every diagnostic. Gini is mixed for Ours versus Global, and BER30 is mixed (`0`, positive, and negative small differences across seeds). No BER or Gini superiority claim is made. The report also does not claim full end-to-end reproducibility because the source model was held fixed.

The full per-seed rows are in `continuation_seed_summary.csv`; paired deltas are in `continuation_seed_effects.csv`.
