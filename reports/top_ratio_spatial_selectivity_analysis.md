# Top-ratio spatial selectivity analysis

This analysis uses the existing seed17 Hard Patch16/stride8 checkpoint and the fixed manifest. No retraining was performed. The stride8 grid has 225 candidate patches per image.

| Top ratio | Selected patches | Mean unique union | Median union | P10–P90 union | Mean multiplicity | Mean components | Largest component |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 12 | 11.4% | 11.3% | 9.0–13.7% | 1.68× | 3.62 | 11.4% |
| 10% | 23 | 19.4% | 19.5% | 16.0–22.0% | 1.88× | 4.56 | 20.8% |
| 15% | 34 | 26.1% | 26.2% | 21.8–29.8% | 2.06× | 4.66 | 31.6% |
| 20% | 45 | 32.5% | 32.6% | 28.9–35.6% | 2.18× | 4.68 | 45.7% |
| 25% | 57 | 38.9% | 39.5% | 34.0–42.2% | 2.30× | 4.36 | 62.4% |

Redundancy ratio, defined as selected patch area divided by unique selected area, increases from about `1.64×` at Top5 to `2.29×` at Top25.

## Diagnosis

Top25 is spatially broad under overlap. It supervises nearly 39% of the image pixels, and the largest connected selected region covers about 62% of the image on average. This is no longer a narrow extreme-tail objective.

Top10 gives about 19.4% unique coverage with 23 selected patches, while Top15 gives 26.1% with 34 selected patches. Both remain non-degenerate; Top10 is the more selective candidate.

## Decision

If one reduced-ratio training is approved, choose **Top10%** rather than Top15% because it moves union coverage into a clearly selective range while retaining 23 candidate regions. This is a single candidate, not a ratio grid.

Plots are stored under `/mnt/wmcontent/GLX/icassp/MBRS/visualizations/top_ratio_spatial_selectivity/`:

- `ratio_vs_union.png`
- `ratio_vs_multiplicity.png`
- `ratio_vs_components.png`
- `selected_regions_examples.png`
