# Calibration of the 90% per-frequency band

Source `results/thickening-plausible-set-validation/intervals_three_layer.json`.

- Per-seed empirical coverage at K=120: 0.9291, 0.9359, 0.9126.
- Range 0.913 to 0.936, median 0.929.
- Median band width per seed: 1, 1, 1.
- Number of frequency bins per seed: 296, 281, 286.

## Stability in the number of draws

| K | coverage range | median width |
|---|---|---|
| 120 | 0.913--0.936 | 1 |
| 500 | 0.919--0.928 | 1 |
| 2000 | 0.923--0.935 | 1 |

## What the interval contains

- **prior_refitting**: The prior is fitted once per case and passed as a fixed array to plausible_set. The K posterior draws do not refit it, and repeated thinnings in this benchmark do not refit it either.
- **uncertainty_included**: posterior sampling variability of the population counts conditional on the fitted prior
- **uncertainty_excluded**: prior parameter estimation error, prior misspecification, and the unknown abundance of types absent from the sample
- **distinct_output**: The 95% per-type intervals and the L1 model-content figure of 0.953 are different quantities and do not validate this band.
