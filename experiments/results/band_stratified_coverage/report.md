# Stratified per-frequency band coverage (new R3 evidence)

Independent grid. Population Zipf n=100000, vocab=30000, p=0.1, K=120, j<=512.

## Per configuration

| alpha | seed | pooled | populated bins | zero-mass bins | n populated | a0 |
|---|---|---|---|---|---|---|
| 1.0 | 0 | 0.9043 | 0.6797 | 1.0000 | 153 | 0.001 |
| 1.0 | 1 | 0.8887 | 0.6200 | 1.0000 | 150 | 0.001 |
| 1.0 | 2 | 0.9023 | 0.6667 | 1.0000 | 150 | 0.001 |
| 1.5 | 0 | 0.9512 | 0.7549 | 1.0000 | 102 | 0.001 |
| 1.5 | 1 | 0.9531 | 0.7745 | 0.9976 | 102 | 0.001 |
| 1.5 | 2 | 0.9492 | 0.7400 | 1.0000 | 100 | 0.001 |
| 2.0 | 0 | 0.9746 | 0.7818 | 0.9978 | 55 | 0.001 |
| 2.0 | 1 | 0.9727 | 0.7347 | 0.9978 | 49 | 0.001 |
| 2.0 | 2 | 0.9805 | 0.8421 | 0.9978 | 57 | 0.001 |

## Pooled over 4608 bins (9 configurations)

Overall coverage 0.9418.

### By true bin mass

| layer | n bins | covered | coverage | Wilson CI95 |
|---|---|---|---|---|
| head_gt_100 | 24 | 2 | 0.0833 | [0.0232, 0.2585] |
| mid_11_100 | 106 | 52 | 0.4906 | [0.3974, 0.5844] |
| rare_1_10 | 788 | 600 | 0.7614 | [0.7304, 0.7899] |
| zero_mass | 3690 | 3686 | 0.9989 | [0.9972, 0.9996] |

### By frequency index

| layer | n bins | covered | coverage | Wilson CI95 |
|---|---|---|---|---|
| j_21_512 | 4428 | 4243 | 0.9582 | [0.9519, 0.9637] |
| j_2_5 | 36 | 21 | 0.5833 | [0.4220, 0.7286] |
| j_6_20 | 135 | 76 | 0.5630 | [0.4787, 0.6437] |
| j_eq_1 | 9 | 0 | 0.0000 | [0.0000, 0.2991] |

## Prior parameter sensitivity

| delta | a0 | b0 | pooled | populated | median width |
|---|---|---|---|---|---|
| -0.50 | 0.0005 | 0.06802 | 0.9160 | 0.7190 | 1.000 |
| -0.25 | 0.00075 | 0.102 | 0.9141 | 0.7124 | 1.000 |
| -0.10 | 0.0009 | 0.1224 | 0.9102 | 0.6993 | 1.000 |
| +0.00 | 0.001 | 0.136 | 0.9062 | 0.6863 | 1.000 |
| +0.10 | 0.0011 | 0.1497 | 0.8965 | 0.6536 | 1.000 |
| +0.25 | 0.00125 | 0.1701 | 0.8887 | 0.6275 | 1.000 |
| +0.50 | 0.0015 | 0.2041 | 0.8809 | 0.6013 | 1.000 |

## Split-conformal calibration of populated bins

Calibration on 6 configurations, holdout on 3, q = 1.950.

| holdout | pooled | populated | zero_mass | rare | mid | head |
|---|---|---|---|---|---|---|
| holdout_raw | 0.9440 | 0.7231 | 0.9992 | 0.7774 | 0.4412 | 0.1250 |
| holdout_calibrated | 0.9798 | 0.8990 | 1.0000 | 0.9623 | 0.5882 | 0.1250 |

Fitted in-sample populated coverage 0.9034.


### Layer-wise inflations

q by layer: head_gt_100=222.80, mid_11_100=20.00, rare_1_10=1.00, zero_mass=0.00

| split | populated | median width | zero_mass | rare | mid | head |
|---|---|---|---|---|---|---|
| calibration_in_sample | 0.9640 | 1.00 | 0.9988 | 0.9694 | 0.9167 | 1.0000 |
| holdout | 0.9414 | 1.00 | 0.9992 | 0.9547 | 0.8529 | 0.8750 |

Layer-wise inflations are fitted on six configurations. Layers with few calibration bins have unstable quantiles, so this is a diagnostic of scale sensitivity rather than a deployable guarantee.

Headline: Pooled coverage is near nominal, but populated frequency bins are under-covered. A single conformal inflation fitted on six configurations is evaluated on three held-out configurations.

Caveats: Frequency bins inside one configuration are dependent. Wilson intervals are descriptive and are not a simultaneous guarantee.

Empirical holdout coverage on three unseen configurations. Bins within a configuration are dependent, so no simultaneous per-bin guarantee is claimed.

This is per-frequency band coverage on j<=512. It does not validate the 95% per-type count interval or the L1 model-content figure.
