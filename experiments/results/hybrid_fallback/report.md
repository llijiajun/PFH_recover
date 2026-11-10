# Hybrid fallback evidence (fifth-round review, Issue 2)

Cases: 231 reported, 45 flagged.

## Median W1 by strategy

| strategy | all 276 | reported | flagged | flagged improved | Wilcoxon p |
|---|---|---|---|---|---|
| fitted_always | 0.02525 | nan | nan | --/-- | nan |
| hybrid_fitted_good_turing | 0.02525 | 0.00946 | 14.7079 | 21/45 | 0.7122 |
| hybrid_fitted_smoothed_bayesian | 0.02525 | 0.00946 | 9.9424 | 34/45 | 2.755e-07 |
| hybrid_fitted_naive | 0.02525 | 0.00946 | 38.9925 | 0/45 | 5.684e-14 |
| good_turing_always | 0.05189 | nan | nan | --/-- | nan |
| smoothed_bayesian_always | 0.04478 | nan | nan | --/-- | nan |

The pooled median is unchanged by any hybrid because only 45 of 276
cases change and the median index falls inside the reported group.

## Table IV heavy-tail interaction

- concentrated: lowest-rate ours/GT ratio 0.85, GT better at all rates = False
- geopoor: lowest-rate ours/GT ratio 1.29, GT better at all rates = True
- mix: lowest-rate ours/GT ratio 1.26, GT better at all rates = False
- spiked: lowest-rate ours/GT ratio 0.99, GT better at all rates = False
- zipf: lowest-rate ours/GT ratio 1.58, GT better at all rates = False
