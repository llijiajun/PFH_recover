# Join-size Q-error from a recovered frequency histogram

Paired cases 276. Primary convention uses the UNCAPPED ground truth
SUM_v f_v^2 on the 208 cases where capping the truth at j <= 256 changes
it by less than one percent. All spectra still share the estimator support
j <= 256. Q-error is max(est/true, true/est).

## Primary convention

| estimator | finite | geomean | median | p90 | max | within 2x |
|---|---|---|---|---|---|---|
| catalog_uniform_true | 208 | 1.091 | 1.000 | 1.371 | 3.39 | 98.1% |
| catalog_uniform_naive | 207 | 1.302 | 1.192 | 1.939 | 5.8 | 92.8% |
| raw_sample | 207 | 3.590 | 3.194 | 9.806 | 20.8 | 20.8% |
| good_turing | 201 | 3.598 | 3.380 | 13.000 | 44 | 29.9% |
| sb_linear | 207 | 1.332 | 1.211 | 1.935 | 5.8 | 90.8% |
| ours_fitted | 207 | 1.244 | 1.131 | 1.680 | 5.79 | 92.8% |
| ours_oracle | 207 | 1.195 | 1.096 | 1.598 | 5.8 | 94.2% |

## Paired comparisons against ours_fitted

| competitor | n | ours wins | Wilcoxon p |
|---|---|---|---|
| catalog_uniform_true | 207 | 26.1% | 1 |
| catalog_uniform_naive | 207 | 62.3% | 6.29e-08 |
| raw_sample | 207 | 94.2% | 8.64e-32 |
| good_turing | 201 | 89.1% | 8.61e-31 |
| sb_linear | 207 | 64.7% | 1.03e-10 |
| ours_oracle | 207 | 38.6% | 1 |

## Reading, including results unfavorable to our framing

`catalog_uniform_true` uses the TRUE row count and TRUE NDV, so it is an
ORACLE baseline that no deployed catalog can reach. On the primary cases it
achieves geomean 1.091 and
98.1% within 2x, while ours_fitted
reaches 1.244. We do NOT beat this oracle.
The paired win rate is 26.1%.
The 810x example in the introduction is a constructed extreme, not the typical
real column. Most real columns are close enough to uniform that two scalars
already give a good join-size estimate.

The honest gain is over DEPLOYABLE alternatives. The sample plug-in catalog
reaches 1.302 geomean and ours_fitted
wins 62.3% of paired
cases (p = 6.3e-08).
The raw sample is 3.590 geomean.
Scalar NDV estimators remain structurally unable to report entropy, rare-value
mass, or a calibrated interval, which is where the full histogram earns its cost.

The oracle prior row is an information ceiling and is not deployable.

## Sensitivity to the ground-truth convention

Of 276 cases, 68 have population counts above
the common support. On those cases a capped denominator produces enormous
spurious Q-errors, so it is not used for the primary claim.


### Uncapped truth, all cases (n = 276)

| estimator | finite | geomean | median | p90 | max | within 2x |
|---|---|---|---|---|---|---|
| catalog_uniform_true | 276 | 3.196 | 1.051 | 160.414 | 807 | 73.9% |
| catalog_uniform_naive | 275 | 4.555 | 1.290 | 266.316 | 2.42e+03 | 69.8% |
| raw_sample | 275 | 21.921 | 4.252 | 19280.629 | 1.72e+07 | 17.1% |
| good_turing | 269 | 57.994 | 5.000 | 203512.199 | 9.9e+12 | 22.7% |
| sb_linear | 275 | 10.922 | 1.337 | 24035.236 | 1.73e+07 | 68.4% |
| ours_fitted | 275 | 10.349 | 1.221 | 23954.627 | 1.72e+07 | 69.8% |
| ours_oracle | 275 | 9.602 | 1.178 | 19414.370 | 1.7e+07 | 72.4% |

### Capped truth, all cases (n = 276)

| estimator | finite | geomean | median | p90 | max | within 2x |
|---|---|---|---|---|---|---|
| catalog_uniform_true | 276 | 5.052 | 1.051 | 145.157 | 2.23e+07 | 75.4% |
| catalog_uniform_naive | 275 | 4.835 | 1.290 | 102.484 | 4.48e+06 | 72.0% |
| raw_sample | 275 | 3.971 | 3.518 | 11.400 | 29.3 | 17.1% |
| good_turing | 269 | 5.834 | 4.060 | 30.055 | 8.39e+04 | 25.3% |
| sb_linear | 275 | 1.811 | 1.333 | 4.504 | 29.3 | 70.9% |
| ours_fitted | 275 | 1.721 | 1.214 | 4.446 | 29.3 | 72.7% |
| ours_oracle | 275 | 1.730 | 1.183 | 4.621 | 29.2 | 72.4% |

Using the uncapped truth on all cases, every spectrum-based method degrades
sharply because the estimator support stops at j = 256. This is a real
limitation of the common support, reported as limitation (x) in the main
text, not a defect of Q-error as a metric.

This is an offline cardinality readout. No plan quality, execution time, or
end-to-end optimizer measurement is claimed.
