# Oracle vs fitted prior for the closed-form inverse

Same 276 (dataset, attribute, p) cases and seeds as baseline_validation. Two fitted variants, both observable-only:
- **fit** = MoM Gamma on de-thinned observed counts + NB predictive (ignores zero truncation; a lower reference).
- **eb** = empirical-Bayes NB matched to the observed singleton/doubleton mass fractions (C-free identifying equations; the principled deployment prior).


## Overall (median rel err / median metric)

| prior | NDV rel | w10 | w25 | W1 | TV | RMSE |
|---|---|---|---|---|---|---|
| oracle | 6.7% | 59.4% | 81.2% | 0.001 | 0.001 | 0.000 |
| fit | 56.3% | 0.7% | 5.8% | 3.315 | 0.735 | 0.048 |
| eb | 9.7% | 51.1% | 77.2% | 0.025 | 0.017 | 0.001 |

Paired: the EB fitted prior beats the oracle ceiling on NDV rel err in 38.8% of 276 cases.


## Per dataset (NDV rel err median)

| dataset | cases | oracle | MoM fit | EB fit |
|---|---|---|---|---|
| Airlines | 36 | 5.3% | 61.3% | 6.2% |
| kasandr | 20 | 8.5% | 52.4% | 10.5% |
| NCVoter | 220 | 7.4% | 56.6% | 10.3% |