# Shared-prior readout ablation and boundary-fit diagnostic

## Paired comparison of the two readouts

- Source `experiments/results/baseline_validation/w1_tests.json`, 275 scored cases.
- Full-posterior readout wins on 34.9% of cases, Wilcoxon p = 0.008373.
- Median W1 0.02525 (fitted) against 0.04478 (rounded).
- Both rows use one fitted prior and one truncation, so the test isolates the readout. The lower median does not become a per-case win.

## Boundary-fit diagnostic

- 45 of 276 cases have $a_0 \le$ 0.001.
- Median W1 of the empirical-Bayes inverse is 9.2633 on boundary cases against 0.0095 on interior cases.
- Median W1 on boundary cases by estimator:
  - eb_fit_w1: 11.2554
  - smooth_bayes_w1: 9.9424
  - good_turing_w1: 14.7079
  - naive_w1: 38.9925
- Per-case lowest-W1 counts on boundary cases: eb_fit_w1=9, good_turing_w1=20, smooth_bayes_w1=16

The boundary group has no accurate estimator in this benchmark. Good--Turing attains the lowest W1 on the largest number of boundary cases but has a worse median than the fitted inverse, so a blanket Good--Turing fallback is not supported by these data. The supported action is to flag the fit and withhold the point histogram.
