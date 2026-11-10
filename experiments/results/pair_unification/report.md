# Case-level unification of the readout paired statistics

Source: `experiments/results/readout_2x2/metrics.csv` with case key dataset, attribute, p. No new experiment was run.

## Case-count chain

- 276 planned cases
- 275 scored cases after the numerical guard
- 231 retained and 45 refused by the boundary flag
- 230 retained cases with a valid paired metric

The retained count of 231 includes the guard case. Only 230 retained cases produce a valid paired metric, so 231 of 276 is a flag-retention fraction and not a usable-output fraction.

## Cross-table identity

Maximum absolute difference B4 versus `eb_fit_w1` is 0. Maximum absolute difference B1 versus `smooth_bayes_w1` is 1.31e-11. Table III (Ours fitted, SB linear) and Table V (B4, B1) are the same two case-level columns to floating-point precision, so they must carry one paired statistic.

## Authoritative B4 versus B1 on W1

n = 275, win rate = 0.3673, two-sided Wilcoxon p = 0.035872, median paired difference = 2.313e-04.

Retired archive statistic: win rate 0.3491 and p = 0.008373 from `experiments/results/baseline_validation/w1_tests.json`. The archive stores summary statistics only. Its win rate and p-value cannot be reproduced from any archived case-level columns, and its medians match the two columns used here exactly. It is retired rather than kept as a second authoritative result.

## Per-metric paired statistics

| pair | metric | n | win rate | p two-sided |
|---|---|---|---|---|
| B4 vs B1 | W1 | 275 | 0.367 | 0.0359 |
| B4 vs B1 | F1 | 272 | 0.485 | 0.805 |
| B4 vs B1 | entropy | 275 | 0.542 | 0.0816 |
| B2 vs B1 | W1 | 275 | 0.342 | 0.000117 |
| B2 vs B1 | F1 | 272 | 0.445 | 0.00366 |
| B2 vs B1 | entropy | 275 | 0.571 | 0.00287 |
| B3 vs B1 | W1 | 275 | 0.495 | 0.000108 |
| B3 vs B1 | F1 | 272 | 0.614 | 1.16e-09 |
| B3 vs B1 | entropy | 275 | 0.487 | 0.129 |
