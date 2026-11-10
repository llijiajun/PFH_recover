# Entropy consistency check (A3)

Source `experiments/results/modern_baselines/modern_metrics.csv`, 276 rows.

## Median relative entropy error, deployment state

| table row | median % | n |
|---|---|---|
| naive_1/p | 24.2 | 275 |
| Good--Turing | 29.1 | 269 |
| NPMLE | 6.7 | 275 |
| SB linear | 3.1 | 275 |
| Ours fitted | 2.9 | 275 |
| Ours oracle | 2.9 | 275 |

## Paired tests

| comparison | n | a lower | median diff | p |
|---|---|---|---|---|
| fitted_vs_sb | 275 | 0.542 | -1.12e-05 | 0.08145 |
| oracle_vs_sb | 275 | 0.611 | -5.28e-05 | 1.087e-06 |
| fitted_vs_naive | 275 | 0.924 | -0.189 | 4.118e-42 |
| oracle_vs_naive | 275 | 0.953 | -0.19 | 4.261e-45 |
| sb_vs_naive | 275 | 0.924 | -0.186 | 1.147e-42 |

## Correction

- **table**: SB linear is 3.1% and Ours fitted is 2.9%, so the table is correct.
- **prose**: The prose claimed both reach 2.9%. It must say 2.9% against 3.1%.
- **p_value**: The archived p=4.4e-45 magnitude belongs to a fitted/oracle estimator against the naive plug-in, not to fitted against SB. The fitted-versus-SB two-sided p is reported above.
- **interpretation**: A p near 1e-45 is incompatible with 'statistically indistinguishable'. The fitted-versus-SB comparison is reported with its own p value and direction.
