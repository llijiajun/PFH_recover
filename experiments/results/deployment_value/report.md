# Deployment value of the full method (new R1 evidence)

All numbers are recomputed from the archived per-case CSV. No model is retrained.

## Rate cells, fitted full-posterior readout vs linear smoothed Bayes

| p | n | median fitted | median SB | ratio | cluster CI95 | wins | median paired diff | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 68 | 0.04802 | 0.04674 | 1.027 | [0.740, 2.029] | 21/68 | +0.00059 | 2.976e-06 |
| 0.2 | 69 | 0.03200 | 0.05071 | 0.631 | [0.358, 1.303] | 23/69 | +0.00026 | 0.007427 |
| 0.3 | 69 | 0.02140 | 0.04655 | 0.460 | [0.188, 0.717] | 26/69 | +0.00023 | 0.2695 |
| 0.5 | 69 | 0.01264 | 0.03787 | 0.334 | [0.155, 0.632] | 31/69 | +0.00010 | 0.02606 |

## Selectable regime p >= 0.2

- n = 207, median 0.02140 vs 0.04478, ratio 0.478, cluster CI95 [0.234, 0.741]
- wins 80, ties 0, losses 127 (win rate 0.386), median paired diff +0.00010, Wilcoxon p 0.65
- This is an aggregate gain, not per-case dominance.

## Held-out attribute halves at p >= 0.2

- half A: n=90, 30 columns, ratio 0.531, CI95 [0.230, 0.985], win rate 0.411
- half B: n=117, 39 columns, ratio 0.471, CI95 [0.149, 0.911], win rate 0.368

## By dataset at p >= 0.2

- Airlines: n=27, ratio 0.957, win rate 0.556
- NCVoter: n=165, ratio 0.337, win rate 0.358
- kasandr: n=15, ratio 1.034, win rate 0.400

## Leave-one-dataset-out at p >= 0.2

- drop none: n=207, ratio 0.478, win rate 0.386
- drop Airlines: n=180, ratio 0.405, win rate 0.361
- drop NCVoter: n=42, ratio 0.532, win rate 0.500
- drop kasandr: n=192, ratio 0.481, win rate 0.385

- p >= 0.3 only: n=138, ratio 0.362, CI95 [0.20038621884952204, 0.6424956083104679], win rate 0.413

Caveat: The aggregate gain is strongest on NCVoter, which supplies 165 of the 207 cases in the selectable regime.  Viewed separately, Airlines and kasandr have ratios near one.  The leave-one-dataset-out ratios stay below one, but a per-dataset gain is not established.

## Complete strategies on the same scored cases

| strategy | n | median | p75 | p90 | mean | fraction W1>0.10 |
|---|---|---|---|---|---|---|
| fitted_full_posterior | 275 | 0.02525 | 0.56750 | 15.94956 | 11.3601 | 0.305 |
| sb_linear | 275 | 0.04478 | 0.39455 | 15.78606 | 11.2956 | 0.364 |
| good_turing | 269 | 0.05189 | 1.37053 | 15.61111 | 7.3054 | 0.387 |
| npmle | 275 | 0.64507 | 2.48821 | 24.21014 | 11.9113 | 0.676 |
| naive_1_over_p | 275 | 4.00000 | 9.02891 | 48.54701 | 20.2905 | 1.000 |
| oracle_prior_ceiling | 275 | 0.00133 | 0.06538 | 15.76012 | 10.4583 | 0.247 |

## Readout ablation on the same 276 cases

- B1 median W1 0.04478, B4 0.02525, ratio 0.564
