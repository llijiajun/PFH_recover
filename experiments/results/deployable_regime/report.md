# Boundary refusal and a sample-selectable deployment regime

## Selector

- **rule**: report the point histogram only when fitted a0 > 1e-3
- **computed_from**: the thinned sample alone, via fitted_prior_eb
- **threshold_origin**: the lower bound of the optimizer for a0, not a tuned constant and not chosen using ground truth or test error
- **flagged_case_semantics**: no point histogram is returned, a boundary flag is returned instead

## Usable output

- 231 of 276 cases usable, 83.7% of the benchmark.
- 45 cases flagged at the a0 bound.

## Median W1 on the two groups

| group | n | fitted | SB linear | Good--Turing | naive |
|---|---|---|---|---|---|
| kept | 231 | 0.009456 | 0.02138 | 0.03226 | 3.973 |
| refused | 45 | 11.26 | 9.942 | 14.71 | 38.99 |

## Large errors missed by the flag

- Definition: W1 > 0.1, fixed before inspecting the kept group and about ten times its fitted median.
- Kept 39, refused 45.
- Fraction of large errors not flagged: 46.4%.

## Baselines on the same kept set

- fitted_vs_smooth_bayes_w1: win 39.1%, cluster CI [0.28819463229895276, 0.4978165938864629], p=0.7642
- fitted_vs_good_turing_w1: win 44.2%, cluster CI [0.3303954924480806, 0.5517282042940794], p=0.05168
- fitted_vs_naive_w1: win 100.0%, cluster CI [1.0, 1.0], p=1.667e-39

## Held-out split by attribute

- split_0: n=115, fitted median W1 0.008376, win vs GT 41.6% (p=0.2936), win vs SB 33.9%
- split_1: n=116, fitted median W1 0.01061, win vs GT 46.8% (p=0.09177), win vs SB 44.3%

The selector uses no ground-truth quantity. The held-out split keeps every attribute entirely on one side, so the two halves are independent columns rather than repeated rates of the same column.

