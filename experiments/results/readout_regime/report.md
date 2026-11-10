# Where the 2x2 readout gain holds

The integrated readout lowers the pooled median W1 on interior fits but is not a per-case win. On boundary fits it is significantly worse than the rounded readout. The sample-only a0 flag therefore identifies both the usable regime and the regime where the added readout should not be used.

| group | n | median B4 | median B1 | B4 lower | median paired diff | p |
|---|---|---|---|---|---|---|
| all | 275 | 0.02525 | 0.04478 | 36.7% | +0.00023126 | 0.03587 |
| interior | 230 | 0.009456 | 0.02138 | 39.1% | +9.984e-05 | 0.7642 |
| boundary | 45 | 11.26 | 9.942 | 24.4% | +0.32039 | 2.755e-07 |

## Interior fits by rate

| p | n | median B4 | median B1 | B4 lower | p |
|---|---|---|---|---|---|
| 0.1 | 54 | 0.0191 | 0.0131 | 37.0% | 0.001032 |
| 0.2 | 60 | 0.02113 | 0.03358 | 36.7% | 0.1185 |
| 0.3 | 58 | 0.007351 | 0.02929 | 43.1% | 0.4133 |
| 0.5 | 58 | 0.005263 | 0.01798 | 39.7% | 0.2069 |

## Kept-set paired comparisons with uncertainty

- fitted vs good_turing_w1: n=224, win 44.2%, cluster CI [0.33, 0.552], median diff +9.984e-05, p=0.05168
- fitted vs smooth_bayes_w1: n=230, win 39.1%, cluster CI [0.288, 0.498], median diff +9.984e-05, p=0.7642
