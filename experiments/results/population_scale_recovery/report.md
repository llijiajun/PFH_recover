# Population-scale recovery (new R2 evidence)

Both quantities live on the j<=512 truncated support. True row mass median 268.0, true type count median 124.0.

| arm | type count abs relerr median | row mass abs relerr median | row mass signed relerr median |
|---|---|---|---|
| B1 | 0.0979 | 0.1668 | -0.0127 |
| B4 | 0.0979 | 0.1506 | +0.0231 |

## Readout improvement on truncated row mass

- median abs relerr 0.1668 (B1) to 0.1506 (B4), ratio 0.903, cluster CI95 [0.6415393851926068, 0.9987523884069247]
- B4 better on 144 cases, ties 1, worse on 131, Wilcoxon p 0.3472
- The full-posterior readout improves the truncated row mass but leaves the truncated type count unchanged, because the type count is set by the unseen-type compensation.

## By retention rate

| p | n | true types | true mass | B1 mass | B4 mass | B1 abs relerr | B4 abs relerr |
|---|---|---|---|---|---|---|---|
| 0.1 | 69 | 124.0 | 268.0 | 229.9 | 230.0 | 0.2125 | 0.2000 |
| 0.2 | 69 | 124.0 | 268.0 | 249.9 | 250.0 | 0.1668 | 0.1476 |
| 0.3 | 69 | 124.0 | 268.0 | 226.8 | 236.7 | 0.1567 | 0.1495 |
| 0.5 | 69 | 124.0 | 268.0 | 268.1 | 276.0 | 0.1163 | 0.0824 |

## Column cross-check

- B1_total_mass_vs_smooth_bayes_pop_rel: match=True max diff 1.9e-12
- B4_total_mass_vs_eb_fit_pop_rel: match=True max diff 7.1e-15
- total_types_vs_eb_fit_ndv_rel: match=True max diff 3.5e-12
- total_types_vs_smooth_bayes_ndv_rel: match=True max diff 1.9e-12

These two quantities are not validated by W1, TV, or any per-frequency band coverage result, and they are not estimates of the full population size.
