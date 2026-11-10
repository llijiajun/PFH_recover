# Fixed-prior 2x2 readout factorial

276 real-column cases. All four arms share one sample, one fitted prior, one unseen-type count, and one scoring protocol.

| arm | observed types | unseen types | median W1 | median F1 rel err | median entropy rel err | median total types | median total mass |
|---|---|---|---|---|---|---|---|
| B1 | rounded | singleton | 0.04478 | 0.1599 | 0.0309 | 139.9 | 239.9 |
| B2 | rounded | profile | 0.03178 | 0.167 | 0.02677 | 139.9 | 241 |
| B3 | integrated | singleton | 0.02893 | 0.1248 | 0.03249 | 139.9 | 240.8 |
| B4 | integrated | profile | 0.02525 | 0.1622 | 0.02944 | 139.9 | 243.4 |

## Paired comparisons

- B4_vs_B1_w1: n=275, arm-a lower on 36.7%, median diff 0.0002313, Wilcoxon p=0.03587
- B2_vs_B1_w1: n=275, arm-a lower on 34.2%, median diff 0.0001245, Wilcoxon p=0.0001165
- B3_vs_B1_w1: n=275, arm-a lower on 49.5%, median diff 6.656e-05, Wilcoxon p=0.0001081
- B4_vs_B2_w1: n=275, arm-a lower on 44.7%, median diff 6.656e-05, Wilcoxon p=0.02664
- B4_vs_B3_w1: n=275, arm-a lower on 28.7%, median diff 0.000172, Wilcoxon p=6.715e-09
- B2_vs_B3_w1: n=275, arm-a lower on 34.5%, median diff 8.597e-05, Wilcoxon p=3.201e-10
- B4_vs_B1_f1_err: n=272, arm-a lower on 48.5%, median diff 7.951e-05, Wilcoxon p=0.8055
- B2_vs_B1_f1_err: n=272, arm-a lower on 44.5%, median diff 7.131e-05, Wilcoxon p=0.003659
- B3_vs_B1_f1_err: n=272, arm-a lower on 61.4%, median diff -0.0001111, Wilcoxon p=1.16e-09
- B4_vs_B2_f1_err: n=272, arm-a lower on 61.4%, median diff -0.0001111, Wilcoxon p=1.145e-10
- B4_vs_B3_f1_err: n=272, arm-a lower on 43.4%, median diff 8.283e-05, Wilcoxon p=0.001503
- B2_vs_B3_f1_err: n=272, arm-a lower on 39.0%, median diff 3.93e-05, Wilcoxon p=2.318e-06
- B4_vs_B1_entropy_err: n=275, arm-a lower on 54.2%, median diff -1.123e-05, Wilcoxon p=0.08158
- B2_vs_B1_entropy_err: n=275, arm-a lower on 57.1%, median diff -9.612e-06, Wilcoxon p=0.002874
- B3_vs_B1_entropy_err: n=275, arm-a lower on 48.7%, median diff 6.333e-06, Wilcoxon p=0.1288
- B4_vs_B2_entropy_err: n=275, arm-a lower on 49.8%, median diff 5.138e-06, Wilcoxon p=0.292
- B4_vs_B3_entropy_err: n=275, arm-a lower on 55.6%, median diff -5.581e-06, Wilcoxon p=0.01387
- B2_vs_B3_entropy_err: n=275, arm-a lower on 53.1%, median diff -1.751e-06, Wilcoxon p=0.06222

## Conservation

- Max within-case spread of total types across the four arms: 1.72065e-10.
- Total mass differs across arms because the profile allocation assigns unseen types to frequencies above one. W1 and TV are computed on renormalized spectra, so the shape comparison is unaffected.

## Cost (median wall-clock per case, ms)

- prior_fit: 1.914
- B1: 0.128
- B2: 0.041
- B3: 0.038
- B4: 0.037
