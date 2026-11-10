# Downstream fair comparison (ICDE revision)

Cases: 276 full, 208 truncated-support subset, 68 excluded.

## Estimator definitions
- `raw_unscaled`: sum_v g_v^2, bounded above by the truth
- `rescaled_plugin`: sum_v (g_v/p)^2 = sum_v g_v^2 / p^2
- `factorial_unbiased`: sum_v [g_v(g_v-1)/p^2 + g_v/p]
- `naive_spectrum_readout`: self-join read off the naive 1/p recovered spectrum, a different estimator

## full_276 (relative error, median / p25 / p75 / p90)

| estimator | n | p25 | median | p75 | p90 |
| --- | --- | --- | --- | --- | --- |
| raw_unscaled | 276 | 0.7099 | 0.8116 | 0.9101 | 0.9602 |
| rescaled_plugin | 276 | 0.2773 | 1.6848 | 3.6388 | 7.4241 |
| factorial_unbiased | 276 | 0.0157 | 0.0768 | 0.2093 | 0.4161 |
| sb_linear | 276 | 0.1288 | 0.2699 | 0.6912 | 2.9547 |
| ours_fitted | 276 | 0.0698 | 0.2006 | 0.6908 | 3.0078 |
| ours_oracle | 276 | 0.0412 | 0.1667 | 0.8206 | 3.5695 |
| good_turing | 276 | 0.6601 | 0.9987 | 3.3807 | 8.5148 |
| npmle | 276 | 0.1510 | 0.8406 | 2.6451 | 4.6871 |
| naive_spectrum_readout | 276 | 1.2810 | 2.5032 | 4.9204 | 10.3333 |

## subset_208 (relative error, median / p25 / p75 / p90)

| estimator | n | p25 | median | p75 | p90 |
| --- | --- | --- | --- | --- | --- |
| raw_unscaled | 208 | 0.6380 | 0.7770 | 0.8710 | 0.9201 |
| rescaled_plugin | 208 | 1.1600 | 2.4910 | 4.5180 | 8.7742 |
| factorial_unbiased | 208 | 0.0434 | 0.1208 | 0.2480 | 0.5020 |
| sb_linear | 208 | 0.0914 | 0.1952 | 0.3331 | 0.5478 |
| ours_fitted | 208 | 0.0453 | 0.1248 | 0.2537 | 0.5417 |
| ours_oracle | 208 | 0.0310 | 0.0909 | 0.2069 | 0.4800 |
| good_turing | 208 | 0.5947 | 1.0391 | 3.4389 | 8.2490 |
| npmle | 208 | 0.0842 | 0.4510 | 1.2587 | 3.4295 |
| naive_spectrum_readout | 208 | 1.0824 | 2.1485 | 4.2333 | 8.7742 |

## Three-case recomputation

- Airlines.CRSArrTime.csv p=0.1: truth=1874, raw=128, rescaled=12800, factorial=1460; relerr raw=0.932, rescaled=5.830, factorial=0.221, ours=0.218, naive spectrum=5.830
  - rescaled identity check |S2_rescaled - S2_raw/p^2| = 1.819e-12
- NCVoter.mail_city.csv p=0.3: truth=11399428, raw=1127777, rescaled=12530856, factorial=12514499; relerr raw=0.901, rescaled=0.099, factorial=0.098, ours=1.389, naive spectrum=3.112
  - rescaled identity check |S2_rescaled - S2_raw/p^2| = 3.725e-09
- kasandr.userid.csv p=0.5: truth=20255699233, raw=5048381941, rescaled=20193527764, factorial=20193236298; relerr raw=0.751, rescaled=0.003, factorial=0.003, ours=2.535, naive spectrum=2.646
  - rescaled identity check |S2_rescaled - S2_raw/p^2| = 0.000e+00

## Unbiasedness check: f=7, p=0.2: E[estimator]=49.000000 vs f^2=49, diff=7.105e-15

## Win rates of ours (fitted) with column-cluster bootstrap 95% CI

| competitor | ours win rate | CI low | CI high | Wilcoxon p (two-sided) |
| --- | --- | --- | --- | --- |
| raw_unscaled | 0.761 | 0.659 | 0.855 | 5.538e-05 |
| rescaled_plugin | 0.725 | 0.623 | 0.822 | 7.165e-11 |
| factorial_unbiased | 0.377 | 0.304 | 0.449 | 3.631e-13 |
| sb_linear | 0.525 | 0.442 | 0.609 | 0.1973 |
| ours_oracle | 0.522 | 0.449 | 0.598 | 0.1481 |
| good_turing | 0.728 | 0.641 | 0.812 | 1.067e-13 |
| npmle | 0.746 | 0.692 | 0.797 | 4.448e-26 |

## Spectrum W1 win rates with column-cluster bootstrap 95% CI

| competitor | ours fitted W1 win rate | CI low | CI high | Wilcoxon p |
| --- | --- | --- | --- | --- |
| sb_linear | 0.349 | 0.258 | 0.442 | 0.008373 |
| good_turing | 0.454 | 0.344 | 0.561 | 0.08611 |
| naive_rescaled | 1.000 | 1.000 | 1.000 | 7.481e-47 |

## Per-dataset medians and win rates

- Airlines (n=36): factorial median 0.1064, ours 0.1666, SB 0.2802; ours wins vs factorial 0.444 [0.278,0.583], vs SB 0.722 [0.556,0.889]
- NCVoter (n=220): factorial median 0.0744, ours 0.2041, SB 0.2527; ours wins vs factorial 0.377 [0.295,0.455], vs SB 0.505 [0.409,0.595]
- kasandr (n=20): factorial median 0.0037, ours 2.4646, SB 2.4480; ours wins vs factorial 0.250 [0.000,0.550], vs SB 0.400 [0.100,0.700]

## Factor control

{
 "shared_prior_parameters_agreement": {
  "max_abs_diff": 2.0667282538966436e-06,
  "n_pairs": 276
 },
 "published_spectrum_rows": {
  "SB linear": {
   "prior_family": "Gamma-Poisson",
   "prior_params": "fitted by smoothed-Bayes moment match (a0_sb, b0_sb)",
   "unseen_compensation": "none beyond the fitted prior",
   "truncation": "j<=512 renormalized",
   "readout": "affine posterior mean, re-bucketed",
   "w1_median": {
    "finite_cases": 275,
    "mean": 6.252632244848654,
    "median": 0.044778634681253204
   }
  },
  "Ours (fitted)": {
   "prior_family": "Gamma-Poisson",
   "prior_params": "fitted by zero-truncation-aware empirical Bayes",
   "unseen_compensation": "V0 = D w/(1-w) zero-truncated term",
   "truncation": "j<=512 renormalized",
   "readout": "posterior-mean spectrum",
   "w1_median": {
    "finite_cases": 275,
    "mean": 6.3248791734887035,
    "median": 0.025250310672515486
   }
  }
 },
 "interpretation": "Same prior family, different fitted parameters and different unseen-type compensation. Feeding SB linear the SAME fitted parameters reproduces our spectrum to the archived max_abs_diff, so the published median gap is a prior-fit and unseen-compensation gap, not an inversion gap."
}
