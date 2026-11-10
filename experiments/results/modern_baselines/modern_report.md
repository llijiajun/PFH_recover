# Modern baselines and downstream utility audit

276 (dataset, attribute, p) cases, identical binomial-thinning channel and stable seeds as the canonical baseline run.

## 1. NDV scalar, median relative error (lower is better)

| method | median | mean | within 10% | finite cases |
|---|---|---|---|---|
| sgt | 26.5% | 39.3% | 26.5% | 275/276 |
| jack1 | 46.2% | 49.1% | 18.5% | 276/276 |
| jack2 | 42.5% | 47.1% | 10.9% | 276/276 |
| coverage | 85.4% | 278.3% | 0.7% | 141/276 |
| poly | 54.4% | 101.9% | 9.3% | 140/276 |
| npmle | 24.5% | 35.3% | 32.2% | 276/276 |
| smooth_bayes | 9.8% | 22.4% | 50.4% | 276/276 |

## 2. Full-spectrum recovery, median W1 (lower is better)

| method | W1 | TV | RMSE | NDV rel err |
|---|---|---|---|---|
| naive | 4 | 1 | 0.06057 | 69.9% |
| svd | 7.972 | 0.8072 | 0.03658 | 30.3% |
| good_turing | 0.05189 | 0.04481 | 0.002608 | 100.0% |
| ours | 0.001335 | 0.0008992 | 4.606e-05 | 6.7% |
| npmle | 0.6451 | 0.5104 | 0.01824 | 24.5% |
| smooth_bayes | 0.04478 | 0.03584 | 0.002192 | 9.8% |
| eb_fit | 0.02525 | 0.01736 | 0.0009553 | 9.8% |

## 3. Downstream functional error (median relative error)

Functionals that a scalar NDV estimate cannot produce. self_join is SUM f_v^2; rare_frac is the share of distinct values with frequency <= 5; entropy is the Shannon entropy of the value distribution.

| method | NDV | self-join | rare fraction | entropy |
|---|---|---|---|---|
| naive | 69.9% | 248.2% | 2.7% | 24.2% |
| svd | 30.3% | 2179.3% | 42.0% | 8.9% |
| good_turing | 100.0% | 99.7% | 0.0% | 29.1% |
| ours | 6.7% | 15.3% | 0.0% | 2.9% |
| npmle | 24.5% | 71.3% | 0.0% | 6.7% |
| smooth_bayes | 9.8% | 25.5% | 0.0% | 3.1% |
| eb_fit | 9.8% | 19.7% | 0.0% | 2.9% |

## 4. What this audit tests

- `npmle` fits the abundance prior by EM on the zero-truncated FoF profile (PML/NPMLE, Acharya et al. ICML 2017; Wang & Lindsay 2005/2007) and then runs the paper's OWN closed-form posterior with that prior. If it matches or beats `ours`, the diffusion vocabulary is an analogy layered on top of empirical Bayes deconvolution.
- `smooth_bayes` is the smoothed-Bayesian linear estimator (Beraha, Favaro, Sesia; JASA 2025). Under a Gamma-Poisson abundance prior the reverse posterior mean is linear in the observed count, E[f|g] = g + (1-p)(a0+g)/(b0+p). This is exactly the 'reverse diffusion step' written as a shrinkage estimator.
- `sgt` is smoothed Good-Toulmin (Orlitsky, Suresh, Wu; PNAS 2016), near-minimax for species extrapolation up to t ~ log n.
- `poly` is a ridge-fitted polynomial linear estimator in the spirit of Wu & Yang (Annals of Statistics 2019) and the SIGMOD 2024 polynomial property-estimation framework, which evaluates on these same datasets.
- Learned fusion methods (AdaNDV VLDB 2025, PLM4NDV SIGMOD 2025, learned NDV PVLDB 2021) need external training corpora or schema, so they are discussed in related work but not run here. The comparison is restricted to single-sample, no-training estimators.

## 5. Caveats

- The NPMLE support is truncated to f <= 2048 for tractability; frequencies above H_CAP = 512 are folded into the last bin for every method, so this does not change the reported spectrum metric.
- Smoothed Good-Toulmin assumes iid sampling with replacement from a fixed underlying distribution, whereas the data use finite-population binomial thinning. We set t = (1-p)/p, which is the natural adaptation, and report the resulting numbers without hiding the mismatch.
- Every summary statistic is computed over the cases where that method returns a finite value. Good coverage and Poly fail on roughly half the cases, and SGT fails once, so their rows are optimistic relative to the Ours rows, which return a finite estimate on all cases. The finite-case count is reported in the first table.
- All spectra (truth included) are capped at j <= 512.
- Paired tests drop rows where either arm is non-finite. The W1 comparison between the full posterior and SB linear uses 275 pairs.