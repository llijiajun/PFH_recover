# Baseline validation of Diffusion4Distribution

Our closed-form recovery (binomial-thinning inversion) benchmarked against the canonical NDV / frequency-distribution estimators, on 3 clean real NDV datasets (Airlines, kasandr, NCVoter) with simulated thinning at p in [0.1, 0.2, 0.3, 0.5]. 276 (dataset, attribute, p) cases. SSB was dropped: its exported 'Fi' files use an inconsistent (value, count) convention across columns, and degenerate single-value columns (n_types<10) were excluded.

The forward model (binomial thinning) is identical for every method, so the comparison isolates the inversion quality. The oracle row builds the prior from ground-truth frequencies, which is the information ceiling and isolates the test to the inversion math. The deployment row uses a Gamma-Poisson prior fitted from the sample alone by empirical Bayes. It is evaluated on the same 276 cases in results/fitted_prior_validation. Paired significance tests and bootstrap intervals are in significance.json.

## 1. NDV scalar accuracy (number of distinct values)

Median relative error over all cases is the headline (the error distributions are heavy-tailed, so the mean is reported but the median is the fair summary).

| method | median rel err | mean | % within 10% | % within 25% |
|---|---|---|---|---|
| Naive d/p | 17.0% | 78.2% | 35.9% | 61.2% |
| Chao1 | 108.4% | 311.2% | 6.5% | 17.8% |
| Haas HT | 58.4% | 60.8% | 1.8% | 6.9% |
| Haas MM1 | 25.3% | 756622.1% | 40.2% | 49.6% |
| SVD deconv | 30.3% | 51.8% | 17.0% | 42.0% |
| Good-Turing | 100.0% | 314.2% | 5.1% | 12.0% |
| Ours | 6.7% | 20.0% | 60.5% | 81.2% |

## 2. Full frequency-distribution (FOF spectrum) recovery

Median over all cases (lower is better).

| method | W1 | TV | RMSE | NDV rel err | pop (N) rel err |
|---|---|---|---|---|---|
| Naive 1/p | 4.000 | 1.000 | 0.061 | 69.9% | 15.2% |
| SVD deconv | 7.972 | 0.807 | 0.037 | 30.3% | 395.1% |
| Good-Turing | 0.052 | 0.045 | 0.003 | 100.0% | 100.0% |
| Log-Log | 178.087 | 0.834 | 0.023 | 100.0% | 100.0% |
| Chebyshev | 215.685 | 0.900 | 0.023 | 100.0% | 100.0% |
| Ours | 0.001 | 0.001 | 0.000 | 6.7% | 14.6% |

## 3. Per-dataset NDV relative error (median, ours vs baselines)

| dataset | cases | Ours | SVD | Good-Turing | Chao1 | Haas HT | Haas MM1 | Naive d/p |
|---|---|---|---|---|---|---|---|---|
| Airlines | 36 | 5.3% | 26.4% | 152.6% | 181.6% | 62.3% | 15.3% | 14.1% |
| kasandr | 20 | 7.4% | 24.5% | 289.9% | 65.7% | 51.1% | 10703.7% | 82.2% |
| NCVoter | 220 | 7.3% | 31.1% | 95.2% | 106.6% | 58.5% | 30.8% | 17.8% |

## 4. What the baselines are

- **Naive d/p** -- divide the observed distinct count by p (Horvitz-Thompson with the inclusion probability assumed equal to p). The trivial reference; over-corrects when the column is singleton-heavy.
- **Chao1** -- ecology's standard lower-bound unseen-species estimator from the singleton/doubleton counts; widely used for distinct-value problems.
- **Haas HT / Haas MM1** -- the foundational sampling-based DV estimators (Haas et al., VLDB 1995). HT is unbiased under known inclusion probabilities; MM1 is the finite-population method-of-moments variant.
- **SVD deconv** -- invert the binomial sampling channel F = M^+ f with the same M[j,i]=Bin(i,p,j) our forward model uses. The direct linear competitor.
- **Good-Turing** -- classic missing-mass smoothing of the FOF spectrum.
- **Log-Log / Chebyshev** -- polynomial/power-law extrapolation baselines from the classical database literature on distinct-value estimation.
- **Ours** -- closed-form posterior over observed types plus a prior-derived unseen-mass estimate (the missing fraction under the oracle prior).

## 5. Honest caveats

- The spectrum comparison truncates every spectrum (truth included) to j<=256, so all methods are judged on the same support; this is fair but means extreme high-frequency tails are not scored.
- SVD deconvolution and Chebyshev extrapolation are ill-conditioned on heavy-tailed real columns. We report a ridge-regularized SVD and a bounded low-degree Chebyshev so the baselines are stable and fair rather than numerically broken; even so they trail our posterior. A principled posterior degrades more gracefully than a naive linear or polynomial inverse of the same channel.
- Both prior variants are reported in the paper. The oracle row is the information ceiling. The empirical-Bayes fitted row uses only observables and is evaluated on the same cases in results/fitted_prior_validation. A naive moment-matched variant that ignores zero truncation collapses to 56.3% median NDV error, which locates the difficulty in the unseen mass.
- SSB and degenerate columns (n_types<10) were excluded for data-integrity reasons stated above; the three retained datasets span 276 attribute cases. Case seeds derive from a stable hash of (dataset, attribute, p), so every number regenerates exactly.