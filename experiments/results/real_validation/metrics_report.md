# Real-data validation of Diffusion4Distribution

Closed-form analytic posterior (no training) tested on 4 real NDV datasets by simulating binomial thinning at p in [0.1, 0.2, 0.3, 0.5] with K=80 posterior draws.

## Overall (pooled over all attributes and p)

- cases: 392 (of 420 = 105 attributes x 4 retention levels; 28 excluded because those attributes have fewer than 2 distinct values and carry no learnable spectrum)
- mean NDV lost under thinning: 63.18%
- spectrum W1 improvement, method: +80.6% (vs naive 1/p: -1715.7%)
- singleton (F1) rel error: observed 58.3% -> method 4.4%
- observed-type total-count rel error: observed 24.6% -> method 1.1%
- value-level calibration: cov90=0.979 (target 0.90), cov95=0.989 (target 0.95)
- spectrum band coverage (90%): 0.992
- max calibration deviation: 0.080

## Per dataset

| dataset | cases | W1 impr (method) | W1 impr (naive) | F1 relerr (obs->meth) | count relerr (obs->meth) | cov95 | band90 |
|---|---|---|---|---|---|---|---|
| Airlines | 40 | +74.4% | -3228.7% | 12.1% -> 1.3% | 15.8% -> 1.6% | 0.986 | 0.989 |
| kasandr | 24 | +85.5% | -2150.8% | 127.6% -> 3.7% | 40.9% -> 0.5% | 0.974 | 1.000 |
| NCVoter | 264 | +80.3% | -1875.4% | 28.6% -> 3.0% | 19.3% -> 1.3% | 0.991 | 0.990 |
| SSB | 64 | +83.3% | +34.0% | 200.7% -> 13.2% | 45.2% -> 0.3% | 0.987 | 1.000 |

## What the numbers mean

- W1 improvement: how much the recovered spectrum closes the gap to the true spectrum relative to using the thinned observation directly. Positive means recovery helps.
- Naive 1/p per-type scaling (recorded in metrics.csv as W1_improve_naive) is a tempting baseline but is only fair on dense attributes. On singleton-heavy real columns it over-inflates every observed count and can make the spectrum worse than doing nothing, which is exactly why a principled posterior is needed.
- Calibration sits on or above the diagonal. On real data the credible intervals are conservative (they over-cover): the truth is reliably inside the 90 percent set, with some width cost. This is checked on real data because the ground truth f is known.
- The prior is built from the ground-truth frequencies (oracle, best case), which isolates the test to the inversion math. A deployed model would fit this prior from data. The learned Thickener and count-space endpoint are exercised on synthetic data in the companion notebook; training them on a real fitted prior is the deployment step.