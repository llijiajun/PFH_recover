# FoF recovery against the full-population ground truth

## What the ground truth is

F_j is the number of distinct values whose frequency in the FULL
population equals j.  Every method in the table estimates that same
object, so all curves share one count axis and one metric space.

The FoF of the thinned sample counts distinct values whose SAMPLED
frequency equals j.  Its count axis is g rather than f and its support
is a fraction of the truth.  It is not an estimator of F_j and is
therefore excluded from both the figure and the table.

## Why the previous figure was wrong

The earlier figure showed only an Ours arm that used prior_from_truth.
That function builds the empirical prior from the full-population
frequencies, so the estimator received the true histogram through its
prior.  Its near-perfect overlap with the truth at retention rates down
to 0.01 was a self-consistency check of the posterior formula.  It was
not a recovery from the sample.  The ground truth was also a thin black
line drawn underneath the thick vermillion curve that overlapped it,
which made the truth effectively invisible.

This version fixes both defects.  The ground truth is the last artist
drawn, a thick black line with hollow circles above every estimate.
The deployed Ours arm uses fitted_prior_eb, which sees the thinned
sample only.  The oracle arm is kept as a clearly labelled ceiling in
the deviation figure, where a ratio axis makes the gap between an
information ceiling and a recovery visible.

## Setting

Retention rates are 0.01 to 0.05.  Only 5 to 19 percent of the true
distinct values leave any observation.  W1, TV and RMSE are computed on
normalized truncated spectra.  The NDV relative error uses absolute
counts and is signed as estimate minus truth divided by truth.

At these rates the Gamma-Poisson fit of fitted_prior_eb reaches its
lower bound on the shape parameter in every case.  The observed
spectrum carries almost no doubletons, so the prior shape is not
identified by the sample.  This is the dominant error source in the
deployment arm and the reason the recovered F_1 sits far below truth.
The paper evaluates real columns at retention rates of 0.1 and above.
These synthetic cases are deliberately sparser and sit outside that
regime, so the numbers here are a boundary study and not the pooled
accuracy reported in the paper.

| case | method | W1 (lower better) | TV | RMSE | NDV rel err |
|---|---|---|---|---|---|
| Zipf alpha 1.5, N 1.0M, p 0.01 | Naive 1/p | 184.945 | 0.9997 | 0.04050 | -94.8% |
| Zipf alpha 1.5, N 1.0M, p 0.01 | Good-Turing | 4.891 | 0.3258 | 0.01021 | -88.4% |
| Zipf alpha 1.5, N 1.0M, p 0.01 | Chebyshev | 56.879 | 0.8164 | 0.02694 | -84.4% |
| Zipf alpha 1.5, N 1.0M, p 0.01 | SB linear (fitted prior) | 294.898 | 0.9802 | 0.03964 | -94.7% |
| Zipf alpha 1.5, N 1.0M, p 0.01 | Ours (fitted prior) | 28.972 | 0.4421 | 0.01800 | -71.0% |
| Zipf alpha 1.5, N 1.0M, p 0.01 | Ours (oracle prior) | 3.548 | 0.0071 | 0.00036 | -3.1% |
| Zipf alpha 1.5, N 300k, p 0.02 | Naive 1/p | 113.555 | 0.9994 | 0.04093 | -91.4% |
| Zipf alpha 1.5, N 300k, p 0.02 | Good-Turing | 6.572 | 0.3180 | 0.00869 | -78.8% |
| Zipf alpha 1.5, N 300k, p 0.02 | Chebyshev | 33.650 | 0.7770 | 0.02711 | -79.4% |
| Zipf alpha 1.5, N 300k, p 0.02 | SB linear (fitted prior) | 117.193 | 0.8586 | 0.03380 | -90.0% |
| Zipf alpha 1.5, N 300k, p 0.02 | Ours (fitted prior) | 17.062 | 0.4072 | 0.01713 | -57.8% |
| Zipf alpha 1.5, N 300k, p 0.02 | Ours (oracle prior) | 3.235 | 0.0069 | 0.00035 | +3.3% |
| Zipf alpha 1.3, N 100k, p 0.02 | Naive 1/p | 90.727 | 0.9990 | 0.04391 | -92.9% |
| Zipf alpha 1.3, N 100k, p 0.02 | Good-Turing | 3.125 | 0.3264 | 0.01262 | -81.1% |
| Zipf alpha 1.3, N 100k, p 0.02 | Chebyshev | 354.498 | 0.9686 | 0.02866 | +304.9% |
| Zipf alpha 1.3, N 100k, p 0.02 | SB linear (fitted prior) | 47.494 | 0.5329 | 0.02044 | -86.7% |
| Zipf alpha 1.3, N 100k, p 0.02 | Ours (fitted prior) | 8.902 | 0.3871 | 0.01641 | -54.8% |
| Zipf alpha 1.3, N 100k, p 0.02 | Ours (oracle prior) | 1.374 | 0.0036 | 0.00016 | +4.0% |
| Zipf alpha 1.8, N 100k, p 0.05 | Naive 1/p | 86.447 | 1.0000 | 0.03592 | -81.5% |
| Zipf alpha 1.8, N 100k, p 0.05 | Good-Turing | 8.886 | 0.3816 | 0.01127 | -61.4% |
| Zipf alpha 1.8, N 100k, p 0.05 | Chebyshev | 39.498 | 0.7509 | 0.02539 | -28.5% |
| Zipf alpha 1.8, N 100k, p 0.05 | SB linear (fitted prior) | 89.850 | 0.9310 | 0.03272 | -80.2% |
| Zipf alpha 1.8, N 100k, p 0.05 | Ours (fitted prior) | 23.117 | 0.3725 | 0.01538 | -47.2% |
| Zipf alpha 1.8, N 100k, p 0.05 | Ours (oracle prior) | 8.510 | 0.0206 | 0.00089 | -4.0% |

## Reading the comparison figure

The thick black curve with hollow circles is the ground-truth FoF.
The thick vermillion curve is the posterior-mean FoF from the analytic
inverse with a prior fitted from the thinned sample.  It sits well
below the truth at small j and carries too much mass in the tail.
The oracle ceiling is not drawn in these panels because it lies on
top of the truth line and would be an invisible legend entry.  The
deviation figure shows fitted and oracle side by side on a ratio axis.

Good-Turing underestimates F_1 by a large factor and produces isolated
spikes at higher j.  Chebyshev returns a nearly flat curve.  Naive 1/p
moves observed counts to j about 1/p, which lies outside the displayed
range for p at or below 0.02.  SB linear with a fitted prior collapses
almost all mass into the first bin.

## Reading the deviation figure

Panel (a) divides each estimate by the truth at every j.  A value of
one is perfect recovery.  The fitted arm stays far below one at the
head of the spectrum while the oracle ceiling sits on the line.
Panel (b) sweeps the retention rate on one fixed population.  The
fitted arm degrades steadily as p falls while the oracle ceiling stays
near 0.01 TV, which is only possible because it reads the truth prior.

| p | observed types | TV fitted | TV oracle | TV Good-Turing |
|---|---|---|---|---|
| 0.002 | 1.6% | 0.5208 | 0.0059 | 0.3204 |
| 0.005 | 3.6% | 0.4695 | 0.0062 | 0.3311 |
| 0.010 | 5.1% | 0.4524 | 0.0069 | 0.3061 |
| 0.020 | 8.1% | 0.4102 | 0.0075 | 0.3194 |
| 0.050 | 15.6% | 0.3431 | 0.0070 | 0.3281 |
| 0.100 | 24.5% | 0.2913 | 0.0070 | 0.3277 |

## Case-level truth and recovery at the head of the spectrum

### Zipf alpha 1.5, N 1.0M, p 0.01

Observed distinct values 608 out of 11672 true ones (5.2%).

F_1 to F_6 truth  7011 1609 688 407 256 197

F_1 to F_6 fitted 673.7 334.8 221.7 165.1 131.2 108.6

F_1 to F_6 oracle 6742.7 1547.3 661.6 391.3 246.1 189.4

### Zipf alpha 1.5, N 300k, p 0.02

Observed distinct values 460 out of 5353 true ones (8.6%).

F_1 to F_6 truth  3285 714 300 150 123 105

F_1 to F_6 fitted 524.6 258.6 169.9 125.5 98.9 81.2

F_1 to F_6 oracle 3370.3 732.6 307.9 153.9 126.3 107.8

### Zipf alpha 1.3, N 100k, p 0.02

Observed distinct values 425 out of 6006 true ones (7.1%).

F_1 to F_6 truth  3768 877 331 174 142 93

F_1 to F_6 fitted 716.4 349.4 227.1 166.0 129.4 105.0

F_1 to F_6 oracle 3908.0 910.0 343.6 180.7 147.6 96.7

### Zipf alpha 1.8, N 100k, p 0.05

Observed distinct values 148 out of 799 true ones (18.5%).

F_1 to F_6 truth  459 91 45 24 24 13

F_1 to F_6 fitted 97.9 48.3 31.7 23.4 18.4 15.1

F_1 to F_6 oracle 432.9 85.7 42.3 22.5 22.5 12.2
