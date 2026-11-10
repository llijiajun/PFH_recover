# Review-response analysis, archived results only

## 1. Why the prior-free arm is more accurate in Table 2

Fixed rate parameter b0 = 0.017016. Design bias of the affine posterior mean for a
fixed true count f is (1-p)(a0 - b0 f)/(b0+p), so the prior mean count a0/b0
sets the systematic bias against the true mean abundance.

Fitted diagnostics give a prior mean count for a never-observed type of 8.548
against a true value of 0.121, a ratio of 70.7.
The singleton class is corrected by 0.839x where roughly 4x is
needed, so the error is a systematic under-correction rather than sampling noise.

| a0 | a0/b0 | W1 mean | W1 sd | F1 rel err | F2 rel err |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 14.7 | 3.914 | 0.137 | 0.285 | 0.648 |
| 0.5 | 29.4 | 5.041 | 0.123 | 0.522 | 0.529 |
| 1 | 58.8 | 7.913 | 0.141 | 0.791 | 0.587 |
| 2 | 117.5 | 12.420 | 0.234 | 0.959 | 0.839 |
| 4 | 235.1 | 17.542 | 0.403 | 0.998 | 0.984 |
| 8 | 470.1 | 21.655 | 0.575 | 1.000 | 1.000 |

W1 rises 5.53x from a0=0.25 to a0=8,
and the across-population W1 sd rises 4.19x.
The oracle selects a0=0.25 in 72 of 72 populations, the lower end of the grid.
A stronger prior therefore adds bias without buying variance reduction. The plain arm
uses Monte-Carlo targets and no statistical prior, so it is not exposed to this bias.
The shared-prior ablation shows the inversion family itself is not the cause, since
feeding the smoothed-Bayesian estimator the same parameters reproduces our spectrum.

## 2. Why Good-Turing wins on the joint distribution

| regime | p | tuples per occupied cell | tuples per marginal value | scale ratio | unseen cells | Q ours | Q GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heavy a1.3 V200 | 0.1 | 11.38 | 500 | 43.9 | 0.727 | 49.62 | 2.16 |
| heavy a1.3 V200 | 0.3 | 11.39 | 500 | 43.9 | 0.475 | 51.41 | 5.44 |
| sparse a0.4 V500 | 0.1 | 1.33 | 200 | 150.2 | 0.872 | 5.92 | 1.27 |
| sparse a0.4 V500 | 0.3 | 1.33 | 200 | 150.0 | 0.639 | 2.16 | 1.26 |

The entrywise binomial inverse is correct. What fails is the abundance prior, which is
fitted to one column and then applied to a joint whose occupied cells are one to two
orders of magnitude sparser. Good-Turing estimates the unseen mass directly from the
observed joint frequency profile and needs no factorized prior.
