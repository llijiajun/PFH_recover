# Joint-distribution recovery under the binomial-thinning channel

Controlled demonstration on synthetic V x V co-occurrence matrices (N=100000 tuples, 2D Zipf product-of-marginals), thinned entrywise at p, then inverted with the paper's closed-form per-cell posterior and a Gamma-Poisson prior fitted from the observed joint only.  This recovers the joint count matrix and its join-cardinality functional sum f^2.  It is a controlled demonstration with known ground truth, NOT a live query-optimizer evaluation.


## Join-cardinality Q-error (geometric mean over populations x seeds, lower better)

| regime | alpha | V | p | ours | raw 1/p | Good-Turing |
|---|---|---|---|---|---|---|
| heavy | 1.3 | 200 | 0.1 | 49.563 | 50.564 | 1.983 |
| heavy | 1.3 | 200 | 0.3 | 51.366 | 51.563 | 5.112 |
| sparse | 0.4 | 500 | 0.1 | 5.924 | 6.109 | 1.270 |
| sparse | 0.4 | 500 | 0.3 | 2.158 | 2.320 | 1.260 |

## Joint FOF-spectrum W1 (median, lower better)

| regime | alpha | V | p | ours | raw 1/p | Good-Turing |
|---|---|---|---|---|---|---|
| heavy | 1.3 | 200 | 0.1 | 17.863 | 2.627 | 5.403 |
| heavy | 1.3 | 200 | 0.3 | 6.458 | 1.627 | 17.570 |
| sparse | 0.4 | 500 | 0.1 | 9.036 | 0.295 | 0.331 |
| sparse | 0.4 | 500 | 0.3 | 2.357 | 0.226 | 0.333 |

## Reading
The binomial-thinning kernel is binomial on every cell, so the same closed-form per-cell posterior that recovers a single frequency histogram inverts a joint co-occurrence matrix entry by entry. We verify this mechanically: the recovered joint runs end to end and its join-cardinality functional sum f^2 is computable from the same formula. The empirical result is a prior-transfer limit, not a win, and we report it plainly.  On join cardinality (the functional an equi-join optimizer needs) our recovery ties the naive raw 1/p scaling and is dominated by Good-Turing in every regime (Q-error 1.98-5.11 for Good-Turing against 49-51 on the heavy join and 1.3-2.3 on the sparse join for our method): sum f^2 is driven by the unseen and singleton-cell mass, which Good-Turing estimates directly and the Gamma-Poisson prior fitted from a single column does not.  On the joint FOF spectrum (W1) the prior-inflated recovery is farther from the truth than naive 1/p in every regime (e.g. 17.9 vs 2.6 at p=0.1 on the heavy join), because the joint's cell-count distribution is more singleton-heavy than the single-column marginal the prior was tuned for, so the posterior over-corrects rare cells.  This is the same singleton over-correction the single-column study already documents, now exposed on the joint because no heavy tail dominates sum f^2 there.  Conclusion: the framework covers joint recovery in principle (entrywise kernel, identical formula), but realizing the optimizer payoff requires a prior fit to the joint's own cell-count distribution; a joint-aware prior is future work.  A live DBMS join-cardinality evaluation remains future work.
