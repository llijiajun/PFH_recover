# Protocol reconciliation for the raw baseline (A4)

The two numbers come from different protocols and must not be compared as repeated measurements of one quantity.

## Protocol A, 2.51 +/- 0.04

- Where: tables/tab_main.tex and tables/tab_necessity.tex, raw or no-thickening row at alpha=1.5, p=0.1
- Definition: network ablation grid, corrected protocol with the schedule forced to p_T=1 and the exp-1 readout, three seeds over 24 evaluation populations, distance aggregated by that grid
- Populations: the network grid evaluation populations and its own seeds

## Protocol B, 0.64

- Where: sections/experiments.tex, mechanism paragraph
- Definition: independent analytic diagnostic grid from analyze_table2_mechanism.py, three seeds, N=1e5, vocab=3e4, j<=512, truth spectrum restricted to f<=J, W1 on the renormalized CDF
- Populations: seed 710000 + 13*seed + 1000*p + 100*alpha, regenerated inside that script, not the network grid populations

## Same-population, two-metric recomputation

| seed | d_obs | W1 analytic grid metric | W1 folded-truth metric | true types with f>J |
|---|---|---|---|---|
| 0 | 624 | 0.7114 | 0.7114 | 18 |
| 1 | 604 | 0.6372 | 0.6372 | 18 |
| 2 | 636 | 0.5724 | 0.5724 | 18 |

The two W1 implementations agree to floating-point precision on identical samples. The gap between 2.51 and 0.64 therefore comes from the populations and the evaluation grid, not from the distance implementation. Every raw W1 value needs a protocol label.

## Required actions

- Remove the claim that the controlled run uses the same populations as Table II.
- Label 0.64 explicitly as the analytic diagnostic grid.
- State that 2.51 and 0.64 are not comparable and use each only inside its own protocol.
- Correct the analyze_table2_mechanism.py docstring, which wrongly says the script measures mechanisms on the same populations as Table II.
