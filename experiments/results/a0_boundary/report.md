# Boundary fit of the abundance-prior shape a0

Both sources use `eb_prior.fitted_prior_eb`, whose least-squares lower
bound on $a_0$ is 0.001. A case is a boundary fit when $a_0 \le$
0.001001.

## Real columns

- 45 of 276 cases (16.3%) are boundary fits, spanning 14 distinct attributes.
- Overall $a_0$ median 0.2500, range [0.001, 2].
- Airlines: 0/36 boundary fits.
- NCVoter: 37/220 boundary fits.
- kasandr: 8/20 boundary fits.
- Median $\Wone$ of the empirical-Bayes inverse is 9.2633 on boundary cases against 0.0095 on interior cases.

The boundary fit therefore also occurs on real columns. It is
associated with much larger spectrum error and marks a regime where
the two identifying equations do not determine $a_0$.

## Synthetic families (5 families x 4 rates x 3 seeds)

- concentrated: 0/12 boundary fits.
- geopoor: 2/12 boundary fits.
- mix: 0/12 boundary fits.
- spiked: 12/12 boundary fits.
- zipf: 12/12 boundary fits.
