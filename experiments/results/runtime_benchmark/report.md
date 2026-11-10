# Runtime and memory of the Table V estimators

Rate $p=0.3$, median of 4 runs, peak from `tracemalloc`.

## NCVoter/county_commiss_desc.csv (d_obs=44, n=44)

| Method | ms | MB |
|---|---:|---:|
| Naive $1/p$ | 0.0 | 0.0 |
| Good--Turing | 0.0 | 0.0 |
| SVD deconvolution | 186.7 | 4.0 |
| Chebyshev | 0.0 | 0.0 |
| NPMLE prior fit | 15.7 | 0.1 |
| NPMLE spectrum | 0.1 | 0.1 |
| SB linear | 0.2 | 0.0 |
| EB prior fit | 24.8 | 0.1 |
| Ours, spectrum | 4.5 | 0.1 |
| Ours, $120$-draw band | 18.4 | 0.4 |

## kasandr/userid.csv (d_obs=1690, n=87148)

| Method | ms | MB |
|---|---:|---:|
| Naive $1/p$ | 0.0 | 0.0 |
| Good--Turing | 0.4 | 0.0 |
| SVD deconvolution | 350.6 | 6.0 |
| Chebyshev | 1.1 | 0.4 |
| NPMLE prior fit | 724.8 | 24.8 |
| NPMLE spectrum | 6.6 | 12.1 |
| SB linear | 0.1 | 0.1 |
| EB prior fit | 1590.7 | 31.5 |
| Ours, spectrum | 1485.8 | 13.4 |
| Ours, $120$-draw band | 1013.2 | 368.0 |
