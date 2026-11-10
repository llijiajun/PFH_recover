# PFH_recover

Reproducibility artifact for *Recovering Population Frequency Histograms from a Known Sampling Rate*.

A population frequency histogram counts how many distinct values appear exactly $j$ times in a table. Sampling at a known retention rate removes most of that histogram. We model sampling as a binomial thinning channel, invert the channel in closed form under a fitted Gamma-Poisson prior, and read out the full posterior as a frequency spectrum. The estimator needs no training data, no reverse chain and no truth-informed prior. A degenerate prior fit raises a boundary flag and withholds the point histogram.

This repository holds the code, the aggregated inputs and the archived results only. The manuscript, its LaTeX sources and the scripts that build its figures are not part of this release. The manuscript is under review at the time of this release, and this repository is not a separate publication.

## Contents

| Path | What it holds |
| --- | --- |
| `experiments/code/` | Active reproduction scripts and two walkthrough notebooks |
| `experiments/data/` | Real-column inputs, already aggregated to value and count pairs |
| `experiments/results/` | Archived JSON, CSV, reports and figures behind every reported number |

All reported numbers come from the archived files under `experiments/results/`. Analysis scripts read those archives and write new reports, JSON files, table fragments and figures back under `experiments/results/`.

## Reproduce

Install the Python dependencies.

```bash
python3 -m pip install -r requirements.txt
```

Run an analysis script from the repository root.

```bash
python3 experiments/code/analyze_pair_unification.py
python3 experiments/code/analyze_deployment_value.py
python3 experiments/code/validate_baselines.py
python3 experiments/code/make_fof_recovery_figure.py
```

The five core modules `d4d_logic.py`, `d4d_metrics.py`, `d4d_baselines.py`, `d4d_modern_baselines.py` and `eb_prior.py` import each other by module name. Keep them in the same directory. The learned-endpoint refinement is optional and requires PyTorch.

Scripts that emit table fragments write them to `experiments/results/tables/`. Scripts that draw comparison panels write them to `experiments/results/figures/` or `experiments/code/figures/`. Both output directories are created on demand.

## Data

The files under `experiments/data/real/` are aggregated frequency pairs with two columns. The first column is a value or a bucket label. The second column is its observed count. No individual records are included. The two sources are the Star Schema Benchmark and the North Carolina voter registration snapshot. Every result that depends on these inputs is already archived, so the data files are provided for inspection and re-scoring.

## Notes on naming

The code documentation predates the final manuscript wording. The table below maps older internal names to the current terms.

| Code name | Current term |
| --- | --- |
| `plausible_set` | conditional per-frequency credible band |
| `de-thin` | invert the binomial channel |
| `prior_from_truth` | oracle prior, an information ceiling only |
| `fitted_prior_eb` | sample-fitted prior, the deployed estimate |
| B1 | rounded posterior means with unseen types in the singleton bin |
| B4 | integrated posterior probabilities with unseen types spread by the posterior profile |

The walkthrough notebooks carry an early working title. The manuscript title is the one at the top of this file.

Internal review documents, editor backups, one-off ablation archives, LaTeX build intermediates and the manuscript sources are not part of this release.

## License

MIT. See `LICENSE`.
