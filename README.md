# PFH_recover

Reproducibility artifact for *Recovering Population Frequency Histograms from a Known Sampling Rate*.

A population frequency histogram counts how many distinct values appear exactly $j$ times in a table. Sampling at a known retention rate removes most of that histogram. We model sampling as a binomial thinning channel, invert the channel in closed form under a fitted Gamma-Poisson prior, and read out the full posterior as a frequency spectrum. The estimator needs no training data, no reverse chain and no truth-informed prior. A degenerate prior fit raises a boundary flag and withholds the point histogram.

This repository holds the code, the aggregated inputs and the archived results only. The manuscript, its LaTeX sources, the compiled PDFs and the code that builds the figures embedded in the manuscript are not part of this release. The manuscript is under review at the time of this release, and this repository is not a separate publication.

## Contents

| Path | What it holds |
| --- | --- |
| `reproduce_all.ipynb` | One-click notebook that reruns every analysis and can also rescore the per-case metrics |
| `experiments/code/` | Reproduction scripts and the five core algorithm modules |
| `experiments/data/` | Real-column inputs, already aggregated to value and count pairs |
| `experiments/results/` | Archived JSON, CSV, reports and per-case metrics behind every reported number |
| `experiments/results/archived_inputs/` | Four archived grids that several analysis scripts read as inputs |

All reported numbers come from the archived files under `experiments/results/`. Analysis scripts read those archives and write new reports, JSON files, table fragments and inspection plots back under `experiments/results/`.

## One-click reproduction

Install the dependencies, then open `reproduce_all.ipynb` from the repository root and use *Kernel → Restart & Run All*.

```bash
python3 -m pip install -r requirements.txt
jupyter notebook reproduce_all.ipynb
```

The notebook locates the repository root on its own, checks that `numpy`, `scipy` and `matplotlib` are present, runs the analysis scripts as fresh subprocesses, prints progress, and finishes with a pass or fail table and a short readout of the regenerated summaries. Failures never abort the run. No GPU and no PyTorch are needed.

The default run finishes in a few minutes. It re-runs every paired test, analysis, diagnostic, theory check and local plot from the archived per-case metrics. Seven scripts that re-score all 276 real columns from the aggregated inputs are skipped by default because they take hours. Set `SKIP = set()` in the configuration cell to include them and reproduce the complete chain. `TIMEOUT` controls the per-script limit. Scripts that need archived grids read them from `experiments/results/archived_inputs/`.

Running the notebook refreshes `experiments/results/` in place. That is the intended behaviour of a reproduction run. Compare with `git diff experiments/results`. Restore the committed archives and inspect any files created by the run with the following commands.

```bash
git checkout -- experiments/results
git status --short experiments/results
```

Generated table fragments, local inspection plots and `reproduction_report.json` are ignored by Git.

## Command line reproduction

Every script can also be run on its own. All paths are resolved from the script location, so the current working directory does not matter.

```bash
python3 experiments/code/analyze_pair_unification.py
python3 experiments/code/analyze_deployment_value.py
python3 experiments/code/validate_baselines.py
python3 experiments/code/make_data_dependency.py
```

The five core modules `d4d_logic.py`, `d4d_metrics.py`, `d4d_baselines.py`, `d4d_modern_baselines.py` and `eb_prior.py` import each other by module name. Keep them in the same directory. Run the scripts in the order listed in `reproduce_all.ipynb` when reproducing a full chain, because several analyses read per-case CSVs produced by earlier stages.

Scripts that emit table fragments write them to `experiments/results/tables/`. Scripts that draw diagnostic panels write them to `experiments/results/figures/`. Both directories are created on demand. These outputs are local inspection material and are not manuscript figures.

## Data

The files under `experiments/data/real/` are aggregated frequency pairs with two columns. The first column is a value or a bucket label. The second column is its observed count. No individual records are included. The two sources are the Star Schema Benchmark and the North Carolina voter registration snapshot. Every result that depends on these inputs is already archived, so the data files are provided for inspection and re-scoring.

The four files in `experiments/results/archived_inputs/` are archived grids from earlier controlled experiments. They are inputs to the prior-sensitivity, band-calibration and unknown-support analyses, not outputs of the current run.

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

The code documentation predates the final manuscript wording, so a few early working names remain inside the scripts. The table above is the authoritative mapping.

Internal review documents, editor backups, one-off ablation archives, LaTeX build intermediates and the manuscript sources are not part of this release.

## License

MIT. See `LICENSE`.
