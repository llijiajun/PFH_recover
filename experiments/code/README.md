# Reproduction code

This directory holds the executable parts of the PFH_recover artifact. The fastest way to run
everything is the notebook `reproduce_all.ipynb` at the repository root. That notebook lists the
scripts in dependency order and runs each one in a fresh subprocess.

The directory contains no manuscript text, no LaTeX build chain and no code that produces the
figures embedded in the manuscript.

## Layout

| Path | Purpose |
| --- | --- |
| `d4d_logic.py` | Binomial thinning, the closed-form posterior, the per-frequency band, spectrum readout and W1 |
| `d4d_metrics.py` | Per-type posterior means, single-attribute recovery and the oracle prior |
| `d4d_baselines.py` | Classical spectrum and distinct-count baselines |
| `d4d_modern_baselines.py` | NPMLE, smoothed-Bayes and downstream functional baselines |
| `eb_prior.py` | Empirical-Bayes fit of the Gamma-Poisson prior from the observed sample alone |
| `validate_*.py` | Experiments that build the per-case metric tables from the aggregated inputs |
| `analyze_*.py` | Analyses that read those tables and emit summaries, reports and table fragments |
| `make_*.py` | Controlled grids, diagnostic panels and text exports of the archived results |
| `verify_*.py`, `significance_tests.py`, `benchmark_runtime.py` | Self-checks, paired tests and timing |

The five core modules import each other by plain module name. Keep them together in this
directory. Every script resolves its own paths from `__file__`, so no absolute path to a local
machine is needed and any working directory works.

## Inputs and outputs

Archived inputs live in `../results/`. Four grids from earlier controlled experiments are kept in
`../results/archived_inputs/` because several analyses read them. Those files are inputs, never
outputs of a reproduction run.

Scripts write to `../results/`, to `../results/tables/` for text table fragments and to
`../results/figures/` for local inspection panels. Those last two directories are created on
demand and are ignored by Git. The text fragments are plain exports of computed numbers. They are
not used to build the manuscript.

All quoted numbers must be traceable to a `summary.json`, a CSV or a report under `../results/`.
Check the corresponding archive before changing any value.

## Environment

Python 3.11 or newer with `numpy`, `scipy` and `matplotlib`. Open the root notebook with
`jupyter`. No PyTorch and no GPU are required by any script in this directory.

Run one script directly from any directory.

```bash
python3 experiments/code/analyze_pair_unification.py
```

The internal names `plausible_set`, `de-thin` and `prior_from_truth` predate the manuscript
wording. The root `README.md` maps them to the current terms.
