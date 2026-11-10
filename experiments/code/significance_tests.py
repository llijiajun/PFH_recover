"""Paired significance tests for the 276-case baseline comparison.

Reads the two archived CSVs, computes Wilcoxon signed-rank tests, paired win
rates, and bootstrap confidence intervals for the median NDV relative error,
then writes experiments/results/baseline_validation/significance.json.
All randomness is seeded, so the paper numbers regenerate exactly.

Run:  python significance_tests.py
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "baseline_validation")
B = os.path.join(OUT, "metrics.csv")
F = os.path.join(HERE, "..", "results", "fitted_prior_validation",
                 "fitted_metrics.csv")


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def vals(rows, key):
    return np.array([float(r[key]) for r in rows
                     if r[key] not in ("", "nan")
                     and np.isfinite(float(r[key]))], dtype=float)


def boot_ci(x, seed=0, n_boot=4000):
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(rng.choice(x, size=x.size, replace=True))
                      for _ in range(n_boot)])
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    b, f = load(B), load(F)
    arms = {
        "ours_oracle": vals(b, "NDVrel_ours"),
        "ours_fitted": vals(f, "eb_ndv_rel"),
        "naive_dp": vals(b, "NDVrel_naive_ht"),
        "haas_mm1": vals(b, "NDVrel_haas_mm1"),
        "svd": vals(b, "NDVrel_svd"),
        "good_turing": vals(b, "NDVrel_good_turing"),
        "chao1": vals(b, "NDVrel_chao1"),
    }
    n = int(arms["ours_oracle"].size)
    out = {"n_cases": n, "bootstrap_seed": 0, "n_bootstrap": 4000,
           "medians": {}, "median_ci95": {}, "paired": {}}

    for k, x in arms.items():
        out["medians"][k] = float(np.median(x))
        out["median_ci95"][k] = boot_ci(x)

    for a in ("ours_oracle", "ours_fitted"):
        for c in ("naive_dp", "svd", "good_turing", "chao1"):
            x, y = arms[a][:n], arms[c][:n]
            m = min(x.size, y.size)
            x, y = x[:m], y[:m]
            out["paired"][f"{a}_vs_{c}"] = {
                "win_rate": float(np.mean(x < y)),
                "wilcoxon_p_less": float(wilcoxon(x, y, alternative="less").pvalue),
            }

    with open(os.path.join(OUT, "significance.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    for k in ("ours_oracle", "ours_fitted", "naive_dp"):
        lo, hi = out["median_ci95"][k]
        print(f"{k:14s} median={out['medians'][k]:.4f}  CI=[{lo:.4f}, {hi:.4f}]")
    for key, r in out["paired"].items():
        if "naive_dp" in key:
            print(f"{key:32s} win={r['win_rate']:.3f}  p={r['wilcoxon_p_less']:.3g}")
    print("wrote significance.json")


if __name__ == "__main__":
    main()
