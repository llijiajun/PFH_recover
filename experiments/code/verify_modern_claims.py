"""Recompute every statistic quoted in the Q7 / Discussion sections.

Reads the archived per-case CSV and writes claims_verification.json so that
each number in the paper can be traced to a script rather than to prose.

Run with the project scientific python (needs scipy):
  python3 \
      experiments/code/verify_modern_claims.py
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "results", "modern_baselines")
CSV = os.path.join(OUT_DIR, "modern_metrics.csv")

NDV_METHODS = ["sgt", "jack1", "jack2", "coverage", "poly", "npmle",
               "smooth_bayes"]
SPEC_METHODS = ["naive", "svd", "good_turing", "ours", "npmle",
                "smooth_bayes", "eb_fit"]
FUNCS = ["ndv", "self_join", "rare_frac", "entropy"]


def load():
    rows = []
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v)
                except (ValueError, TypeError):
                    d[k] = v
            rows.append(d)
    return rows


def col(rows, key):
    return np.array([np.nan if not np.isscalar(r.get(key))
                     else float(r.get(key, np.nan)) for r in rows])


def finite_col(rows, key):
    v = col(rows, key)
    return v[np.isfinite(v)]


def paired(rows, a, b, alternative="less"):
    x, y = col(rows, a), col(rows, b)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    return {
        "n_pairs": int(m.sum()),
        "wilcoxon_p": float(wilcoxon(x, y, alternative=alternative).pvalue),
        "win_rate_a_less_b": float(np.mean(x < y)),
        "median_a": float(np.median(x)),
        "median_b": float(np.median(y)),
        "mean_a": float(np.mean(x)),
        "mean_b": float(np.mean(y)),
    }


def main():
    rows = load()
    out = {"n_cases": len(rows), "source": "modern_metrics.csv"}

    out["ndv"] = {}
    for m in NDV_METHODS:
        v = finite_col(rows, f"NDVrel_{m}")
        out["ndv"][m] = {
            "finite_cases": int(v.size),
            "median": float(np.median(v)),
            "mean": float(np.mean(v)),
            "within10": float(np.mean(v <= 0.10)),
        }
    for m in ["ours", "eb_fit"]:
        v = finite_col(rows, f"{m}_func_ndv")
        out["ndv"][m] = {
            "finite_cases": int(v.size),
            "median": float(np.median(v)),
            "mean": float(np.mean(v)),
            "within10": float(np.mean(v <= 0.10)),
        }

    out["spectrum_w1"] = {}
    for m in SPEC_METHODS:
        v = finite_col(rows, f"{m}_w1")
        out["spectrum_w1"][m] = {
            "finite_cases": int(v.size),
            "median": float(np.median(v)),
            "mean": float(np.mean(v)),
        }
    w = out["spectrum_w1"]
    out["spectrum_w1"]["ratio_sb_over_ours_fitted"] = (
        w["smooth_bayes"]["median"] / w["eb_fit"]["median"])

    out["functionals"] = {}
    for m in SPEC_METHODS:
        out["functionals"][m] = {
            k: float(np.median(finite_col(rows, f"{m}_func_{k}")))
            for k in FUNCS
        }
    f = out["functionals"]
    out["functionals"]["gain_self_join_naive_over_ours_fitted"] = (
        f["naive"]["self_join"] / f["eb_fit"]["self_join"])
    out["functionals"]["gain_entropy_naive_over_ours_fitted"] = (
        f["naive"]["entropy"] / f["eb_fit"]["entropy"])

    out["paired_tests"] = {
        "w1_ours_fitted_vs_sb_linear": paired(rows, "eb_fit_w1", "smooth_bayes_w1"),
        "self_join_ours_fitted_vs_raw": paired(
            rows, "eb_fit_func_self_join", "naive_func_self_join"),
        "entropy_ours_fitted_vs_raw": paired(
            rows, "eb_fit_func_entropy", "naive_func_entropy"),
        "self_join_ours_fitted_vs_sb_linear": paired(
            rows, "eb_fit_func_self_join", "smooth_bayes_func_self_join"),
    }

    a = col(rows, "smooth_bayes_func_ndv")
    b = col(rows, "eb_fit_func_ndv")
    m = np.isfinite(a) & np.isfinite(b)
    out["sb_equals_ours_fitted_ndv"] = {
        "n_pairs": int(m.sum()),
        "max_abs_diff": float(np.max(np.abs(a[m] - b[m]))),
    }

    path = os.path.join(OUT_DIR, "claims_verification.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote", path)

    print("NDV medians (%)")
    for k, v in out["ndv"].items():
        print("  %-14s %6.1f  finite %d" % (k, 100 * v["median"],
                                            v["finite_cases"]))
    print("W1 ratio SB / ours fitted: %.3f" %
          out["spectrum_w1"]["ratio_sb_over_ours_fitted"])
    print("self-join gain: %.2fx, entropy gain: %.2fx" %
          (out["functionals"]["gain_self_join_naive_over_ours_fitted"],
           out["functionals"]["gain_entropy_naive_over_ours_fitted"]))
    for k, v in out["paired_tests"].items():
        print("  %-40s p=%.4g  win=%.3f  n=%d" %
              (k, v["wilcoxon_p"], v["win_rate_a_less_b"], v["n_pairs"]))
    print("SB vs ours fitted NDV max abs diff: %.3g" %
          out["sb_equals_ours_fitted_ndv"]["max_abs_diff"])


if __name__ == "__main__":
    main()
