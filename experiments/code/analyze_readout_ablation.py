"""Statistics for the shared-prior readout ablation (review round 2, Issue 1).

The two rows of Table V reuse one fitted Gamma--Poisson prior and one truncation.
They differ only in the readout.  This script collects the paired comparison of
those two rows and the boundary-fit diagnostic used for the fallback discussion.

Inputs (all archived, no new random experiment)
  experiments/results/baseline_validation/w1_tests.json
      authoritative paired Wilcoxon tests over the 275 scored real cases.
      Key fitted_vs_sb compares the deployed fitted posterior (full-posterior
      readout) with the smoothed-Bayesian linear row (rounded readout).
  experiments/results/modern_baselines/modern_metrics.csv
      per-case W1 for every estimator in Table V.
  experiments/results/fitted_prior_validation/fitted_metrics.csv
      column a0_eb from eb_prior.fitted_prior_eb, whose lower bound is 1e-3,
      and eb_w1, the W1 of the empirical-Bayes inverse used in Section VI-D.

Outputs
  experiments/results/readout_ablation/summary.json
  experiments/results/readout_ablation/report.md
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W1_TESTS = ROOT / "experiments/results/baseline_validation/w1_tests.json"
MODERN_CSV = ROOT / "experiments/results/modern_baselines/modern_metrics.csv"
FITTED_CSV = ROOT / "experiments/results/fitted_prior_validation/fitted_metrics.csv"
OUT_DIR = ROOT / "experiments/results/readout_ablation"

BOUND = 1e-3
TOL = 1.001e-3  # least squares returns values a few 1e-15 above the bound
CANDIDATES = ("eb_fit_w1", "smooth_bayes_w1", "good_turing_w1", "naive_w1")


def rows(path):
    return list(csv.DictReader(path.open()))


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def main():
    tests = json.loads(W1_TESTS.read_text())
    paired = tests["fitted_vs_sb"]

    modern = {(r["dataset"], r["attribute"], float(r["p"])): r for r in rows(MODERN_CSV)}
    fitted = {(r["dataset"], r["attribute"], float(r["p"])): r for r in rows(FITTED_CSV)}

    boundary, interior = [], []
    for key, fr in fitted.items():
        a0 = num(fr["a0_eb"])
        mr = modern.get(key)
        if a0 is None or mr is None:
            continue
        rec = {
            "key": key,
            "a0": a0,
            "eb_w1": num(fr["eb_w1"]),
            **{c: num(mr.get(c)) for c in CANDIDATES},
        }
        (boundary if a0 <= TOL else interior).append(rec)

    def med(rs, c):
        v = [r[c] for r in rs if r[c] is not None]
        return median(v) if v else None

    def win_count(rs):
        out = defaultdict(int)
        for r in rs:
            vals = [(c, r[c]) for c in CANDIDATES if r[c] is not None]
            if vals:
                out[min(vals, key=lambda t: t[1])[0]] += 1
        return dict(out)

    summary = {
        "paired_readout_test": {
            "source": str(W1_TESTS.relative_to(ROOT)),
            "n_cases": paired["n"],
            "win_rate_full_posterior_readout": paired["win_rate"],
            "wilcoxon_p": paired["p"],
            "median_w1_fitted": paired["med_a"],
            "median_w1_rounded": paired["med_b"],
            "note": ("Both rows share one fitted prior and one truncation. The test "
                     "therefore compares the two readouts."),
        },
        "boundary_diagnostic": {
            "source_real": str(FITTED_CSV.relative_to(ROOT)),
            "source_spectra": str(MODERN_CSV.relative_to(ROOT)),
            "a0_bound": BOUND,
            "n_boundary": len(boundary),
            "n_interior": len(interior),
            "eb_w1_median_boundary": med(boundary, "eb_w1"),
            "eb_w1_median_interior": med(interior, "eb_w1"),
            "median_w1_by_estimator_boundary": {c: med(boundary, c) for c in CANDIDATES},
            "per_case_best_count_boundary": win_count(boundary),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    p = summary["paired_readout_test"]
    b = summary["boundary_diagnostic"]
    L = ["# Shared-prior readout ablation and boundary-fit diagnostic", "",
         "## Paired comparison of the two readouts", "",
         f"- Source `{p['source']}`, {p['n_cases']} scored cases.",
         f"- Full-posterior readout wins on {p['win_rate_full_posterior_readout']:.1%} "
         f"of cases, Wilcoxon p = {p['wilcoxon_p']:.4g}.",
         f"- Median W1 {p['median_w1_fitted']:.5g} (fitted) against "
         f"{p['median_w1_rounded']:.5g} (rounded).",
         "- Both rows use one fitted prior and one truncation, so the test "
         "isolates the readout. The lower median does not become a per-case win.", "",
         "## Boundary-fit diagnostic", "",
         f"- {b['n_boundary']} of {b['n_boundary'] + b['n_interior']} cases have "
         f"$a_0 \\le$ {BOUND:g}.",
         f"- Median W1 of the empirical-Bayes inverse is "
         f"{b['eb_w1_median_boundary']:.4f} on boundary cases against "
         f"{b['eb_w1_median_interior']:.4f} on interior cases.",
         "- Median W1 on boundary cases by estimator:"]
    for c, v in b["median_w1_by_estimator_boundary"].items():
        L.append(f"  - {c}: {v:.4f}")
    L += ["- Per-case lowest-W1 counts on boundary cases: "
          + ", ".join(f"{k}={v}" for k, v in sorted(b["per_case_best_count_boundary"].items())),
          "", "The boundary group has no accurate estimator in this benchmark. "
          "Good--Turing attains the lowest W1 on the largest number of boundary "
          "cases but has a worse median than the fitted inverse, so a blanket "
          "Good--Turing fallback is not supported by these data. The supported "
          "action is to flag the fit and withhold the point histogram.", ""]
    (OUT_DIR / "report.md").write_text("\n".join(L))
    print(json.dumps(summary, indent=2))
    print("wrote", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
