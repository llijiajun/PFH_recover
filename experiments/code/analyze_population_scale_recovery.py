"""Population-scale recovery. Quantify the two population-scale errors.

Question
--------
The camera-ready text says W1 measures a normalized shape truncated to j<=512,
and that support-size error and row-mass error are distinct quantities that the
shape metrics do not assess.  A reviewer asks for the evaluation range of full
population recovery.  This script measures both quantities on the same 276
real-column cases, so the earlier caveat becomes a reported number rather than
an unquantified limitation.

Object isolation (verified against the archived columns)
--------------------------------------------------------
  true_types = sum_{j<=512} F_j, the number of types on the truncated support
  true_mass  = sum_{j<=512} j F_j, the truncated row mass, NOT the full row count N
  B1_total_types / B4_total_types are identical, because the estimated type
  count depends on the unseen-type compensation and not on the observed readout
  B1_total_mass matches smooth_bayes_pop_rel in modern_metrics.csv
  B4_total_mass matches eb_fit_pop_rel in modern_metrics.csv
  eb_fit_ndv_rel and smooth_bayes_ndv_rel both match |total_types - true_types|
  divided by true_types, up to 1e-11 numerical noise

Outputs
-------
  experiments/results/population_scale_recovery/summary.json
  experiments/results/population_scale_recovery/report.md

Run with:
  python3 \
      experiments/code/analyze_population_scale_recovery.py
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R2_CSV = os.path.join(ROOT, "experiments", "results", "readout_2x2", "metrics.csv")
MOD_CSV = os.path.join(ROOT, "experiments", "results", "modern_baselines", "modern_metrics.csv")
OUT_DIR = os.path.join(ROOT, "experiments", "results", "population_scale_recovery")
BOOT, BOOT_SEED = 4000, 20261110


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def col(rows, k):
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        try:
            v = float(r[k])
            out[i] = v if math.isfinite(v) else np.nan
        except (ValueError, TypeError, KeyError):
            out[i] = np.nan
    return out


def desc(x):
    f = x[np.isfinite(x)]
    return {
        "n": int(f.size),
        "median": float(np.median(f)),
        "p25": float(np.percentile(f, 25)),
        "p75": float(np.percentile(f, 75)),
        "p90": float(np.percentile(f, 90)),
        "mean": float(np.mean(f)),
    }


def signed_and_abs(est, truth):
    rel = (est - truth) / np.maximum(truth, 1e-9)
    m = np.isfinite(rel)
    return rel[m], np.abs(rel[m])


def cluster_ratio_ci(a, b, groups, rng):
    names = sorted(set(groups))
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    idx = {k: np.asarray(v, int) for k, v in idx.items()}
    vals = []
    for _ in range(BOOT):
        pick = rng.choice(len(names), len(names), replace=True)
        ii = np.concatenate([idx[names[j]] for j in pick])
        mb = float(np.median(b[ii]))
        if mb > 0:
            vals.append(float(np.median(a[ii])) / mb)
    vals = np.asarray(vals)
    return {
        "n_clusters": len(names),
        "point": float(np.median(a) / np.median(b)),
        "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    R = read(R2_CSV)
    M = read(MOD_CSV)
    rng = np.random.default_rng(BOOT_SEED)

    tt = col(R, "true_types")
    tm = col(R, "true_mass")
    p = col(R, "p")
    attr = np.array([r["attribute"] for r in R], dtype=object)
    rate = np.array([r["p"] for r in R], dtype=object)

    arms = {}
    for arm in ("B1", "B4"):
        s_types, a_types = signed_and_abs(col(R, arm + "_total_types"), tt)
        s_mass, a_mass = signed_and_abs(col(R, arm + "_total_mass"), tm)
        arms[arm] = {
            "type_count_signed_relerr": desc(s_types),
            "type_count_abs_relerr": desc(a_types),
            "row_mass_signed_relerr": desc(s_mass),
            "row_mass_abs_relerr": desc(a_mass),
            "_s_mass": s_mass, "_a_mass": a_mass, "_a_types": a_types,
        }

    # Cross-check against the independently archived modern-baseline columns.
    cross = {}
    for k, colname, arm, obj in (
        ("B1_total_mass_vs_smooth_bayes_pop_rel", "smooth_bayes_pop_rel", "B1", "_a_mass"),
        ("B4_total_mass_vs_eb_fit_pop_rel", "eb_fit_pop_rel", "B4", "_a_mass"),
        ("total_types_vs_eb_fit_ndv_rel", "eb_fit_ndv_rel", "B1", "_a_types"),
        ("total_types_vs_smooth_bayes_ndv_rel", "smooth_bayes_ndv_rel", "B1", "_a_types"),
    ):
        ref = col(M, colname)
        v = arms[arm][obj]
        m = np.isfinite(ref) & np.isfinite(v)
        cross[k] = {
            "n": int(m.sum()),
            "max_abs_difference": float(np.max(np.abs(ref[m] - v[m]))),
            "columns_match": bool(np.max(np.abs(ref[m] - v[m])) < 1e-6),
        }

    a1, a4 = arms["B1"]["_a_mass"], arms["B4"]["_a_mass"]
    paired = a4 - a1
    try:
        wp = float(wilcoxon(paired, alternative="two-sided", zero_method="wilcox",
                            method="auto").pvalue)
    except ValueError:
        wp = float("nan")
    improvement = {
        "n": int(a1.size),
        "median_abs_relerr_B1": float(np.median(a1)),
        "median_abs_relerr_B4": float(np.median(a4)),
        "median_ratio_B4_over_B1": float(np.median(a4) / np.median(a1)),
        "ratio_ci95_cluster": cluster_ratio_ci(a4, a1, attr.tolist(), rng),
        "n_cases_B4_better": int(np.sum(paired < 0)),
        "n_ties": int(np.sum(paired == 0)),
        "n_cases_B4_worse": int(np.sum(paired > 0)),
        "median_paired_diff": float(np.median(paired)),
        "wilcoxon_p_two_sided": wp,
        "interpretation": (
            "The full-posterior readout improves the truncated row mass but "
            "leaves the truncated type count unchanged, because the type count "
            "is set by the unseen-type compensation."),
    }

    by_rate = {}
    for pv in (0.1, 0.2, 0.3, 0.5):
        i = np.abs(p - pv) < 1e-9
        by_rate[str(pv)] = {
            "n": int(i.sum()),
            "true_types_median": float(np.median(tt[i])),
            "true_mass_median": float(np.median(tm[i])),
            "B1_mass_median": float(np.median(col(R, "B1_total_mass")[i])),
            "B4_mass_median": float(np.median(col(R, "B4_total_mass")[i])),
            "B1_mass_abs_relerr_median": float(np.median(a1[i])),
            "B4_mass_abs_relerr_median": float(np.median(a4[i])),
        }

    summary = {
        "protocol": {
            "question": "evaluation range of population-scale recovery",
            "source": "experiments/results/readout_2x2/metrics.csv (276 cases)",
            "cross_check_source": "experiments/results/modern_baselines/modern_metrics.csv",
            "support": "j<=512 truncated spectrum",
            "true_types_definition": "sum of F_j on j<=512, not the full number of distinct values V",
            "true_mass_definition": "sum of j*F_j on j<=512, not the full row count N",
            "truth_row_mass_median": float(np.median(tm[np.isfinite(tm)])),
            "truth_type_count_median": float(np.median(tt[np.isfinite(tt)])),
            "object_isolation": (
                "These two quantities are not validated by W1, TV, or any "
                "per-frequency band coverage result, and they are not estimates "
                "of the full population size."),
        },
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in arms.items()},
        "column_cross_check": cross,
        "row_mass_readout_improvement": improvement,
        "by_rate": by_rate,
    }

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    L = []
    L.append("# Population-scale recovery (new R2 evidence)\n")
    L.append("Both quantities live on the j<=512 truncated support. "
             "True row mass median %.1f, true type count median %.1f.\n" % (
                 summary["protocol"]["truth_row_mass_median"],
                 summary["protocol"]["truth_type_count_median"]))
    L.append("| arm | type count abs relerr median | row mass abs relerr median | row mass signed relerr median |")
    L.append("|---|---|---|---|")
    for arm in ("B1", "B4"):
        v = arms[arm]
        L.append("| %s | %.4f | %.4f | %+.4f |" % (
            arm, v["type_count_abs_relerr"]["median"],
            v["row_mass_abs_relerr"]["median"],
            v["row_mass_signed_relerr"]["median"]))
    L.append("")
    L.append("## Readout improvement on truncated row mass\n")
    L.append("- median abs relerr %.4f (B1) to %.4f (B4), ratio %.3f, cluster CI95 %s" % (
        improvement["median_abs_relerr_B1"], improvement["median_abs_relerr_B4"],
        improvement["median_ratio_B4_over_B1"],
        improvement["ratio_ci95_cluster"]["ci95"]))
    L.append("- B4 better on %d cases, ties %d, worse on %d, Wilcoxon p %.4g" % (
        improvement["n_cases_B4_better"], improvement["n_ties"],
        improvement["n_cases_B4_worse"], improvement["wilcoxon_p_two_sided"]))
    L.append("- " + improvement["interpretation"] + "\n")
    L.append("## By retention rate\n")
    L.append("| p | n | true types | true mass | B1 mass | B4 mass | B1 abs relerr | B4 abs relerr |")
    L.append("|---|---|---|---|---|---|---|---|")
    for pv, v in by_rate.items():
        L.append("| %s | %d | %.1f | %.1f | %.1f | %.1f | %.4f | %.4f |" % (
            pv, v["n"], v["true_types_median"], v["true_mass_median"],
            v["B1_mass_median"], v["B4_mass_median"],
            v["B1_mass_abs_relerr_median"], v["B4_mass_abs_relerr_median"]))
    L.append("")
    L.append("## Column cross-check\n")
    for k, v in cross.items():
        L.append("- %s: match=%s max diff %.2g" % (k, v["columns_match"], v["max_abs_difference"]))
    L.append("")
    L.append(summary["protocol"]["object_isolation"])
    with open(os.path.join(OUT_DIR, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    print("type count abs relerr median %.4f (both arms)" %
          arms["B1"]["type_count_abs_relerr"]["median"])
    print("row mass abs relerr B1 %.4f -> B4 %.4f, ratio %.3f CI %s, p %.4g" % (
        improvement["median_abs_relerr_B1"], improvement["median_abs_relerr_B4"],
        improvement["median_ratio_B4_over_B1"],
        improvement["ratio_ci95_cluster"]["ci95"], improvement["wilcoxon_p_two_sided"]))
    for k, v in cross.items():
        print("  cross", k, v["columns_match"], "%.2g" % v["max_abs_difference"])
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
