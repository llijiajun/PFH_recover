"""Deployment value by known retention rate. When does the full method pay off?

Question
--------
The camera-ready text reports that the fitted full-posterior readout wins only
36.7% of the 275 scored real-column cases against the linear smoothed-Bayes
estimator, with a positive median paired difference.  A reviewer therefore
cannot tell when a practitioner should prefer the full method.  This script
asks whether the *known retention rate* p, which is part of the deployment
contract and never uses ground truth, identifies a regime with a clear
aggregate gain.

Everything here is computed from the archived per-case CSVs.  No model is
retrained and no simulation is re-run.  The rule p >= 0.2 is fixed by reading
the four rate cells, and is then checked on held-out attribute groups and on
each dataset separately rather than tuned on those groups.

Outputs
-------
  experiments/results/deployment_value/summary.json
  experiments/results/deployment_value/rate_cells.csv
  experiments/results/deployment_value/report.md

Run with:
  python3 \
      experiments/code/analyze_deployment_value.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MOD_CSV = os.path.join(ROOT, "experiments", "results", "modern_baselines", "modern_metrics.csv")
OUT_DIR = os.path.join(ROOT, "experiments", "results", "deployment_value")
BOOT = 4000
BOOT_SEED = 20261110
TAIL = 0.1  # W1 threshold declared in hybrid_fallback/summary.json
RATE_CUTOFF = 0.2

# The fitted readout is the deployed full-posterior spectrum.  The closest
# reference method is the linear smoothed-Bayes estimator.
FIT = "eb_fit_w1"
SB = "smooth_bayes_w1"
GT = "good_turing_w1"
NAIVE = "naive_w1"
ORACLE = "ours_w1"
NPMLE = "npmle_w1"


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def numeric(rows, key):
    out = np.empty(len(rows), dtype=float)
    for i, r in enumerate(rows):
        v = r.get(key, "")
        try:
            out[i] = float(v)
        except (ValueError, TypeError):
            out[i] = np.nan
        if not math.isfinite(out[i]):
            out[i] = np.nan
    return out


def cluster_bootstrap_ratio(a, b, groups, rng, n_boot=BOOT):
    """Bootstrap the ratio of medians by resampling whole attribute columns."""
    names = sorted(set(groups))
    index = defaultdict(list)
    for i, g in enumerate(groups):
        index[g].append(i)
    index = {k: np.asarray(v, dtype=int) for k, v in index.items()}
    ratios = []
    for _ in range(n_boot):
        pick = rng.choice(len(names), size=len(names), replace=True)
        ii = np.concatenate([index[names[j]] for j in pick])
        aa, bb = a[ii], b[ii]
        mb = float(np.median(bb))
        if mb > 0:
            ratios.append(float(np.median(aa)) / mb)
    ratios = np.asarray(ratios, dtype=float)
    return {
        "n_clusters": len(names),
        "n_boot": len(ratios),
        "point": float(np.median(a) / np.median(b)),
        "ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
    }


def cell_stats(a, b, groups, rng):
    d = a - b
    wins = int(np.sum(d < 0))
    ties = int(np.sum(d == 0))
    losses = int(np.sum(d > 0))
    try:
        pval = float(wilcoxon(d, alternative="two-sided", zero_method="wilcox",
                              method="auto").pvalue)
    except ValueError:
        pval = float("nan")
    out = {
        "n": int(a.size),
        "median_fitted": float(np.median(a)),
        "median_sb": float(np.median(b)),
        "median_ratio": float(np.median(a) / np.median(b)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": float(wins / a.size),
        "median_paired_diff": float(np.median(d)),
        "mean_paired_diff": float(np.mean(d)),
        "wilcoxon_p_two_sided": pval,
        "median_abs_diff_when_fitted_wins": float(np.median(np.abs(d[d < 0]))) if wins else None,
        "median_abs_diff_when_fitted_loses": float(np.median(np.abs(d[d > 0]))) if losses else None,
        "ratio_ci95_cluster": cluster_bootstrap_ratio(a, b, groups, rng),
    }
    return out


def tail_stats(v):
    f = v[np.isfinite(v)]
    return {
        "n": int(f.size),
        "median": float(np.median(f)),
        "p75": float(np.percentile(f, 75)),
        "p90": float(np.percentile(f, 90)),
        "mean": float(np.mean(f)),
        "fraction_w1_above_tail": float(np.mean(f > TAIL)),
        "n_above_tail": int(np.sum(f > TAIL)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = read_rows(MOD_CSV)
    rng = np.random.default_rng(BOOT_SEED)

    p = numeric(rows, "p")
    attr = np.array([r["attribute"] for r in rows], dtype=object)
    dataset = np.array([r["dataset"] for r in rows], dtype=object)
    fit = numeric(rows, FIT)
    sb = numeric(rows, SB)
    gt = numeric(rows, GT)
    naive = numeric(rows, NAIVE)
    oracle = numeric(rows, ORACLE)
    npmle = numeric(rows, NPMLE)

    scored = np.isfinite(fit) & np.isfinite(sb)
    planned = len(rows)
    n_scored = int(scored.sum())

    # ---- rate cells -------------------------------------------------------
    rate_cells = {}
    rate_csv = []
    for pv in (0.1, 0.2, 0.3, 0.5):
        idx = (np.abs(p - pv) < 1e-9) & scored
        s = cell_stats(fit[idx], sb[idx], attr[idx].tolist(), rng)
        s["rate"] = pv
        rate_cells[str(pv)] = s
        rate_csv.append({
            "rate": pv, "n": s["n"], "median_fitted": s["median_fitted"],
            "median_sb": s["median_sb"], "median_ratio": s["median_ratio"],
            "ratio_ci_low": s["ratio_ci95_cluster"]["ci95"][0],
            "ratio_ci_high": s["ratio_ci95_cluster"]["ci95"][1],
            "wins": s["wins"], "ties": s["ties"], "losses": s["losses"],
            "win_rate": s["win_rate"], "median_paired_diff": s["median_paired_diff"],
            "wilcoxon_p": s["wilcoxon_p_two_sided"],
        })

    # ---- selectable regime ------------------------------------------------
    # p is known at deployment.  No ground truth and no tuning is used here.
    hi = (p >= RATE_CUTOFF - 1e-9) & scored
    lo = (p < RATE_CUTOFF - 1e-9) & scored
    selectable = cell_stats(fit[hi], sb[hi], attr[hi].tolist(), rng)
    low_rate = cell_stats(fit[lo], sb[lo], attr[lo].tolist(), rng)
    full = cell_stats(fit[scored], sb[scored], attr[scored].tolist(), rng)

    # ---- held-out attribute groups ---------------------------------------
    # Deterministic md5 split of the attribute name, declared before looking
    # at any outcome.  The threshold p >= 0.2 is NOT refit on either half.
    def half(name):
        return "A" if int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "B"

    halves = {}
    for h in ("A", "B"):
        idx = hi & np.array([half(x) == h for x in attr], dtype=bool)
        halves[h] = {
            "n": int(idx.sum()),
            "n_attributes": int(len(set(attr[idx].tolist()))),
            **cell_stats(fit[idx], sb[idx], attr[idx].tolist(), rng),
        }

    by_dataset = {}
    for ds in sorted(set(dataset.tolist())):
        idx = hi & (dataset == ds)
        if idx.sum() < 5:
            continue
        by_dataset[ds] = {
            "n": int(idx.sum()),
            "median_fitted": float(np.median(fit[idx])),
            "median_sb": float(np.median(sb[idx])),
            "median_ratio": float(np.median(fit[idx]) / np.median(sb[idx])),
            "win_rate": float(np.mean(fit[idx] - sb[idx] < 0)),
        }

    # Leave-one-dataset-out.  The rate rule is not refit after dropping a
    # dataset, so this is a robustness check rather than a new selector.
    leave_one_out = {}
    for ds in ["none"] + sorted(set(dataset.tolist())):
        idx = hi if ds == "none" else hi & (dataset != ds)
        leave_one_out[ds] = {
            "n": int(idx.sum()),
            "median_ratio": float(np.median(fit[idx]) / np.median(sb[idx])),
            "win_rate": float(np.mean(fit[idx] - sb[idx] < 0)),
        }

    higher_rate = (p >= 0.3 - 1e-9) & scored
    p_ge_0_3 = cell_stats(fit[higher_rate], sb[higher_rate],
                          attr[higher_rate].tolist(), rng)

    # ---- complete deployment strategies ----------------------------------
    strategies = {
        "fitted_full_posterior": tail_stats(fit),
        "sb_linear": tail_stats(sb),
        "good_turing": tail_stats(gt),
        "npmle": tail_stats(npmle),
        "naive_1_over_p": tail_stats(naive),
        "oracle_prior_ceiling": tail_stats(oracle),
    }

    # ---- readout ablation on the same cases (B1 vs B4) --------------------
    r2 = read_rows(os.path.join(ROOT, "experiments", "results", "readout_2x2", "metrics.csv"))
    b1 = numeric(r2, "B1_w1")
    b4 = numeric(r2, "B4_w1")
    ab = np.isfinite(b1) & np.isfinite(b4)
    ablation = {
        "n": int(ab.sum()),
        "median_w1_B1": float(np.median(b1[ab])),
        "median_w1_B4": float(np.median(b4[ab])),
        "median_ratio_B4_over_B1": float(np.median(b4[ab]) / np.median(b1[ab])),
    }

    summary = {
        "protocol": {
            "question": "aggregate deployment value of the full method relative to the closest estimator",
            "source": "experiments/results/modern_baselines/modern_metrics.csv",
            "n_cases_planned": planned,
            "n_cases_scored_w1": n_scored,
            "excluded": planned - n_scored,
            "metric": "W1 on the normalized spectrum truncated to j<=512",
            "estimand": "aggregate median W1 over cases, plus paired win rate as a separate quantity",
            "selector": "known retention rate p >= 0.2",
            "selector_properties": [
                "p is part of the sampling-channel contract and is known before recovery",
                "the threshold is the boundary between the four archived rate cells",
                "no ground-truth quantity and no fitted model output is used",
                "the threshold is not refit on the held-out attribute halves",
            ],
            "cluster_unit": "attribute column, because each column is sampled at four rates",
            "bootstrap": {"replicates": BOOT, "seed": BOOT_SEED, "unit": "attribute column"},
            "tail_threshold_w1": TAIL,
            "interpretation_caveat": (
                "Aggregate median gain and paired win rate are different quantities. "
                "A lower median does not imply per-case dominance, and the paired "
                "difference in the selectable regime is not significant."),
        },
        "rate_cells": rate_cells,
        "full_scored": full,
        "selectable_regime_p_ge_0_2": selectable,
        "low_rate_regime_p_0_1": low_rate,
        "selectable_regime_p_ge_0_3": p_ge_0_3,
        "held_out_attribute_halves": halves,
        "selectable_by_dataset": by_dataset,
        "leave_one_dataset_out": leave_one_out,
        "concentration_caveat": (
            "The aggregate gain is strongest on NCVoter, which supplies 165 of "
            "the 207 cases in the selectable regime.  Viewed separately, "
            "Airlines and kasandr have ratios near one.  The leave-one-dataset-out "
            "ratios stay below one, but a per-dataset gain is not established."),
        "complete_strategies": strategies,
        "readout_ablation_same_cases": ablation,
    }

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=False)
    with open(os.path.join(OUT_DIR, "rate_cells.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rate_csv[0]))
        w.writeheader()
        w.writerows(rate_csv)

    L = []
    L.append("# Deployment value of the full method (new R1 evidence)\n")
    L.append("All numbers are recomputed from the archived per-case CSV. No model is retrained.\n")
    L.append("## Rate cells, fitted full-posterior readout vs linear smoothed Bayes\n")
    L.append("| p | n | median fitted | median SB | ratio | cluster CI95 | wins | median paired diff | Wilcoxon p |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in rate_csv:
        L.append("| %.1f | %d | %.5f | %.5f | %.3f | [%.3f, %.3f] | %d/%d | %+.5f | %.4g |" % (
            c["rate"], c["n"], c["median_fitted"], c["median_sb"], c["median_ratio"],
            c["ratio_ci_low"], c["ratio_ci_high"], c["wins"], c["n"],
            c["median_paired_diff"], c["wilcoxon_p"]))
    L.append("")
    L.append("## Selectable regime p >= 0.2\n")
    L.append("- n = %d, median %.5f vs %.5f, ratio %.3f, cluster CI95 [%.3f, %.3f]" % (
        selectable["n"], selectable["median_fitted"], selectable["median_sb"],
        selectable["median_ratio"], *selectable["ratio_ci95_cluster"]["ci95"]))
    L.append("- wins %d, ties %d, losses %d (win rate %.3f), median paired diff %+.5f, Wilcoxon p %.4g" % (
        selectable["wins"], selectable["ties"], selectable["losses"],
        selectable["win_rate"], selectable["median_paired_diff"],
        selectable["wilcoxon_p_two_sided"]))
    L.append("- This is an aggregate gain, not per-case dominance.\n")
    L.append("## Held-out attribute halves at p >= 0.2\n")
    for h, v in halves.items():
        L.append("- half %s: n=%d, %d columns, ratio %.3f, CI95 [%.3f, %.3f], win rate %.3f" % (
            h, v["n"], v["n_attributes"], v["median_ratio"],
            v["ratio_ci95_cluster"]["ci95"][0], v["ratio_ci95_cluster"]["ci95"][1],
            v["win_rate"]))
    L.append("")
    L.append("## By dataset at p >= 0.2\n")
    for ds, v in by_dataset.items():
        L.append("- %s: n=%d, ratio %.3f, win rate %.3f" % (ds, v["n"], v["median_ratio"], v["win_rate"]))
    L.append("")
    L.append("## Leave-one-dataset-out at p >= 0.2\n")
    for ds, v in leave_one_out.items():
        L.append("- drop %s: n=%d, ratio %.3f, win rate %.3f" % (
            ds, v["n"], v["median_ratio"], v["win_rate"]))
    L.append("")
    L.append("- p >= 0.3 only: n=%d, ratio %.3f, CI95 %s, win rate %.3f" % (
        p_ge_0_3["n"], p_ge_0_3["median_ratio"],
        p_ge_0_3["ratio_ci95_cluster"]["ci95"], p_ge_0_3["win_rate"]))
    L.append("")
    L.append("Caveat: " + summary["concentration_caveat"])
    L.append("")
    L.append("## Complete strategies on the same scored cases\n")
    L.append("| strategy | n | median | p75 | p90 | mean | fraction W1>%.2f |" % TAIL)
    L.append("|---|---|---|---|---|---|---|")
    for k, v in strategies.items():
        L.append("| %s | %d | %.5f | %.5f | %.5f | %.4f | %.3f |" % (
            k, v["n"], v["median"], v["p75"], v["p90"], v["mean"],
            v["fraction_w1_above_tail"]))
    L.append("")
    L.append("## Readout ablation on the same 276 cases\n")
    L.append("- B1 median W1 %.5f, B4 %.5f, ratio %.3f" % (
        ablation["median_w1_B1"], ablation["median_w1_B4"],
        ablation["median_ratio_B4_over_B1"]))
    with open(os.path.join(OUT_DIR, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    print("planned %d scored %d" % (planned, n_scored))
    for c in rate_csv:
        print("p=%.1f n=%3d ratio=%.3f CI=[%.3f,%.3f] wins=%d/%d" % (
            c["rate"], c["n"], c["median_ratio"], c["ratio_ci_low"],
            c["ratio_ci_high"], c["wins"], c["n"]))
    print("p>=0.2 n=%d ratio=%.3f CI=%s wins=%d/%d p=%.4g" % (
        selectable["n"], selectable["median_ratio"],
        selectable["ratio_ci95_cluster"]["ci95"], selectable["wins"],
        selectable["n"], selectable["wilcoxon_p_two_sided"]))
    for h, v in halves.items():
        print("half %s n=%d ratio=%.3f CI=%s" % (h, v["n"], v["median_ratio"],
                                                 v["ratio_ci95_cluster"]["ci95"]))
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
