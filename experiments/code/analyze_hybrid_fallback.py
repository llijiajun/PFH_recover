"""Hybrid fallback evidence for the fifth-round review, Issue 2.

This script discusses a hybrid that uses the fitted inverse
when the fitted shape a0 is interior and a classical estimator when a0 sits at
the optimizer's lower bound. This script evaluates that switch on the archived
per-case results. It runs no new population simulation. All inputs are the
committed CSV and JSON artifacts under experiments/results.

Outputs
  results/hybrid_fallback/summary.json   every number quoted in the paper
  results/hybrid_fallback/report.md      human-readable audit trail
  experiments/results/tables/tab_hybrid_fallback.tex   table emitted from JSON
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "experiments" / "results"
OUT_TABLES = ROOT / "experiments" / "results" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

MODERN = RES / "modern_baselines" / "modern_metrics.csv"
FITTED = RES / "fitted_prior_validation" / "fitted_metrics.csv"
DATA_DEP = RES / "data_dependency" / "summary.json"

BOUND = 1e-3
TOL = 1.001e-3
OUT = RES / "hybrid_fallback"

EST = {
    "fitted": "eb_fit_w1",
    "good_turing": "good_turing_w1",
    "smoothed_bayesian": "smooth_bayes_w1",
    "naive": "naive_w1",
}


def rows(path):
    return list(csv.DictReader(path.open()))


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and np.isfinite(v) else None


def med(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def load():
    modern = {(r["dataset"], r["attribute"], float(r["p"])): r
              for r in rows(MODERN)}
    fitted = {(r["dataset"], r["attribute"], float(r["p"])): r
              for r in rows(FITTED)}
    recs = []
    for key, fr in fitted.items():
        mr = modern.get(key)
        a0 = num(fr.get("a0_eb"))
        if mr is None or a0 is None:
            continue
        recs.append({
            "key": key,
            "a0": a0,
            "flagged": a0 <= TOL,
            **{name: num(mr.get(col)) for name, col in EST.items()},
        })
    return recs


def strategy(rec, fallback):
    if not rec["flagged"] or fallback is None:
        return rec["fitted"]
    return rec[fallback]


def summarise(recs):
    flagged = [r for r in recs if r["flagged"]]
    interior = [r for r in recs if not r["flagged"]]
    strategies = {
        "fitted_always": None,
        "hybrid_fitted_good_turing": "good_turing",
        "hybrid_fitted_smoothed_bayesian": "smoothed_bayesian",
        "hybrid_fitted_naive": "naive",
        "good_turing_always": "good_turing",
        "smoothed_bayesian_always": "smoothed_bayesian",
    }
    out = {}
    for name, fb in strategies.items():
        if name.endswith("_always") and name != "fitted_always":
            vals = [r[fb] for r in recs]
        else:
            vals = [strategy(r, fb) for r in recs]
        entry = {"median_w1_all": med(vals)}
        if name.startswith("hybrid"):
            entry["median_w1_flagged"] = med(
                [strategy(r, fb) for r in flagged])
            entry["median_w1_reported"] = med(
                [strategy(r, fb) for r in interior])
            improved = sum(
                1 for r in flagged
                if r["fitted"] is not None and r[fb] is not None
                and r[fb] < r["fitted"])
            entry["flagged_cases_improved"] = improved
            entry["flagged_cases_n"] = len(flagged)
            pairs = [(r["fitted"], r[fb]) for r in flagged
                     if r["fitted"] is not None and r[fb] is not None]
            if pairs:
                x = np.asarray([a for a, _ in pairs])
                y = np.asarray([b for _, b in pairs])
                entry["paired_wilcoxon_p_two_sided"] = float(
                    wilcoxon(x, y, alternative="two-sided").pvalue)
                entry["paired_median_diff_fitted_minus_fallback"] = float(
                    np.median(x - y))
        out[name] = entry
    out["refusal_policy"] = {
        "rule": "return the point histogram only when a0 > 1e-3",
        "n_reported": len(interior),
        "n_flagged": len(flagged),
        "median_w1_reported": med([r["fitted"] for r in interior]),
        "flagged_group_returns": "boundary flag and no point histogram",
    }
    out["groups"] = {
        "n_cases": len(recs),
        "flagged_median_by_estimator": {
            k: med([r[k] for r in flagged]) for k in EST},
        "reported_median_by_estimator": {
            k: med([r[k] for r in interior]) for k in EST},
    }
    return out


def heavy_tail(dep):
    agg = dep["agg"]
    families = sorted({k.split("|")[0] for k in agg})
    out = {}
    for fam in families:
        rates = sorted((k.split("|")[1], v) for k, v in agg.items()
                       if k.startswith(fam + "|"))
        out[fam] = {
            "ratio_ours_over_gt": {r: v["ratio"] for r, v in rates},
            "gt_better_at_all_rates": all(v["ratio"] > 1 for _, v in rates),
            "lowest_rate_ratio": rates[0][1]["ratio"],
        }
    return out


def write_table(payload):
    s = payload["strategies"]
    g = payload["strategies"]["groups"]["flagged_median_by_estimator"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Hybrid fallback on the $276$ real-column cases. The selector",
        "flags a case when the fitted shape reaches the optimizer lower bound of",
        "$10^{-3}$. Each hybrid reports the fitted inverse on interior fits and",
        "one classical estimator on flagged fits. Medians are $\\Wone$. Lower is",
        "better.}",
        "\\label{tab:hybrid-fallback}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Strategy & all $276$ & reported $231$ & flagged $45$ \\\\",
        "\\midrule",
    ]
    labels = [
        ("fitted_always", "Fitted inverse always"),
        ("hybrid_fitted_good_turing", "Hybrid, Good--Turing on flag"),
        ("hybrid_fitted_smoothed_bayesian", "Hybrid, SB linear on flag"),
        ("hybrid_fitted_naive", "Hybrid, naive on flag"),
        ("good_turing_always", "Good--Turing always"),
        ("smoothed_bayesian_always", "SB linear always"),
    ]
    for key, label in labels:
        e = s[key]
        a = f"{e['median_w1_all']:.4f}"
        if key.startswith("hybrid"):
            b = f"{e['median_w1_reported']:.4f}"
            c = f"{e['median_w1_flagged']:.2f}"
        else:
            b, c = "--", "--"
        lines.append(f"{label} & {a} & {b} & {c} \\\\")
    lines += [
        "\\midrule",
        f"Flagged group, fitted & -- & -- & {g['fitted']:.2f} \\\\",
        f"Flagged group, Good--Turing & -- & -- & {g['good_turing']:.2f} \\\\",
        f"Flagged group, SB linear & -- & -- & {g['smoothed_bayesian']:.2f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    (OUT_TABLES / "tab_hybrid_fallback.tex").write_text("\n".join(lines))


def main():
    recs = load()
    dep = json.loads(DATA_DEP.read_text())
    payload = {
        "inputs": {
            "modern_metrics": str(MODERN.relative_to(ROOT)),
            "fitted_metrics": str(FITTED.relative_to(ROOT)),
            "data_dependency": str(DATA_DEP.relative_to(ROOT)),
        },
        "selector": {
            "rule": "flag when fitted a0 <= 1.001e-3 (optimizer lower bound 1e-3)",
            "computed_from": "the thinned sample alone, no ground truth",
        },
        "strategies": summarise(recs),
        "table_iv_heavy_tails": heavy_tail(dep),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_table(payload)
    s = payload["strategies"]
    rep = [
        "# Hybrid fallback evidence (fifth-round review, Issue 2)",
        "",
        f"Cases: {payload['strategies']['refusal_policy']['n_reported']} reported, "
        f"{payload['strategies']['refusal_policy']['n_flagged']} flagged.",
        "",
        "## Median W1 by strategy",
        "",
        "| strategy | all 276 | reported | flagged | flagged improved | Wilcoxon p |",
        "|---|---|---|---|---|---|",
    ]
    for name, e in s.items():
        if name in ("refusal_policy", "groups"):
            continue
        rep.append(
            f"| {name} | {e['median_w1_all']:.5f} | "
            f"{e.get('median_w1_reported', float('nan')):.5f} | "
            f"{e.get('median_w1_flagged', float('nan')):.4f} | "
            f"{e.get('flagged_cases_improved', '--')}/"
            f"{e.get('flagged_cases_n', '--')} | "
            f"{e.get('paired_wilcoxon_p_two_sided', float('nan')):.4g} |")
    rep += [
        "",
        "The pooled median is unchanged by any hybrid because only 45 of 276",
        "cases change and the median index falls inside the reported group.",
        "",
        "## Table IV heavy-tail interaction",
        "",
    ]
    for fam, e in payload["table_iv_heavy_tails"].items():
        rep.append(f"- {fam}: lowest-rate ours/GT ratio "
                   f"{e['lowest_rate_ratio']:.2f}, "
                   f"GT better at all rates = {e['gt_better_at_all_rates']}")
    (OUT / "report.md").write_text("\n".join(rep) + "\n")
    print(f"wrote {OUT / 'summary.json'}")
    print(f"wrote {OUT / 'report.md'}")
    print(f"wrote {OUT_TABLES / 'tab_hybrid_fallback.tex'}")
    for name in ("fitted_always", "hybrid_fitted_good_turing",
                 "hybrid_fitted_smoothed_bayesian"):
        print(name, s[name])


if __name__ == "__main__":
    main()
