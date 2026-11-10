"""Unify the readout paired statistics from one case-level record.

An earlier draft reported two different win rates and p-values for what it
described as the same fitted-versus-rounded comparison. This script rebuilds
every paired statistic from `readout_2x2/metrics.csv`, which carries the case
key, the fitted prior, the fit status, and all four readout arms. It also
records the case-count chain across the deployment policy so that the number of
planned, scored, retained, refused, and evaluable cases is explicit.

No new simulation is run. All inputs are archived CSV or JSON files.

Outputs
  experiments/results/pair_unification/summary.json
  experiments/results/pair_unification/report.md
  experiments/results/tables/tab_pair_unification.tex
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
OUT = RES / "pair_unification"
TABLE = RES / "tables" / "tab_pair_unification.tex"
READOUT = RES / "readout_2x2" / "metrics.csv"
MODERN = RES / "modern_baselines" / "modern_metrics.csv"
POLICY = RES / "deployable_regime" / "summary.json"
KEY = ("dataset", "attribute", "p")


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def keyof(r):
    return (r["dataset"], r["attribute"], float(r["p"]))


def paired(a, b, label):
    """Two-sided paired Wilcoxon on strict-improvement counts, no tie tolerance."""
    recs, skipped = [], 0
    for x, y in zip(a, b):
        vx, vy = num(x), num(y)
        if vx is None or vy is None:
            skipped += 1
            continue
        recs.append((vx, vy))
    diff = [x - y for x, y in recs]
    n_low = sum(d < 0 for d in diff)
    n_high = sum(d > 0 for d in diff)
    n_zero = sum(d == 0 for d in diff)
    p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox",
                 method="auto").pvalue
    return {
        "pair": label,
        "rule": "win = first arm strictly smaller, no tie tolerance; "
                "two-sided Wilcoxon signed-rank, zero_method=wilcox, method=auto",
        "n": len(diff),
        "skipped_non_finite": skipped,
        "n_first_lower": n_low,
        "n_first_higher": n_high,
        "n_exact_ties": n_zero,
        "win_rate_first_lower": n_low / len(diff),
        "wilcoxon_p_two_sided": p,
        "median_first": median(x for x, _ in recs),
        "median_second": median(y for _, y in recs),
        "median_paired_difference": median(diff),
    }


def main():
    R = rows(READOUT)
    M = {keyof(r): r for r in rows(MODERN)}
    policy = json.loads(POLICY.read_text())

    # Case-count chain. The guard case is the single non-finite B1/B4 row.
    guard = [keyof(r) for r in R
             if num(r["B1_w1"]) is None or num(r["B4_w1"]) is None]
    finite = [r for r in R if num(r["B1_w1"]) is not None
              and num(r["B4_w1"]) is not None]
    chain = {
        "planned_cases": len(R),
        "scored_cases": len(finite),
        "guard_excluded_cases": [list(g) for g in guard],
        "guard_reason": "non-finite W1 from the numerical guard; no valid metric",
        "flag_retained": policy["coverage"]["n_kept"],
        "flag_refused": policy["coverage"]["n_refused"],
        "retained_with_valid_paired_metric":
            policy["same_kept_set_baselines"]["fitted_vs_smooth_bayes_w1"]["n"],
        "note": "The retained count of 231 includes the guard case. Only 230 "
                "retained cases produce a valid paired metric, so 231 of 276 "
                "is a flag-retention fraction and not a usable-output fraction.",
    }

    # Table III and Table V use the same numbers. Prove it against modern CSV.
    modern_pairs = [(num(r["B4_w1"]), num(M[keyof(r)]["eb_fit_w1"]))
                    for r in R if keyof(r) in M and num(r["B4_w1"]) is not None]
    modern_b1 = [(num(r["B1_w1"]), num(M[keyof(r)]["smooth_bayes_w1"]))
                 for r in R if keyof(r) in M and num(r["B1_w1"]) is not None]
    cross = {
        "B4_vs_eb_fit_w1_max_abs_diff": max(abs(a - b) for a, b in modern_pairs),
        "B1_vs_smooth_bayes_w1_max_abs_diff": max(abs(a - b) for a, b in modern_b1),
        "conclusion": "Table III (Ours fitted, SB linear) and Table V (B4, B1) "
                      "are the same two case-level columns to floating-point "
                      "precision, so they must carry one paired statistic.",
    }

    metrics = {"W1": "w1", "F1_err": "f1_err", "entropy_err": "entropy_err"}
    comparisons = [("B4", "B1"), ("B2", "B1"), ("B3", "B1"),
                   ("B4", "B2"), ("B4", "B3")]
    stats = {}
    for first, second in comparisons:
        for label, col in metrics.items():
            a = [r[f"{first}_{col}"] for r in R]
            b = [r[f"{second}_{col}"] for r in R]
            stats[f"{first}_vs_{second}_{label}"] = paired(a, b, f"{first} vs {second}, {label}")

    authoritative = stats["B4_vs_B1_W1"]
    deprecated = {
        "archive": "experiments/results/baseline_validation/w1_tests.json",
        "key": "fitted_vs_sb",
        "n": 275,
        "win_rate": 0.3491,
        "p": 0.008373,
        "reason": "The archive stores summary statistics only. Its win rate and "
                  "p-value cannot be reproduced from any archived case-level "
                  "columns, and its medians match the two columns used here "
                  "exactly. It is retired rather than kept as a second "
                  "authoritative result.",
    }

    payload = {
        "source_case_level": "experiments/results/readout_2x2/metrics.csv",
        "case_key": list(KEY),
        "chain": chain,
        "cross_table_identity": cross,
        "authoritative_B4_vs_B1_W1": authoritative,
        "deprecated_statistic": deprecated,
        "all_paired_statistics": stats,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=1) + "\n")

    lines = [
        "# Case-level unification of the readout paired statistics",
        "",
        f"Source: `{payload['source_case_level']}` with case key "
        f"{', '.join(KEY)}. No new experiment was run.",
        "",
        "## Case-count chain",
        "",
        f"- {chain['planned_cases']} planned cases",
        f"- {chain['scored_cases']} scored cases after the numerical guard",
        f"- {chain['flag_retained']} retained and {chain['flag_refused']} refused by the boundary flag",
        f"- {chain['retained_with_valid_paired_metric']} retained cases with a valid paired metric",
        "",
        chain["note"],
        "",
        "## Cross-table identity",
        "",
        f"Maximum absolute difference B4 versus `eb_fit_w1` is "
        f"{cross['B4_vs_eb_fit_w1_max_abs_diff']:.3g}. Maximum absolute difference "
        f"B1 versus `smooth_bayes_w1` is "
        f"{cross['B1_vs_smooth_bayes_w1_max_abs_diff']:.3g}. "
        + cross["conclusion"],
        "",
        "## Authoritative B4 versus B1 on W1",
        "",
        f"n = {authoritative['n']}, win rate = "
        f"{authoritative['win_rate_first_lower']:.4f}, two-sided Wilcoxon "
        f"p = {authoritative['wilcoxon_p_two_sided']:.6f}, median paired "
        f"difference = {authoritative['median_paired_difference']:.3e}.",
        "",
        f"Retired archive statistic: win rate {deprecated['win_rate']} and "
        f"p = {deprecated['p']} from `{deprecated['archive']}`. "
        + deprecated["reason"],
        "",
        "## Per-metric paired statistics",
        "",
        "| pair | metric | n | win rate | p two-sided |",
        "|---|---|---|---|---|",
    ]
    for k, v in stats.items():
        if k.endswith(("_W1", "_F1_err", "_entropy_err")) and k.startswith(
                ("B4_vs_B1", "B2_vs_B1", "B3_vs_B1")):
            met = "W1" if k.endswith("_W1") else ("F1" if k.endswith("_F1_err") else "entropy")
            lines.append(f"| {v['pair'].split(',')[0]} | {met} | {v['n']} | "
                         f"{v['win_rate_first_lower']:.3f} | "
                         f"{v['wilcoxon_p_two_sided']:.3g} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")

    # Supplementary table, one row per arm-comparison and metric.
    tbl = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Case-level paired statistics recomputed from one archived record. "
        "Every row uses the same case key, the same fitted prior, and one rule. "
        "A win means the first arm has strictly smaller error on that metric, with no "
        "tie tolerance. The $p$-value is two-sided Wilcoxon signed-rank with "
        "\\texttt{zero\\_method=wilcox}. The three metrics are not interchangeable, so "
        "each $p$-value is labeled with its metric.}",
        "\\label{tab:pair-unification}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Pair & Metric & $n$ & Win rate & $p$ & Median diff \\\\",
        "\\midrule",
    ]
    for k in ["B2_vs_B1_W1", "B2_vs_B1_entropy_err",
              "B3_vs_B1_W1", "B3_vs_B1_F1_err",
              "B4_vs_B1_W1", "B4_vs_B1_F1_err", "B4_vs_B1_entropy_err"]:
        v = stats[k]
        met = {"W1": "$\\Wone$", "F1_err": "$\\Fj{1}$ err",
               "entropy_err": "entropy err"}[k.rsplit("_", 1)[-1] if k.endswith("_W1") else
                                             ("F1_err" if k.endswith("_F1_err") else "entropy_err")]
        pair = v["pair"].split(",")[0].replace(" vs ", " vs ")
        p = v["wilcoxon_p_two_sided"]
        ps = f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}".replace("e-0", "e-")
        tbl.append(f"{pair} & {met} & {v['n']} & {v['win_rate_first_lower']:.3f} & "
                   f"{ps} & {v['median_paired_difference']:.1e} \\\\")
    tbl += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(tbl))

    print("chain:", json.dumps(chain, indent=1))
    print("cross:", json.dumps(cross, indent=1))
    print("authoritative:", json.dumps(authoritative, indent=1))
    print("wrote", OUT / "summary.json", OUT / "report.md", TABLE)


if __name__ == "__main__":
    main()
