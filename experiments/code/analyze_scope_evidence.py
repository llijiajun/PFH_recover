"""Emit two diagnostic tables on support size and interval objects.

The first table reports the support-size and row-mass summary from the archived
case-level CSV, with the claim narrowed to the truncated normalized spectrum.
The second table separates the interval objects, their nominal levels, their
width units, and the uncertainty they propagate. The definitions below are
transcribed from the method and the band-calibration archive.

Outputs
  experiments/results/tables/tab_support_summary.tex
  experiments/results/tables/tab_coverage_objects.tex
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
T = ROOT / "experiments" / "results" / "tables"
T.mkdir(parents=True, exist_ok=True)
MODERN = RES / "modern_baselines" / "modern_metrics.csv"
BAND = RES / "band_calibration" / "summary.json"


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def med(vals):
    v = sorted(float(x) for x in vals if x not in ("", "nan", "None") and x is not None)
    if not v:
        return 0.0
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def main():
    M = rows(MODERN)
    rates = sorted({float(r["p"]) for r in M})
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Support-size and row-mass recovery on the real-column cases, "
        "median relative error by sampling rate. $\\widehat V$ is the estimated number "
        "of population types and $\\widehat N=\\sum_j j\\widehat F_j$ the estimated "
        "row count. These are separate from the normalized shape metrics of the main "
        "text, which cannot detect an error in $\\widehat V$. The mass carried by "
        "population types above $j=512$ is not measured separately, so the recovery "
        "claim is limited to the truncated normalized spectrum.}",
        "\\label{tab:support-summary}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Rate & fitted $\\widehat V$ & fitted $\\widehat N$ & raw $\\widehat V$ & raw $\\widehat N$ \\\\",
        "\\midrule",
    ]
    for p in rates + [None]:
        sub = M if p is None else [r for r in M if float(r["p"]) == p]
        vals = [med([r[c] for r in sub]) for c in
                ("eb_fit_ndv_rel", "eb_fit_pop_rel", "naive_ndv_rel", "naive_pop_rel")]
        label = "all" if p is None else f"{p:g}"
        lines.append(f"{label} & " + " & ".join(f"{v:.3f}" for v in vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (T / "tab_support_summary.tex").write_text("\n".join(lines))

    band = json.loads(BAND.read_text())
    k = band["K120"]
    cov = k["coverage_range"]
    obj = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Interval objects and what each one covers. "
        "The main deployed output is the conditional per-frequency credible band. "
        "Its width is a number of types at a fixed frequency $j$, not a number of "
        "frequency bins. The other rows are diagnostics on different objects and "
        "do not validate the deployed band.}",
        "\\label{tab:coverage-objects}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{p{0.28\\linewidth}p{0.62\\linewidth}}",
        "\\toprule",
        "Object & Definition, level, and evidence boundary \\\\",
        "\\midrule",
        "90\\% per-frequency band & Componentwise posterior percentiles of "
        "$F_j$ at fixed $j$ from $K=120$ draws. Empirical coverage "
        f"{cov[0]:.3f}--{cov[1]:.3f} over three seeds, median width one type. "
        "Conditional on the fitted prior. Not a simultaneous region. \\\\",
        "95\\% per-type count interval & Interval for the count $f_v$ of one type. "
        "Empirical coverage $0.58$--$0.74$ for types seen three or more times. "
        "Width is in count units. \\\\",
        "Table II coverage column & Empirical per-type count-interval coverage of the "
        "learned thickening variant on the synthetic grid, $0.81$--$0.83$. "
        "Distinct variant and distinct object. \\\\",
        "Split-conformal band & Calibrated on held-out synthetic populations in "
        "Algorithm~2. It belongs to the learned variant, not the analytic inverse. \\\\",
        "Model content $0.953$ & Average posterior probability the fitted model assigns "
        "to its own nominal $95\\%$ per-type intervals. A model-internal diagnostic. "
        "It is not an empirical coverage value. \\\\",
        "\\midrule",
        "Propagated randomness & Posterior count draws and the Poissonized unseen-type "
        "count, conditional on fitted $\\widehat\\nu,a_0,b_0$. \\\\",
        "Not propagated & Prior parameter estimation error, prior misspecification, "
        "and repeated refitting across thinnings. The prior is fitted once per case. \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    (T / "tab_coverage_objects.tex").write_text("\n".join(obj))
    print("wrote", T / "tab_support_summary.tex")
    print("wrote", T / "tab_coverage_objects.tex")
    print("coverage range", cov)


if __name__ == "__main__":
    main()
