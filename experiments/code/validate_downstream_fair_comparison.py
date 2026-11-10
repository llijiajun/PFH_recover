"""Reproducible downstream functional comparison for the ICDE revision.

Answers four evaluation questions on the SAME 276 real-column cases and the SAME p:

  * sec. 22: tabulate truth, raw sample S2, every correction, and the relative
    error for at least three cases so the >100% numbers can be recomputed.
  * sec. 23: include the direct factorial-moment estimator, which is
    design-unbiased for sum_v f_v^2 under independent Bernoulli thinning.
  * sec. 21 / 5.3: report medians, tail quantiles, win rates and intervals on
    BOTH the full 276 cases and the 208-case truncated-support subset, with a
    cluster bootstrap at the column level because cases are repeated samples
    of a handful of columns.
  * sec. 20: a factor-control table separating prior family, prior parameters,
    unseen-type compensation, truncation and spectrum readout.

Inputs are the archived per-case CSVs produced by validate_modern_baselines.py
and validate_optimizer_readout.py. No experiment is re-simulated here. Every
number written to summary.json is computed from those per-case rows.

Run with: python3 \
    experiments/code/validate_downstream_fair_comparison.py
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binom, wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MOD_CSV = os.path.join(ROOT, "experiments", "results", "modern_baselines", "modern_metrics.csv")
QER_CSV = os.path.join(ROOT, "experiments", "results", "optimizer_readout", "qerror_metrics.csv")
CLAIMS = os.path.join(ROOT, "experiments", "results", "modern_baselines", "claims_verification.json")
OUT_DIR = os.path.join(ROOT, "experiments", "results", "downstream_fair_comparison")

# Relative error columns, all on the same per-case truth S2_true.
ESTIMATORS = {
    "raw_unscaled": "S2rel_raw_unscaled",
    "rescaled_plugin": "S2rel_rescaled_plugin",
    "factorial_unbiased": "S2rel_factorial_unbiased",
    "sb_linear": "smooth_bayes_func_self_join",
    "ours_fitted": "eb_fit_func_self_join",
    "ours_oracle": "ours_func_self_join",
    "good_turing": "good_turing_func_self_join",
    "npmle": "npmle_func_self_join",
}

# The naive 1/p spectrum readout is a DIFFERENT estimator from the per-value
# rescaled plug-in. Its archived column is kept separate so the two cannot be
# confused again.
NAIVE_SPECTRUM_FUNCTIONAL = "naive_func_self_join"


def rows_from(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def key(r):
    return (r["dataset"], r["attribute"], round(float(r["p"]), 6))


def quantiles(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    return {
        "n": int(a.size),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "mean": float(a.mean()),
    }


def cluster_bootstrap_win_rate(rows, col_a, col_b, n_boot=4000, seed=20261106):
    """Resample COLUMNS, not cases, so repeated rates/seeds stay together."""
    clusters = defaultdict(list)
    for r in rows:
        clusters[(r["dataset"], r["attribute"])].append(r)
    names = sorted(clusters)
    rng = np.random.default_rng(seed)
    wins = []
    for _ in range(n_boot):
        pick = rng.choice(len(names), size=len(names), replace=True)
        num = den = 0
        for idx in pick:
            for r in clusters[names[int(idx)]]:
                a, b = float(r[col_a]), float(r[col_b])
                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                den += 1
                if a < b:
                    num += 1
        if den:
            wins.append(num / den)
    w = np.asarray(wins)
    return {
        "point": float(np.mean([1.0 if float(r[col_a]) < float(r[col_b]) else 0.0
                                for r in rows
                                if math.isfinite(float(r[col_a])) and math.isfinite(float(r[col_b]))])),
        "n_boot": n_boot,
        "ci95_low": float(np.percentile(w, 2.5)),
        "ci95_high": float(np.percentile(w, 97.5)),
        "cluster_unit": "(dataset, attribute)",
    }


def wilcoxon_two_sided(rows, col_a, col_b):
    a, b = [], []
    for r in rows:
        x, y = float(r[col_a]), float(r[col_b])
        if math.isfinite(x) and math.isfinite(y):
            a.append(x)
            b.append(y)
    stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {"n_pairs": len(a), "statistic": float(stat), "p_two_sided": float(p)}


def verify_factorial_unbiased(f=7, p=0.2):
    """E[g(g-1)/p^2 + g/p] over g ~ Bin(f,p) equals f^2 exactly."""
    g = np.arange(0, f + 1)
    probs = binom.pmf(g, f, p)
    est = g * (g - 1) / (p ** 2) + g / p
    expectation = float(np.dot(probs, est))
    return {
        "f": f, "p": p,
        "analytic_mean": expectation,
        "truth_f_squared": float(f ** 2),
        "abs_diff": abs(expectation - f ** 2),
    }


def three_case_recomputation(rows):
    """Pick three cases with different character and recompute every term."""
    wanted = [
        ("Airlines", "CRSArrTime.csv", 0.1),
        ("NCVoter", "mail_city.csv", 0.3),
        ("kasandr", "userid.csv", 0.5),
    ]
    out = []
    for ds, attr, p in wanted:
        hit = [r for r in rows if r["dataset"] == ds and r["attribute"] == attr
               and abs(float(r["p"]) - p) < 1e-9]
        if not hit:
            # fall back to the first case of that dataset
            hit = [r for r in rows if r["dataset"] == ds]
        r = hit[0]
        truth = float(r["S2_true"])
        raw = float(r["S2_raw_unscaled"])
        resc = float(r["S2_rescaled_plugin"])
        fact = float(r["S2_factorial_unbiased"])
        pp = float(r["p"])
        out.append({
            "dataset": r["dataset"], "attribute": r["attribute"], "p": pp,
            "S2_true": truth,
            "S2_raw_unscaled": raw,
            "S2_rescaled_plugin": resc,
            "rescaled_identity_check": abs(resc - raw / pp ** 2),
            "S2_factorial_unbiased": fact,
            "relerr_raw_unscaled": abs(raw - truth) / truth,
            "relerr_rescaled_plugin": abs(resc - truth) / truth,
            "relerr_factorial_unbiased": abs(fact - truth) / truth,
            "relerr_ours_fitted": float(r["eb_fit_func_self_join"]),
            "relerr_naive_spectrum_readout": float(r[NAIVE_SPECTRUM_FUNCTIONAL]),
        })
    return out


def write_latex_tables(result, root):
    """Emit the three table fragments directly from the computed result."""
    tdir = os.path.join(root, "experiments", "results", "tables")
    os.makedirs(tdir, exist_ok=True)

    def esc(s):
        return str(s).replace("_", "\\_").replace("%", "\\%")

    def enotation(x):
        if x == 0:
            return "0"
        e = int(math.floor(math.log10(abs(x))))
        m = x / (10 ** e)
        return f"{m:.2f}E{e:+d}"

    # 1. Three-case term-by-term recomputation.
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Term-by-term recomputation of the self-join size on three cases, one sparse, "
        r"one intermediate, and one heavy tailed. Columns three to six are absolute values of "
        r"$\sum_v f_v^2$ and its estimates. The last five are relative errors. The rescaled "
        r"column obeys $\sum_v(g_v/p)^2=\sum_v g_v^2/p^2$ to numerical precision.}",
        r"\label{tab:downstream-recount}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Case & $p$ & truth & raw & rescaled & factorial "
        r"& \multicolumn{5}{c}{relative error} \\",
        r"\cmidrule(lr){7-11}",
        r" & & & & & & raw & rescaled & factorial & naive $1/p$ & ours fitted \\",
        r"\midrule",
    ]
    for c in result["three_case_recomputation"]:
        label = esc(c["dataset"] + "." + c["attribute"].replace(".csv", ""))
        vals = [c["S2_true"], c["S2_raw_unscaled"], c["S2_rescaled_plugin"],
                c["S2_factorial_unbiased"]]
        rels = [c["relerr_raw_unscaled"], c["relerr_rescaled_plugin"],
                c["relerr_factorial_unbiased"], c["relerr_naive_spectrum_readout"],
                c["relerr_ours_fitted"]]
        cells = [r"\texttt{" + label + "}", f"${c['p']:.2f}$"] + [enotation(v) for v in vals] \
                + [enotation(v) for v in rels]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    with open(os.path.join(tdir, "tab_downstream_recount.tex"), "w") as fh:
        fh.write("\n".join(lines))

    # 2. Full-case versus subset quantiles.
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Median and tail quantiles of self-join relative error on all $276$ cases and "
        r"on the $208$ cases whose truth is unchanged by the $j\le256$ support. The screening "
        r"is defined by the truth, so both columns are reported.}",
        r"\label{tab:downstream-quantiles}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Estimator & \multicolumn{2}{c}{all $276$} & \multicolumn{2}{c}{subset $208$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & med & p90 & med & p90 \\",
        r"\midrule",
    ]
    readable = {
        "raw_unscaled": r"Unscaled $\sum_v g_v^2$",
        "rescaled_plugin": r"Rescaled $\sum_v (g_v/p)^2$",
        "factorial_unbiased": r"Factorial-moment unbiased",
        "naive_spectrum_readout": r"Naive $1/p$ spectrum",
        "sb_linear": r"SB linear",
        "ours_fitted": r"Ours (fitted)",
        "ours_oracle": r"Ours (oracle)",
    }
    for name in ("raw_unscaled", "rescaled_plugin", "factorial_unbiased",
                 "naive_spectrum_readout", "sb_linear", "ours_fitted", "ours_oracle"):
        a = result["full_276"][name]
        b = result["subset_208"][name]
        lines.append(f"{readable[name]} & {a['median']:.3f} & {a['p90']:.3f} "
                     f"& {b['median']:.3f} & {b['p90']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(tdir, "tab_downstream_quantiles.tex"), "w") as fh:
        fh.write("\n".join(lines))

    # 3. Factor control for the shared-prior comparison.
    fc = result["factor_control"]
    sb = fc["published_spectrum_rows"]["SB linear"]
    ou = fc["published_spectrum_rows"]["Ours (fitted)"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Factor control for the two spectrum rows of the main-text baseline table. "
        r"Only the prior family and the truncation support are held fixed. Feeding the "
        r"smoothed-Bayesian estimator the same fitted parameters reproduces our spectrum to "
        f"$2.1\\times10^{{-6}}$ over "
        f"${fc['shared_prior_parameters_agreement']['n_pairs']}$ pairs, so the published median "
        r"gap is a prior-fit and unseen-compensation gap rather than an inversion gap.}",
        r"\label{tab:factor-control}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{p{0.13\textwidth}p{0.38\textwidth}p{0.38\textwidth}}",
        r"\toprule",
        r"Factor & SB linear & Ours (fitted) \\",
        r"\midrule",
    ]
    def factor_tex(s):
        replacements = (
            ("Gamma-Poisson", r"Gamma--Poisson"),
            ("(a0_sb, b0_sb)", r"$(a_0,b_0)$"),
            ("V0 = D w/(1-w)", r"$V_0 = D w/(1-w)$"),
            ("j<=512", r"$j\le512$"),
        )
        for a, b in replacements:
            s = s.replace(a, b)
        return s

    for key, label in (("prior_family", "Prior family"),
                       ("prior_params", "Prior parameters"),
                       ("unseen_compensation", "Unseen-type compensation"),
                       ("truncation", "Truncation support"),
                       ("readout", "Spectrum readout")):
        lines.append(f"{label} & {factor_tex(sb[key])} & {factor_tex(ou[key])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    with open(os.path.join(tdir, "tab_factor_control.tex"), "w") as fh:
        fh.write("\n".join(lines))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mod = rows_from(MOD_CSV)
    qer = rows_from(QER_CSV)

    # 208-case subset: truncation at j<=256 does not change the truth.
    subset_keys = {
        key(r) for r in qer
        if abs(float(r["self_join_true_capped"]) - float(r["self_join_true_uncapped"])) < 1e-9
    }
    mod_keys = {key(r) for r in mod}
    assert subset_keys <= mod_keys, "subset keys not aligned with modern CSV"
    subset = [r for r in mod if key(r) in subset_keys]
    excluded = [r for r in mod if key(r) not in subset_keys]

    result = {
        "source": {
            "modern_metrics_csv": os.path.relpath(MOD_CSV, ROOT),
            "qerror_metrics_csv": os.path.relpath(QER_CSV, ROOT),
            "claims_verification_json": os.path.relpath(CLAIMS, ROOT),
        },
        "n_cases_full": len(mod),
        "n_cases_subset": len(subset),
        "n_cases_excluded": len(excluded),
        "subset_definition": "self_join_true_capped == self_join_true_uncapped (j<=256 support)",
        "excluded_by_dataset": dict(sorted(
            defaultdict(int, {d: sum(1 for r in excluded if r["dataset"] == d)
                              for d in {r["dataset"] for r in excluded}}).items())),
        "full_276": {},
        "subset_208": {},
        "win_rates_vs_ours_fitted": {},
        "wilcoxon": {},
        "spectrum_win_rates": {},
        "spectrum_wilcoxon": {},
        "per_dataset": {},
        "three_case_recomputation": three_case_recomputation(mod),
        "factorial_unbiased_enumeration": verify_factorial_unbiased(),
        "estimator_definitions": {
            "raw_unscaled": "sum_v g_v^2, bounded above by the truth",
            "rescaled_plugin": "sum_v (g_v/p)^2 = sum_v g_v^2 / p^2",
            "factorial_unbiased": "sum_v [g_v(g_v-1)/p^2 + g_v/p]",
            "naive_spectrum_readout": "self-join read off the naive 1/p recovered spectrum, a different estimator",
        },
    }

    for label, data in (("full_276", mod), ("subset_208", subset)):
        result[label] = {
            name: quantiles([float(r[col]) for r in data])
            for name, col in ESTIMATORS.items()
            if all(r.get(col) not in (None, "") for r in data)
        }
        result[label]["naive_spectrum_readout"] = quantiles(
            [float(r[NAIVE_SPECTRUM_FUNCTIONAL]) for r in data]
        )

    ours = "eb_fit_func_self_join"
    for name, col in ESTIMATORS.items():
        if col == ours:
            continue
        usable = [r for r in mod if r.get(col) not in (None, "")]
        result["win_rates_vs_ours_fitted"][name] = {
            "ours_fitted_win_rate": cluster_bootstrap_win_rate(usable, ours, col),
            "n_cases": len(usable),
        }
        result["wilcoxon"][f"ours_fitted_vs_{name}"] = wilcoxon_two_sided(usable, ours, col)

    # The same cluster treatment for the spectrum-distance claims of sec. 21.
    # eb_fit_w1 is our fitted posterior, smooth_bayes_w1 the strongest competitor,
    # good_turing_w1 the classical reference.
    for name, col in (("sb_linear", "smooth_bayes_w1"),
                      ("good_turing", "good_turing_w1"),
                      ("naive_rescaled", "naive_w1")):
        usable = [r for r in mod if r.get(col) not in (None, "")
                  and r.get("eb_fit_w1") not in (None, "")]
        result["spectrum_win_rates"][name] = cluster_bootstrap_win_rate(
            usable, "eb_fit_w1", col
        )
        result["spectrum_wilcoxon"][f"ours_fitted_vs_{name}"] = wilcoxon_two_sided(
            usable, "eb_fit_w1", col
        )

    # Stratified by dataset, with column-level clusters preserved.
    for ds in sorted({r["dataset"] for r in mod}):
        block = [r for r in mod if r["dataset"] == ds]
        result["per_dataset"][ds] = {
            "n_cases": len(block),
            "estimator_medians": {
                name: float(np.median([float(r[col]) for r in block]))
                for name, col in ESTIMATORS.items()
                if all(r.get(col) not in (None, "") for r in block)
            },
            "ours_fitted_win_vs_factorial_unbiased": cluster_bootstrap_win_rate(
                block, ours, "S2rel_factorial_unbiased"
            ),
            "ours_fitted_win_vs_sb_linear": cluster_bootstrap_win_rate(
                block, ours, "smooth_bayes_func_self_join"
            ),
        }

    # Factor control. The 2.1e-6 agreement is the archived
    # shared-parameter check; the remaining rows state which factors the
    # published table does and does not hold fixed.
    with open(CLAIMS) as fh:
        claims = json.load(fh)
    result["factor_control"] = {
        "shared_prior_parameters_agreement": claims["sb_equals_ours_fitted_ndv"],
        "published_spectrum_rows": {
            "SB linear": {
                "prior_family": "Gamma-Poisson",
                "prior_params": "fitted by smoothed-Bayes moment match (a0_sb, b0_sb)",
                "unseen_compensation": "none beyond the fitted prior",
                "truncation": "j<=512 renormalized",
                "readout": "affine posterior mean, re-bucketed",
                "w1_median": claims["spectrum_w1"]["smooth_bayes"],
            },
            "Ours (fitted)": {
                "prior_family": "Gamma-Poisson",
                "prior_params": "fitted by zero-truncation-aware empirical Bayes",
                "unseen_compensation": "V0 = D w/(1-w) zero-truncated term",
                "truncation": "j<=512 renormalized",
                "readout": "posterior-mean spectrum",
                "w1_median": claims["spectrum_w1"]["eb_fit"],
            },
        },
        "interpretation": (
            "Same prior family, different fitted parameters and different unseen-type "
            "compensation. Feeding SB linear the SAME fitted parameters reproduces our "
            "spectrum to the archived max_abs_diff, so the published median gap is a "
            "prior-fit and unseen-compensation gap, not an inversion gap."
        ),
    }

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)

    # Human-readable report.
    lines = ["# Downstream fair comparison (ICDE revision)", ""]
    lines.append(f"Cases: {len(mod)} full, {len(subset)} truncated-support subset, "
                 f"{len(excluded)} excluded." )
    lines.append("")
    lines.append("## Estimator definitions")
    for k, v in result["estimator_definitions"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    for label in ("full_276", "subset_208"):
        lines.append(f"## {label} (relative error, median / p25 / p75 / p90)")
        lines.append("")
        lines.append("| estimator | n | p25 | median | p75 | p90 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for name, q in result[label].items():
            lines.append(f"| {name} | {q['n']} | {q['p25']:.4f} | {q['median']:.4f} "
                         f"| {q['p75']:.4f} | {q['p90']:.4f} |")
        lines.append("")
    lines.append("## Three-case recomputation")
    lines.append("")
    for c in result["three_case_recomputation"]:
        lines.append(f"- {c['dataset']}.{c['attribute']} p={c['p']}: "
                     f"truth={c['S2_true']:.0f}, raw={c['S2_raw_unscaled']:.0f}, "
                     f"rescaled={c['S2_rescaled_plugin']:.0f}, "
                     f"factorial={c['S2_factorial_unbiased']:.0f}; "
                     f"relerr raw={c['relerr_raw_unscaled']:.3f}, "
                     f"rescaled={c['relerr_rescaled_plugin']:.3f}, "
                     f"factorial={c['relerr_factorial_unbiased']:.3f}, "
                     f"ours={c['relerr_ours_fitted']:.3f}, "
                     f"naive spectrum={c['relerr_naive_spectrum_readout']:.3f}")
        lines.append(f"  - rescaled identity check |S2_rescaled - S2_raw/p^2| = "
                     f"{c['rescaled_identity_check']:.3e}")
    lines.append("")
    u = result["factorial_unbiased_enumeration"]
    lines.append(f"## Unbiasedness check: f={u['f']}, p={u['p']}: "
                 f"E[estimator]={u['analytic_mean']:.6f} vs f^2={u['truth_f_squared']:.0f}, "
                 f"diff={u['abs_diff']:.3e}")
    lines.append("")
    lines.append("## Win rates of ours (fitted) with column-cluster bootstrap 95% CI")
    lines.append("")
    lines.append("| competitor | ours win rate | CI low | CI high | Wilcoxon p (two-sided) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, d in result["win_rates_vs_ours_fitted"].items():
        w = d["ours_fitted_win_rate"]
        p = result["wilcoxon"][f"ours_fitted_vs_{name}"]["p_two_sided"]
        lines.append(f"| {name} | {w['point']:.3f} | {w['ci95_low']:.3f} | "
                     f"{w['ci95_high']:.3f} | {p:.4g} |")
    lines.append("")
    lines.append("## Spectrum W1 win rates with column-cluster bootstrap 95% CI")
    lines.append("")
    lines.append("| competitor | ours fitted W1 win rate | CI low | CI high | Wilcoxon p |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, w in result["spectrum_win_rates"].items():
        p = result["spectrum_wilcoxon"][f"ours_fitted_vs_{name}"]["p_two_sided"]
        lines.append(f"| {name} | {w['point']:.3f} | {w['ci95_low']:.3f} | "
                     f"{w['ci95_high']:.3f} | {p:.4g} |")
    lines.append("")
    lines.append("## Per-dataset medians and win rates")
    lines.append("")
    for ds, d in result["per_dataset"].items():
        med = d["estimator_medians"]
        wf = d["ours_fitted_win_vs_factorial_unbiased"]
        ws = d["ours_fitted_win_vs_sb_linear"]
        lines.append(f"- {ds} (n={d['n_cases']}): factorial median "
                     f"{med.get('factorial_unbiased', float('nan')):.4f}, "
                     f"ours {med.get('ours_fitted', float('nan')):.4f}, "
                     f"SB {med.get('sb_linear', float('nan')):.4f}; "
                     f"ours wins vs factorial {wf['point']:.3f} "
                     f"[{wf['ci95_low']:.3f},{wf['ci95_high']:.3f}], "
                     f"vs SB {ws['point']:.3f} "
                     f"[{ws['ci95_low']:.3f},{ws['ci95_high']:.3f}]")
    lines.append("")
    lines.append("## Factor control")
    lines.append("")
    lines.append(json.dumps(result["factor_control"], indent=1, ensure_ascii=False))
    lines.append("")
    with open(os.path.join(OUT_DIR, "report.md"), "w") as fh:
        fh.write("\n".join(lines))

    write_latex_tables(result, ROOT)
    print(f"wrote {OUT_DIR}/summary.json and report.md")
    print("wrote 3 tables in experiments/results/tables/")
    print(f"full={len(mod)} subset={len(subset)} excluded={len(excluded)}")
    print("median relerr full:")
    for name, q in result["full_276"].items():
        print(f"  {name:24s} {q['median']:.4f}")


if __name__ == "__main__":
    main()
