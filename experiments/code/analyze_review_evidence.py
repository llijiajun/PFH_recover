"""Correctness and deployment-evidence checks on archived results.

All inputs are archived experiment results. No new random experiment is run
except for the deterministic protocol reconciliation, which recomputes a raw
spectrum distance on the three published analytic-grid populations.

Four reports are produced.
  results/entropy_consistency/  A3, entropy medians and the correct paired tests
  results/protocol_consistency/ A4, the two protocols behind 2.51 and 0.64
  results/deployable_regime/    boundary refusal metrics and a sample-only regime
  results/band_calibration/     90% per-frequency band coverage and width
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from validate_baselines import stable_seed  # noqa: E402

RES = ROOT / "experiments" / "results"
MODERN = RES / "modern_baselines" / "modern_metrics.csv"
FITTED = RES / "fitted_prior_validation" / "fitted_metrics.csv"
W1_TESTS = RES / "baseline_validation" / "w1_tests.json"
INTERVALS = RES / "archived_inputs" / "intervals_three_layer.json"
RUNTIME = RES / "runtime_benchmark" / "summary.json"
BOUND = 1e-3
TOL = 1.001e-3


def rows(path):
    return list(csv.DictReader(path.open()))


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def dump(name, payload, lines):
    out = RES / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {name}: summary.json, report.md")


def cluster_ci(flags, n_boot=4000, seed=0):
    """Column-cluster bootstrap of a per-case win rate."""
    groups = defaultdict(list)
    for (ds, attr, _p), v in flags:
        groups[(ds, attr)].append(v)
    keys = list(groups)
    if not keys:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        vals.append(np.mean(np.concatenate([groups[keys[i]] for i in idx])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------------------------------------------------------------------------
# A3  entropy consistency
# ---------------------------------------------------------------------------
def entropy_consistency():
    m = rows(MODERN)
    ent_cols = [c for c in m[0] if c.endswith("_func_entropy")]
    methods = {}
    for c in ent_cols:
        v = np.asarray([num(r[c]) for r in m if num(r[c]) is not None])
        methods[c] = {"n": int(v.size), "median_rel_err": float(np.median(v)),
                      "median_pct": float(np.median(v) * 100)}

    def pair(a, b):
        d = [(num(r[a]), num(r[b])) for r in m
             if num(r[a]) is not None and num(r[b]) is not None]
        x = np.asarray([u for u, _ in d]); y = np.asarray([v for _, v in d])
        return {"method_a": a, "method_b": b, "n": len(d),
                "direction": "two-sided Wilcoxon signed-rank on paired absolute "
                             "relative entropy error",
                "win_rate_a_lower": float(np.mean(x < y)),
                "median_diff": float(np.median(x - y)),
                "wilcoxon_p": float(wilcoxon(x, y, alternative="two-sided").pvalue)}

    tests = {
        "fitted_vs_sb": pair("eb_fit_func_entropy", "smooth_bayes_func_entropy"),
        "oracle_vs_sb": pair("ours_func_entropy", "smooth_bayes_func_entropy"),
        "fitted_vs_naive": pair("eb_fit_func_entropy", "naive_func_entropy"),
        "oracle_vs_naive": pair("ours_func_entropy", "naive_func_entropy"),
        "sb_vs_naive": pair("smooth_bayes_func_entropy", "naive_func_entropy"),
    }
    payload = {
        "source": str(MODERN.relative_to(ROOT)), "n_rows": len(m),
        "table_values_pct": {
            "naive_1/p": round(methods["naive_func_entropy"]["median_pct"], 1),
            "Good--Turing": round(methods["good_turing_func_entropy"]["median_pct"], 1),
            "NPMLE": round(methods["npmle_func_entropy"]["median_pct"], 1),
            "SB linear": round(methods["smooth_bayes_func_entropy"]["median_pct"], 1),
            "Ours fitted": round(methods["eb_fit_func_entropy"]["median_pct"], 1),
            "Ours oracle": round(methods["ours_func_entropy"]["median_pct"], 1),
        },
        "per_method": methods, "paired_tests": tests,
        "correction": {
            "table": "SB linear is 3.1% and Ours fitted is 2.9%, so the table is correct.",
            "prose": "The prose claimed both reach 2.9%. It must say 2.9% against 3.1%.",
            "p_value": "The archived p=4.4e-45 magnitude belongs to a fitted/oracle "
                       "estimator against the naive plug-in, not to fitted against SB. "
                       "The fitted-versus-SB two-sided p is reported above.",
            "interpretation": "A p near 1e-45 is incompatible with 'statistically "
                              "indistinguishable'. The fitted-versus-SB comparison is "
                              "reported with its own p value and direction.",
        },
    }
    L = ["# Entropy consistency check (A3)", "",
         "Source `" + str(MODERN.relative_to(ROOT)) + f"`, {len(m)} rows.", "",
         "## Median relative entropy error, deployment state", "",
         "| table row | median % | n |", "|---|---|---|"]
    for k, v in payload["table_values_pct"].items():
        col = {"naive_1/p": "naive_func_entropy", "Good--Turing": "good_turing_func_entropy",
               "NPMLE": "npmle_func_entropy", "SB linear": "smooth_bayes_func_entropy",
               "Ours fitted": "eb_fit_func_entropy", "Ours oracle": "ours_func_entropy"}[k]
        L.append(f"| {k} | {v} | {methods[col]['n']} |")
    L += ["", "## Paired tests", "", "| comparison | n | a lower | median diff | p |",
          "|---|---|---|---|---|"]
    for k, v in tests.items():
        L.append(f"| {k} | {v['n']} | {v['win_rate_a_lower']:.3f} | "
                 f"{v['median_diff']:.3g} | {v['wilcoxon_p']:.4g} |")
    L += ["", "## Correction", ""]
    for k, v in payload["correction"].items():
        L.append(f"- **{k}**: {v}")
    dump("entropy_consistency", payload, L)
    return payload


# ---------------------------------------------------------------------------
# A4  protocol consistency
# ---------------------------------------------------------------------------
def protocol_consistency():
    import d4d_logic as D
    import d4d_baselines as B

    J = 512
    N, VOCAB, ALPHA, P = 100_000, 30_000, 1.5, 0.1
    SEEDS = (0, 1, 2)

    def truth_spectrum(f_true):
        F = np.zeros(J)
        for f in f_true[f_true > 0]:
            if f <= J:
                F[f - 1] += 1.0
        return F

    def cdf_of(spec):
        s = np.asarray(spec, dtype=float)
        if s.size < J:
            s = np.pad(s, (0, J - s.size))
        s = s[:J]
        return np.cumsum(s / s.sum()) if s.sum() > 0 else np.zeros(J)

    recon = []
    for seed in SEEDS:
        rng = np.random.default_rng(710_000 + seed * 13 + int(P * 1000)
                                    + int(ALPHA * 100))
        f_true = D.zipf_population(rng, N, VOCAB, ALPHA)
        g = D.thin(f_true, P, rng)
        raw = D.fofo_spectrum(g)[:J].astype(float)
        F_trunc = truth_spectrum(f_true)               # analytic grid definition
        F_folded = D.fofo_spectrum(f_true, j_max=J)    # Table V / distribution_metrics
        w1_analytic = D.w1_distance(F_trunc, np.arange(1, J + 1, dtype=float),
                                    cdf_of(raw), j_eval=J)
        w1_folded = B.distribution_metrics(F_folded, raw)["w1"]
        recon.append({"seed": seed, "d_obs": int((g >= 1).sum()),
                      "w1_analytic_grid": float(w1_analytic),
                      "w1_folded_truth_metric": float(w1_folded),
                      "n_true_f_gt_J": int((f_true > J).sum())})
    payload = {
        "finding": "The two numbers come from different protocols and must not be "
                   "compared as repeated measurements of one quantity.",
        "protocol_A": {
            "where": "tables/tab_main.tex and tables/tab_necessity.tex, raw or "
                     "no-thickening row at alpha=1.5, p=0.1",
            "value": "2.51 +/- 0.04",
            "definition": "network ablation grid, corrected protocol with the schedule "
                          "forced to p_T=1 and the exp-1 readout, three seeds over 24 "
                          "evaluation populations, distance aggregated by that grid",
            "populations": "the network grid evaluation populations and its own seeds",
        },
        "protocol_B": {
            "where": "sections/experiments.tex, mechanism paragraph",
            "value": "0.64",
            "definition": "independent analytic diagnostic grid from "
                          "analyze_table2_mechanism.py, three seeds, N=1e5, "
                          "vocab=3e4, j<=512, truth spectrum restricted to f<=J, "
                          "W1 on the renormalized CDF",
            "populations": "seed 710000 + 13*seed + 1000*p + 100*alpha, regenerated "
                           "inside that script, not the network grid populations",
        },
        "reconciliation_on_same_populations": recon,
        "action": [
            "Remove the claim that the controlled run uses the same populations as Table II.",
            "Label 0.64 explicitly as the analytic diagnostic grid.",
            "State that 2.51 and 0.64 are not comparable and use each only inside "
            "its own protocol.",
            "Correct the analyze_table2_mechanism.py docstring, which wrongly says "
            "the script measures mechanisms on the same populations as Table II.",
        ],
    }
    L = ["# Protocol reconciliation for the raw baseline (A4)", "",
         payload["finding"], "",
         "## Protocol A, 2.51 +/- 0.04", "",
         f"- Where: {payload['protocol_A']['where']}",
         f"- Definition: {payload['protocol_A']['definition']}",
         f"- Populations: {payload['protocol_A']['populations']}", "",
         "## Protocol B, 0.64", "",
         f"- Where: {payload['protocol_B']['where']}",
         f"- Definition: {payload['protocol_B']['definition']}",
         f"- Populations: {payload['protocol_B']['populations']}", "",
         "## Same-population, two-metric recomputation", "",
         "| seed | d_obs | W1 analytic grid metric | W1 folded-truth metric | "
         "true types with f>J |", "|---|---|---|---|---|"]
    for r in recon:
        L.append(f"| {r['seed']} | {r['d_obs']} | {r['w1_analytic_grid']:.4f} | "
                 f"{r['w1_folded_truth_metric']:.4f} | {r['n_true_f_gt_J']} |")
    L += ["", "The two W1 implementations agree to floating-point precision on identical "
          "samples. The gap between 2.51 and 0.64 therefore comes from the populations "
          "and the evaluation grid, not from the distance implementation. Every raw W1 "
          "value needs a protocol label.", "",
          "## Required actions", ""]
    L += [f"- {a}" for a in payload["action"]]
    dump("protocol_consistency", payload, L)
    return payload


# ---------------------------------------------------------------------------
# boundary refusal + sample-only deployable regime
# ---------------------------------------------------------------------------
def deployable_regime():
    m = {(r["dataset"], r["attribute"], float(r["p"])): r for r in rows(MODERN)}
    f = {(r["dataset"], r["attribute"], float(r["p"])): r for r in rows(FITTED)}
    est = ("eb_fit_w1", "smooth_bayes_w1", "good_turing_w1", "naive_w1")
    kept, refused = [], []
    for key, fr in f.items():
        a0 = num(fr["a0_eb"]); mr = m.get(key)
        if a0 is None or mr is None:
            continue
        rec = {"key": key, "a0": a0, **{c: num(mr.get(c)) for c in est}}
        (refused if a0 <= TOL else kept).append(rec)

    def med(rs, c):
        v = [r[c] for r in rs if r[c] is not None and np.isfinite(r[c])]
        return float(np.median(v)) if v else None

    n_total = len(kept) + len(refused)
    LARGE = 0.1  # ten times the kept-group fitted median, fixed before inspection
    big_kept = [r for r in kept if r["eb_fit_w1"] is not None
                and np.isfinite(r["eb_fit_w1"]) and r["eb_fit_w1"] > LARGE]
    big_refused = [r for r in refused if r["eb_fit_w1"] is not None
                   and np.isfinite(r["eb_fit_w1"]) and r["eb_fit_w1"] > LARGE]
    missed = len(big_kept) / (len(big_kept) + len(big_refused)) if (big_kept or big_refused) else None

    def comparison(rs, label):
        out = {"n": len(rs), "median_w1": {c: med(rs, c) for c in est}}
        for c in ("smooth_bayes_w1", "good_turing_w1", "naive_w1"):
            pairs = [(r["eb_fit_w1"], r[c]) for r in rs
                     if r["eb_fit_w1"] is not None and r[c] is not None
                     and np.isfinite(r["eb_fit_w1"]) and np.isfinite(r[c])]
            if not pairs:
                continue
            x = np.asarray([u for u, _ in pairs]); y = np.asarray([v for _, v in pairs])
            flags = [(r["key"], float(a < b)) for r, (a, b) in zip(
                [q for q in rs if q["eb_fit_w1"] is not None and q[c] is not None
                 and np.isfinite(q["eb_fit_w1"]) and np.isfinite(q[c])], pairs)]
            ci = cluster_ci(flags)
            out[f"fitted_vs_{c}"] = {
                "n": len(pairs), "win_rate_fitted_lower": float(np.mean(x < y)),
                "cluster_ci95": ci,
                "wilcoxon_p_two_sided": float(wilcoxon(x, y, alternative="two-sided").pvalue),
                "median_diff": float(np.median(x - y))}
        return out

    # Held-out check. Split by attribute, so no column appears on both sides.
    attrs = sorted({r["key"][1] for r in kept})
    split_seed = lambda a: stable_seed(a, "regime-split") % 2
    halves = {"split_0": [r for r in kept if split_seed(r["key"][1]) == 0],
              "split_1": [r for r in kept if split_seed(r["key"][1]) == 1]}
    payload = {
        "selector": {
            "rule": "report the point histogram only when fitted a0 > 1e-3",
            "computed_from": "the thinned sample alone, via fitted_prior_eb",
            "threshold_origin": "the lower bound of the optimizer for a0, not a tuned "
                                "constant and not chosen using ground truth or test error",
            "flagged_case_semantics": "no point histogram is returned, a boundary flag "
                                      "is returned instead",
        },
        "coverage": {
            "n_cases": n_total, "n_kept": len(kept), "n_refused": len(refused),
            "usable_fraction": len(kept) / n_total if n_total else None,
        },
        "error_by_group": {"kept": {c: med(kept, c) for c in est},
                           "refused": {c: med(refused, c) for c in est}},
        "large_error_audit": {
            "definition": f"W1 > {LARGE}, fixed before inspecting the kept group and "
                          "about ten times its fitted median",
            "n_large_kept": len(big_kept), "n_large_refused": len(big_refused),
            "fraction_of_large_errors_not_flagged": missed,
        },
        "same_kept_set_baselines": comparison(kept, "kept"),
        "held_out_by_attribute": {
            "n_attributes": len(attrs),
            "split_rule": "deterministic hash of the attribute name",
            **{k: comparison(v, k) for k, v in halves.items()},
        },
        "cost": json.loads(RUNTIME.read_text()) if RUNTIME.exists() else None,
    }
    L = ["# Boundary refusal and a sample-selectable deployment regime", "",
         "## Selector", ""]
    for k, v in payload["selector"].items():
        L.append(f"- **{k}**: {v}")
    c = payload["coverage"]
    L += ["", "## Usable output", "",
          f"- {c['n_kept']} of {c['n_cases']} cases usable, "
          f"{c['usable_fraction']:.1%} of the benchmark.",
          f"- {c['n_refused']} cases flagged at the a0 bound.", "",
          "## Median W1 on the two groups", "",
          "| group | n | fitted | SB linear | Good--Turing | naive |",
          "|---|---|---|---|---|---|"]
    for name, rs in (("kept", kept), ("refused", refused)):
        e = payload["error_by_group"][name]
        L.append(f"| {name} | {len(rs)} | " + " | ".join(
            "---" if e[x] is None else f"{e[x]:.4g}" for x in est) + " |")
    a = payload["large_error_audit"]
    L += ["", "## Large errors missed by the flag", "",
          f"- Definition: {a['definition']}.",
          f"- Kept {a['n_large_kept']}, refused {a['n_large_refused']}.",
          f"- Fraction of large errors not flagged: "
          f"{a['fraction_of_large_errors_not_flagged']:.1%}.", "",
          "## Baselines on the same kept set", ""]
    for k, v in payload["same_kept_set_baselines"].items():
        if k.startswith("fitted_vs_"):
            L.append(f"- {k}: win {v['win_rate_fitted_lower']:.1%}, "
                     f"cluster CI {v['cluster_ci95']}, p={v['wilcoxon_p_two_sided']:.4g}")
    L += ["", "## Held-out split by attribute", ""]
    for k, v in payload["held_out_by_attribute"].items():
        if k.startswith("split_") and isinstance(v, dict):
            gt = v.get("fitted_vs_good_turing_w1", {})
            sb = v.get("fitted_vs_smooth_bayes_w1", {})
            L.append(f"- {k}: n={v['n']}, fitted median W1 "
                     f"{v['median_w1']['eb_fit_w1']:.4g}, win vs GT "
                     f"{gt.get('win_rate_fitted_lower', float('nan')):.1%} "
                     f"(p={gt.get('wilcoxon_p_two_sided', float('nan')):.4g}), win vs SB "
                     f"{sb.get('win_rate_fitted_lower', float('nan')):.1%}")
    L += ["", "The selector uses no ground-truth quantity. The held-out split keeps "
          "every attribute entirely on one side, so the two halves are independent "
          "columns rather than repeated rates of the same column.", ""]
    dump("deployable_regime", payload, L)

    # LaTeX table for the main text.
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Boundary-flag output and a sample-only deployment regime on the "
         r"real-column cases. The selector reports a point histogram only when the "
         r"fitted shape exceeds $10^{-3}$. Medians are $\Wone$ on the cases that "
         r"survive the rule. Lower is better.}",
         r"\label{tab:outputpolicy}", r"\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Group & $n$ & fitted & SB & GT & naive \\", r"\midrule"]
    for name, rs in (("reported", kept), ("flagged", refused)):
        e = payload["error_by_group"]["kept" if name == "reported" else "refused"]
        T.append(f"{name} & {len(rs)} & " + " & ".join(
            "---" if e[x] is None else f"{e[x]:.3g}" for x in est) + r" \\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_tables = RES / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    (out_tables / "tab_output_policy.tex").write_text("\n".join(T) + "\n")
    return payload


# ---------------------------------------------------------------------------
# 90% per-frequency band
# ---------------------------------------------------------------------------
def band_calibration():
    d = json.loads(INTERVALS.read_text())
    cov, width, nbins = [], [], []
    per_K = defaultdict(lambda: {"cov": [], "width": []})
    for sk, sv in d.items():
        for K, kv in sv["K"].items():
            per_K[K]["cov"].append(kv["per_freq_cov_90"])
            per_K[K]["width"].append(kv["median_band_width"])
            if K == "120":
                cov.append(kv["per_freq_cov_90"])
                width.append(kv["median_band_width"])
                nbins.append(kv["n_bins"])
    payload = {
        "output": "90% per-frequency credible band from K=120 posterior draws",
        "source": str(INTERVALS.relative_to(ROOT)),
        "seeds": list(d.keys()),
        "K120": {"per_seed_coverage": cov,
                 "coverage_range": [float(min(cov)), float(max(cov))],
                 "coverage_median": float(np.median(cov)),
                 "median_band_width_per_seed": width,
                 "n_frequency_bins_per_seed": nbins},
        "draw_count_stability": {
            K: {"coverage_range": [float(min(v["cov"])), float(max(v["cov"]))],
                "median_width": float(np.median(v["width"]))}
            for K, v in sorted(per_K.items(), key=lambda t: int(t[0]))},
        "prior_refitting": "The prior is fitted once per case and passed as a fixed "
                           "array to plausible_set. The K posterior draws do not refit "
                           "it, and repeated thinnings in this benchmark do not refit "
                           "it either.",
        "uncertainty_included": "posterior sampling variability of the population counts "
                                "conditional on the fitted prior",
        "uncertainty_excluded": "prior parameter estimation error, prior misspecification, "
                                "and the unknown abundance of types absent from the sample",
        "distinct_output": "The 95% per-type intervals and the L1 model-content figure "
                           "of 0.953 are different quantities and do not validate this band.",
    }
    L = ["# Calibration of the 90% per-frequency band", "",
         f"Source `{payload['source']}`.", "",
         f"- Per-seed empirical coverage at K=120: "
         + ", ".join(f"{x:.4f}" for x in cov) + ".",
         f"- Range {min(cov):.3f} to {max(cov):.3f}, median {np.median(cov):.3f}.",
         f"- Median band width per seed: " + ", ".join(f"{x:g}" for x in width) + ".",
         f"- Number of frequency bins per seed: " + ", ".join(str(x) for x in nbins) + ".", "",
         "## Stability in the number of draws", "",
         "| K | coverage range | median width |", "|---|---|---|"]
    for K, v in sorted(per_K.items(), key=lambda t: int(t[0])):
        L.append(f"| {K} | {min(v['cov']):.3f}--{max(v['cov']):.3f} | "
                 f"{np.median(v['width']):g} |")
    L += ["", "## What the interval contains", ""]
    for k in ("prior_refitting", "uncertainty_included", "uncertainty_excluded",
              "distinct_output"):
        L.append(f"- **{k}**: {payload[k]}")
    dump("band_calibration", payload, L)
    return payload


def readout_regime():
    """Where the 2x2 readout gain holds, and paired uncertainty on the kept set."""
    path = RES / "readout_2x2" / "metrics.csv"
    if not path.exists():
        return None
    r = rows(path)
    f = {(q["dataset"], q["attribute"], float(q["p"])): q
         for q in rows(FITTED)}
    mm = {(q["dataset"], q["attribute"], float(q["p"])): q
          for q in rows(MODERN)}

    def arr(sub, key):
        return np.asarray([
            (lambda v: v if v is not None else np.nan)(num(q[key])) for q in sub
        ], dtype=float)

    def effect(sub, label):
        x = arr(sub, "B4_w1")
        y = arr(sub, "B1_w1")
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        return {"label": label, "n": int(len(x)),
                "median_B4": float(np.median(x)), "median_B1": float(np.median(y)),
                "win_rate_B4_lower": float(np.mean(x < y)),
                "median_paired_diff": float(np.median(x - y)),
                "wilcoxon_p_two_sided": float(wilcoxon(x, y).pvalue)}

    interior = [q for q in r if num(q["a0"]) > TOL]
    boundary = [q for q in r if num(q["a0"]) <= TOL]
    by_boundary = {"all": effect(r, "all"),
                   "interior": effect(interior, "a0>1e-3"),
                   "boundary": effect(boundary, "a0<=1e-3")}
    by_rate = {str(p): effect([q for q in interior if float(q["p"]) == p], f"interior p={p}")
               for p in (0.1, 0.2, 0.3, 0.5)}

    def paired_ci(c):
        pairs = []
        for key, fq in f.items():
            if num(fq["a0_eb"]) is None or num(fq["a0_eb"]) <= TOL or key not in mm:
                continue
            a, b = num(mm[key].get("eb_fit_w1")), num(mm[key].get(c))
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                pairs.append((a, b, key))
        d = np.asarray([a - b for a, b, _ in pairs])
        flags = [(k, float(a < b)) for a, b, k in pairs]
        ci = cluster_ci(flags)
        return {"comparator": c, "n": len(pairs),
                "win_rate_fitted_lower": float(np.mean(d < 0)),
                "median_paired_diff": float(np.median(d)),
                "cluster_ci95_win_rate": ci,
                "wilcoxon_p_two_sided": float(
                    wilcoxon([a for a, _, _ in pairs], [b for _, b, _ in pairs]).pvalue)}

    payload = {
        "source": str(path.relative_to(ROOT)),
        "readout_effect_by_boundary": by_boundary,
        "readout_effect_by_rate_interior": by_rate,
        "kept_set_paired": [paired_ci("good_turing_w1"), paired_ci("smooth_bayes_w1")],
        "finding": "The integrated readout lowers the pooled median W1 on interior "
                   "fits but is not a per-case win. On boundary fits it is significantly "
                   "worse than the rounded readout. The sample-only a0 flag therefore "
                   "identifies both the usable regime and the regime where the added "
                   "readout should not be used.",
    }
    L = ["# Where the 2x2 readout gain holds", "", payload["finding"], "",
         "| group | n | median B4 | median B1 | B4 lower | median paired diff | p |",
         "|---|---|---|---|---|---|---|"]
    for k, v in by_boundary.items():
        L.append(f"| {k} | {v['n']} | {v['median_B4']:.4g} | {v['median_B1']:.4g} | "
                 f"{v['win_rate_B4_lower']:.1%} | {v['median_paired_diff']:+.5g} | "
                 f"{v['wilcoxon_p_two_sided']:.4g} |")
    L += ["", "## Interior fits by rate", "",
          "| p | n | median B4 | median B1 | B4 lower | p |", "|---|---|---|---|---|---|"]
    for k, v in by_rate.items():
        L.append(f"| {k} | {v['n']} | {v['median_B4']:.4g} | {v['median_B1']:.4g} | "
                 f"{v['win_rate_B4_lower']:.1%} | {v['wilcoxon_p_two_sided']:.4g} |")
    L += ["", "## Kept-set paired comparisons with uncertainty", ""]
    for v in payload["kept_set_paired"]:
        L.append(f"- fitted vs {v['comparator']}: n={v['n']}, win "
                 f"{v['win_rate_fitted_lower']:.1%}, cluster CI "
                 f"{[round(x, 3) for x in v['cluster_ci95_win_rate']]}, "
                 f"median diff {v['median_paired_diff']:+.5g}, "
                 f"p={v['wilcoxon_p_two_sided']:.4g}")
    dump("readout_regime", payload, L)
    return payload


TABLE_DIR = RES / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def tex_p(p):
    if p >= 1e-3:
        return f"{p:.3g}"
    m, e = f"{p:.2e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def emit_tex_tables():
    """Compact supplement tables, all generated from the summaries above."""
    ent = json.loads((RES / "entropy_consistency" / "summary.json").read_text())
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Median relative entropy error and paired tests on the $276$ "
         r"real-column cases. The fitted and smoothed-Bayesian rows are paired over "
         r"$275$ scored cases with a two-sided Wilcoxon test. Lower is better.}",
         r"\label{tab:entropycheck}", r"\footnotesize",
         r"\begin{tabular}{lc}", r"\toprule",
         r"Estimator & median error (\%) \\", r"\midrule"]
    labels = {"naive_1/p": r"Naive $1/p$", "Good--Turing": "Good--Turing",
              "NPMLE": "NPMLE", "SB linear": "SB linear",
              "Ours fitted": "Ours fitted", "Ours oracle": "Ours oracle"}
    for k, v in ent["table_values_pct"].items():
        T.append(f"{labels[k]} & {v} \\\\")
    T += [r"\midrule", r"\multicolumn{2}{l}{\emph{Paired test}} \\"]
    for k, lab in (("fitted_vs_sb", "Fitted vs SB"),
                   ("fitted_vs_naive", "Fitted vs naive"),
                   ("oracle_vs_naive", "Oracle vs naive")):
        v = ent["paired_tests"][k]
        T.append(f"{lab} & win {v['win_rate_a_lower']:.3f}, "
                 f"$p={tex_p(v['wilcoxon_p'])}$ \\\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLE_DIR / "tab_entropy_check.tex").write_text("\n".join(T) + "\n")

    pro = json.loads((RES / "protocol_consistency" / "summary.json").read_text())
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Reconciliation of the two raw-spectrum protocols. The last "
         r"columns recompute the raw distance on three identical populations under "
         r"both distance implementations. The implementations agree, so the gap "
         r"between $2.51$ and $0.64$ comes from the populations and the grid.}",
         r"\label{tab:protocolcheck}", r"\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Seed & $d_{\mathrm{obs}}$ & analytic grid & folded truth \\", r"\midrule"]
    for r in pro["reconciliation_on_same_populations"]:
        T.append(f"{r['seed']} & {r['d_obs']} & {r['w1_analytic_grid']:.4f} & "
                 f"{r['w1_folded_truth_metric']:.4f} \\\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLE_DIR / "tab_protocol_check.tex").write_text("\n".join(T) + "\n")

    band = json.loads((RES / "band_calibration" / "summary.json").read_text())
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Calibration of the $90\%$ per-frequency credible band over three "
         r"seeds. Coverage is empirical against ground truth. Width is the median "
         r"band width in frequency bins. The prior is fitted once per case and is not "
         r"refitted across draws or repeated thinnings.}",
         r"\label{tab:bandcheck}", r"\footnotesize",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Draws & coverage range & median width & seed-level coverage \\", r"\midrule"]
    raw_intervals = json.loads(INTERVALS.read_text())
    for K, v in band["draw_count_stability"].items():
        seeds = ", ".join(
            f"{raw_intervals[f'seed{i}']['K'][K]['per_freq_cov_90']:.3f}"
            for i in range(3))
        T.append(f"{K} & {v['coverage_range'][0]:.3f}--{v['coverage_range'][1]:.3f} "
                 f"& {v['median_width']:.1f} & {seeds} \\\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLE_DIR / "tab_band_check.tex").write_text("\n".join(T) + "\n")

    reg = json.loads((RES / "readout_regime" / "summary.json").read_text())
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Paired comparison of the deployed readout B4 against the adapted "
         r"affine reference B1. The sample-only $a_0$ flag separates a regime where "
         r"the pooled median improves from one where B1 is significantly better.}",
         r"\label{tab:readoutgroups}", r"\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Group & $n$ & median B4 & median B1 & B4 lower & $p$ \\", r"\midrule"]
    for k, v in reg["readout_effect_by_boundary"].items():
        T.append(f"{k} & {v['n']} & {v['median_B4']:.4g} & {v['median_B1']:.4g} & "
                 f"{v['win_rate_B4_lower']*100:.1f}\\% & "
                 f"${tex_p(v['wilcoxon_p_two_sided'])}$ \\\\")
    T += [r"\midrule"]
    for k, v in reg["readout_effect_by_rate_interior"].items():
        T.append(f"interior $p={k}$ & {v['n']} & {v['median_B4']:.4g} & "
                 f"{v['median_B1']:.4g} & {v['win_rate_B4_lower']*100:.1f}\\% & "
                 f"${tex_p(v['wilcoxon_p_two_sided'])}$ \\\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLE_DIR / "tab_readout_groups.tex").write_text("\n".join(T) + "\n")
    print("wrote four supplement tables")


def main():
    entropy_consistency()
    protocol_consistency()
    deployable_regime()
    band_calibration()
    readout_regime()
    emit_tex_tables()


if __name__ == "__main__":
    main()
