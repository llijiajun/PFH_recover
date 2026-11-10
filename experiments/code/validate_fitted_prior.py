"""Deployment-state check: oracle prior vs FITTED prior for the closed-form inverse.

The paper's Q3 baseline table (tab:baselines) reports the closed-form inverse with an
ORACLE prior (built from ground-truth frequencies) -- the best case.  The experiment also checks the following.
what happens in deployment, where the prior must be estimated from the observable
sample alone?  This script re-runs the IDENTICAL 276 cases (same datasets, same seeds)
with a fitted empirical-Bayes prior that uses only observable quantities:

  1. Chao1 effective support  Ve = d_obs + f1^2 / (2 f2)
  2. Method-of-moments Gamma shape on de-thinned observed counts x = g/p
       mean m, var v  ->  a0 = m^2 / v (clipped to [0.25, 4]; a0 = 2 if v <= m)
       rate b0 = a0 / m
  3. Prior over the population count of an existing type = the Gamma-Poisson
     (negative-binomial) predictive with (a0, b0), truncated to j >= 1 and
     renormalized.  The unseen-type share is then handled exactly as in the paper:
     w = sum_j prior_f[j] (1-p)^j,  D_hat = d_obs / (1 - w).

No ground-truth quantity enters the fitted prior.  Everything else (forward sample,
metrics, support cap) is byte-identical to validate_baselines.py.

Outputs (experiments/results/fitted_prior_validation/):
  fitted_metrics.csv, fitted_summary.json, fitted_report.md
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d4d_logic as D
import d4d_baselines as B
from d4d_metrics import spectrum_w1
from validate_real_data import load_attribute
from validate_baselines import (DATA_DIR, DATASETS, MIN_NDV, P_VALUES, stable_seed)
from eb_prior import fitted_prior_eb
from scipy.special import gammaln

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "fitted_prior_validation")

A0_CLIP = (0.25, 4.0)   # the paper's measured sensitivity range for the Gamma shape


def fitted_prior(g: np.ndarray, p: float, N_max: int):
    """Empirical-Bayes prior from observables only.  Returns (prior_f, logfact)."""
    g = np.asarray(g, dtype=np.int64)
    obs = g[g >= 1].astype(np.float64)
    d_obs = int(obs.size)
    n = int(g.sum())
    f_samp = D.fofo_spectrum(g)
    f1 = float(f_samp[0]) if f_samp.size > 0 else 0.0
    f2 = float(f_samp[1]) if f_samp.size > 1 else 0.0
    ve = d_obs + (f1 * f1) / (2.0 * f2) if f2 > 0 else d_obs + f1 * (f1 - 1.0) / 2.0

    x = obs / p                       # de-thinned observed counts
    m = float(x.mean()) if x.size else 1.0
    v = float(x.var()) if x.size else 0.0
    if v > m and m > 0:
        a0 = float(np.clip(m * m / v, *A0_CLIP))
    else:
        a0 = 2.0                      # paper's interior optimum when under-dispersed
    b0 = a0 / m if m > 0 else 1.0

    # Gamma-Poisson (NB) predictive over the population count j >= 1
    js = np.arange(1, N_max + 1, dtype=np.float64)
    from scipy.special import gammaln
    logw = (gammaln(js + a0) - gammaln(a0) - gammaln(js + 1)
            + a0 * math.log(b0 / (b0 + 1.0)) + js * math.log(1.0 / (b0 + 1.0)))
    w = np.exp(logw)
    s = w.sum()
    prior_f = np.zeros(N_max + 1, dtype=np.float64)
    if s > 0:
        prior_f[1:] = w / s
    logfact = gammaln(np.arange(N_max + 512 + 1) + 1.0)
    return prior_f, logfact, ve, a0, b0


def ours_fit_metrics(f_true: np.ndarray, p: float, rng: np.random.Generator) -> dict:
    """Run ours with oracle prior AND fitted prior on one case; return metrics."""
    f_true = np.asarray(f_true, dtype=np.int64)
    N_total = int(f_true.sum())
    g = D.thin(f_true, p, rng)
    d_obs = int((g >= 1).sum())
    n = int(g.sum())
    F_true = D.fofo_spectrum(f_true, j_max=B.H_CAP)
    N_max = max(int(n / p) + 64, d_obs + 64)

    prior_o, logfact_o = B.prior_from_truth(f_true, N_total)
    F_o = B.spectrum_ours(g, p, prior_o, logfact_o, N_total, d_obs)
    m_o = B.distribution_metrics(F_true, F_o)

    prior_f, logfact_f, ve, a0, b0 = fitted_prior(g, p, N_max)
    F_f = B.spectrum_ours(g, p, prior_f, logfact_f, N_max, d_obs)
    m_f = B.distribution_metrics(F_true, F_f)

    pf_eb, ve_eb, a0_eb, b0_eb = fitted_prior_eb(g, p, N_max)
    lf_eb = gammaln(np.arange(N_max + 513) + 1.0)
    F_e = B.spectrum_ours(g, p, pf_eb, lf_eb, N_max, d_obs)
    m_e = B.distribution_metrics(F_true, F_e)

    out = {
        "p": float(p),
        "NDV_true": int(f_true.size),
        "NDV_true_capped": float(F_true.sum()),
        "N_true": N_total,
        "d_obs": d_obs,
        "n_sample": n,
        "NDV_lost_frac": float(1.0 - d_obs / max(f_true.size, 1)),
        "Ve_chao1": float(ve),
        "a0_fit": a0,
        "b0_fit": b0,
        "a0_eb": a0_eb,
        "b0_eb": b0_eb,
    }
    for tag, m in (("oracle", m_o), ("fit", m_f), ("eb", m_e)):
        for k, val in m.items():
            out[f"{tag}_{k}"] = val
    return out


def block(rows, prefix):
    out = {}
    for k in ("ndv_rel", "w1", "tv", "rmse", "pop_rel"):
        vals = [r[f"{prefix}_{k}"] for r in rows
                if not (isinstance(r[f"{prefix}_{k}"], float) and math.isnan(r[f"{prefix}_{k}"]))]
        if vals:
            out[f"{prefix}_{k}_median"] = float(np.median(vals))
            out[f"{prefix}_{k}_mean"] = float(np.mean(vals))
            out[f"{prefix}_{k}_q25"] = float(np.percentile(vals, 25))
            out[f"{prefix}_{k}_q75"] = float(np.percentile(vals, 75))
        else:
            for s in ("median", "mean", "q25", "q75"):
                out[f"{prefix}_{k}_{s}"] = float("nan")
    for tag in ("oracle", "fit", "eb"):
        rel = [r[f"{tag}_ndv_rel"] for r in rows if not math.isnan(r[f"{tag}_ndv_rel"])]
        out[f"{tag}_within10"] = float(np.mean([x <= 0.10 for x in rel])) if rel else float("nan")
        out[f"{tag}_within25"] = float(np.mean([x <= 0.25 for x in rel])) if rel else float("nan")
    # paired sign comparison: eb (deployment) vs oracle (ceiling)
    both = [(r["eb_ndv_rel"], r["oracle_ndv_rel"]) for r in rows
            if not (math.isnan(r["eb_ndv_rel"]) or math.isnan(r["oracle_ndv_rel"]))]
    out["n_paired"] = len(both)
    out["eb_wins_paired"] = float(np.mean([a < b for a, b in both])) if both else float("nan")
    # paired eb vs best classical (naive d/p numbers come from metrics.csv later)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for ds in DATASETS:
        ds_dir = os.path.join(DATA_DIR, ds)
        attrs = sorted(f for f in os.listdir(ds_dir) if f.endswith(".csv"))
        for attr in attrs:
            f_true = load_attribute(os.path.join(ds_dir, attr))
            if f_true.size < MIN_NDV:
                continue
            for pi, p in enumerate(P_VALUES):
                seed = stable_seed(ds, attr, "B", pi)
                rng = np.random.default_rng(seed)
                try:
                    res = ours_fit_metrics(f_true, p, rng)
                except Exception as e:
                    print(f"  FAIL {ds}/{attr} p={p}: {type(e).__name__}: {e}")
                    continue
                res["dataset"] = ds
                res["attribute"] = attr
                rows.append(res)
                print(f"  {ds}/{attr} p={p}: oracle_ndv={res['oracle_ndv_rel']:.3f} "
                      f"eb_ndv={res['eb_ndv_rel']:.3f} "
                      f"oracle_w1={res['oracle_w1']:.3g} eb_w1={res['eb_w1']:.3g}")
    print(f"ran {len(rows)} cases")

    keys = ["dataset", "attribute", "p", "NDV_true", "NDV_true_capped", "N_true",
            "d_obs", "n_sample", "NDV_lost_frac", "Ve_chao1", "a0_fit", "b0_fit",
            "a0_eb", "b0_eb"]
    for tag in ("oracle", "fit", "eb"):
        for k in ("ndv_rel", "w1", "tv", "rmse", "pop_rel"):
            keys.append(f"{tag}_{k}")
    path = os.path.join(OUT_DIR, "fitted_metrics.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", path)

    summary = {"n_cases": len(rows),
               "overall": block(rows, "oracle") | block(rows, "fit") | block(rows, "eb")}
    per_ds = {}
    for ds in DATASETS:
        sub = [r for r in rows if r["dataset"] == ds]
        per_ds[ds] = block(sub, "oracle") | block(sub, "fit") | block(sub, "eb")
    summary["per_dataset"] = per_ds
    with open(os.path.join(OUT_DIR, "fitted_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    o = summary["overall"]
    lines = []
    lines.append("# Oracle vs fitted prior for the closed-form inverse\n")
    lines.append(f"Same {len(rows)} (dataset, attribute, p) cases and seeds as "
                 "baseline_validation. Two fitted variants, both observable-only:\n"
                 "- **fit** = MoM Gamma on de-thinned observed counts + NB predictive "
                 "(ignores zero truncation; a lower reference).\n"
                 "- **eb** = empirical-Bayes NB matched to the observed singleton/doubleton "
                 "mass fractions (C-free identifying equations; the principled deployment "
                 "prior).\n")
    lines.append("\n## Overall (median rel err / median metric)\n")
    lines.append("| prior | NDV rel | w10 | w25 | W1 | TV | RMSE |")
    lines.append("|---|---|---|---|---|---|---|")
    for tag in ("oracle", "fit", "eb"):
        lines.append(
            f"| {tag} | {o[f'{tag}_ndv_rel_median']:.1%} | "
            f"{o[f'{tag}_within10']:.1%} | {o[f'{tag}_within25']:.1%} | "
            f"{o[f'{tag}_w1_median']:.3f} | {o[f'{tag}_tv_median']:.3f} | "
            f"{o[f'{tag}_rmse_median']:.3f} |")
    lines.append("\nPaired: the EB fitted prior beats the oracle ceiling on NDV rel err in "
                 f"{o['eb_wins_paired']:.1%} of {o['n_paired']} cases.\n")
    lines.append("\n## Per dataset (NDV rel err median)\n")
    lines.append("| dataset | cases | oracle | MoM fit | EB fit |")
    lines.append("|---|---|---|---|---|")
    for ds in DATASETS:
        b = per_ds[ds]
        lines.append(f"| {ds} | {b['n_paired']} | "
                     f"{b['oracle_ndv_rel_median']:.1%} | "
                     f"{b['fit_ndv_rel_median']:.1%} | "
                     f"{b['eb_ndv_rel_median']:.1%} |")
    rep = os.path.join(OUT_DIR, "fitted_report.md")
    with open(rep, "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", rep)
    print(json.dumps({k: v for k, v in o.items() if "ndv_rel" in k or "w1" in k},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
