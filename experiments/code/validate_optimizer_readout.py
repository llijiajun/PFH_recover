"""Optimizer-facing readout. Join-size Q-error from a recovered FoF histogram.

Motivation. A DBMS catalog stores a row count and an NDV scalar. A query optimizer
that needs an equi-join size must then fall back to a uniform assumption, which
gives N^2 / NDV. Recovering the full frequency-of-frequencies histogram instead
gives SUM_v f_v^2 directly. This script measures the gap in Q-error, the standard
cardinality-estimation metric, on the same 276 real-column cases as the main text.

Q-error is max(est/true, true/est), so it is symmetric and always at least one.
All spectra use the common capped support j <= 256, including the ground truth,
so every method is evaluated on exactly the same estimand. The uncapped truth is
reported separately as a sensitivity check.

Outputs in experiments/results/optimizer_readout/:
  qerror_metrics.csv, qerror_summary.json, qerror_report.md
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import d4d_logic as D
import d4d_baselines as B
import d4d_modern_baselines as M
from d4d_metrics import prior_from_truth
from eb_prior import fitted_prior_eb
from validate_real_data import load_attribute
from validate_baselines import DATASETS, MIN_NDV, P_VALUES, stable_seed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "results", "optimizer_readout")
DATA_DIR = os.path.join(HERE, "..", "data", "real")

ARMS = [
    "catalog_uniform_true", "catalog_uniform_naive", "raw_sample",
    "good_turing", "sb_linear", "ours_fitted", "ours_oracle",
]


def q_error(est: float, true: float) -> float:
    """Symmetric cardinality Q-error. Non-finite inputs stay non-finite."""
    if not (np.isfinite(est) and np.isfinite(true)) or true <= 0 or est <= 0:
        return float("nan")
    return float(max(est / true, true / est))


def self_join(F: np.ndarray) -> float:
    return M.downstream_functionals(F)["self_join"]


def evaluate_one(f_true: np.ndarray, p: float, rng) -> dict:
    f_true = np.asarray(f_true, dtype=np.int64)
    N_total = int(f_true.sum())
    g = D.thin(f_true, p, rng)
    d_obs = int((g >= 1).sum())
    n = int(g.sum())
    f_samp = D.fofo_spectrum(g)
    F_true = D.fofo_spectrum(f_true, j_max=M.H_CAP)
    F_true_uncapped = D.fofo_spectrum(f_true)
    ndv_true = float(F_true.sum())

    out = {
        "p": float(p), "NDV_true": int(f_true.size), "N_true": N_total,
        "d_obs": d_obs, "n_sample": n,
    }

    prior_f, logfact = prior_from_truth(f_true, N_total)
    pf_eb, ve, a0, b0 = fitted_prior_eb(g, p, N_total)
    out["a0_sb"], out["b0_sb"] = a0, b0

    specs = {
        "raw_sample": B.spectrum_naive(g, p),
        "good_turing": B.spectrum_good_turing(f_samp),
        "sb_linear": M.spectrum_smooth_bayes(g, p, a0, b0, d_obs),
        "ours_fitted": B.spectrum_ours(g, p, pf_eb, logfact, N_total, d_obs),
        "ours_oracle": B.spectrum_ours(g, p, prior_f, logfact, N_total, d_obs),
    }

    true_sj = self_join(F_true)
    true_sj_uncapped = self_join(F_true_uncapped)
    out["self_join_true_capped"] = true_sj
    out["self_join_true_uncapped"] = true_sj_uncapped

    N_hat = n / p
    ndv_hat = d_obs / p
    scalar = {
        # Uniform catalog estimate with the TRUE row count and TRUE NDV.
        # This is optimistic, since a real catalog has neither.
        "catalog_uniform_true": N_total ** 2 / ndv_true if ndv_true > 0 else np.nan,
        # Uniform catalog estimate from the two scalars a sample can supply.
        "catalog_uniform_naive": N_hat ** 2 / ndv_hat if ndv_hat > 0 else np.nan,
    }

    for name in ARMS:
        est = scalar[name] if name in scalar else self_join(specs[name])
        out[f"{name}_est"] = est
        out[f"{name}_q"] = q_error(est, true_sj)
        out[f"{name}_rel"] = abs(est - true_sj) / true_sj if true_sj > 0 else np.nan
    return out


def _arr(rows: list, key: str) -> np.ndarray:
    return np.array([float(r[key]) if r.get(key) not in (None, "", "nan", "inf", "-inf")
                     else np.nan for r in rows], dtype=float)


def qstats(rows: list, truth_key: str, mask: np.ndarray) -> dict:
    """Q-error statistics for one (truth, case-subset) evaluation convention."""
    truth = _arr(rows, truth_key)
    q = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name in ARMS:
            est = _arr(rows, f"{name}_est")
            q[name] = np.where(mask, np.maximum(est / truth, truth / est), np.nan)
    arms = {}
    for name in ARMS:
        v = q[name]
        v = v[np.isfinite(v) & (v > 0)]
        arms[name] = {
            "n_finite": int(v.size),
            "q_geomean": float(np.exp(np.mean(np.log(v)))) if v.size else None,
            "q_median": float(np.median(v)) if v.size else None,
            "q_p90": float(np.percentile(v, 90)) if v.size else None,
            "q_max": float(v.max()) if v.size else None,
            "within_2x": float(np.mean(v <= 2.0)) if v.size else None,
        }
    paired = {}
    for name in ARMS:
        if name == "ours_fitted":
            continue
        a, b = q["ours_fitted"], q[name]
        m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if m.sum() < 10:
            continue
        x, y = a[m], b[m]
        paired[f"ours_fitted_vs_{name}"] = {
            "n": int(m.sum()),
            "win_rate": float(np.mean(x < y)),
            "wilcoxon_p_less": float(wilcoxon(x, y, alternative="less").pvalue),
        }
    return {"truth": truth_key, "n_cases": int(mask.sum()),
            "arms": arms, "paired": paired}


def summarize(rows: list) -> dict:
    cap = _arr(rows, "self_join_true_capped")
    unc = _arr(rows, "self_join_true_uncapped")
    ratio = np.where(unc > 0, cap / unc, 1.0)
    # A case is "clean" when capping the truth at 256 changes SUM f_v^2 by < 1%.
    # On the other cases the capped truth is not the optimizer's estimand, so
    # using it as a Q-error denominator would manufacture huge spurious errors.
    clean = ratio >= 0.99
    every = np.ones(len(rows), dtype=bool)
    return {
        "n_cases": len(rows),
        "n_clean": int(clean.sum()),
        "n_support_truncated": int((~clean).sum()),
        "primary": qstats(rows, "self_join_true_uncapped", clean),
        "sensitivity": {
            "uncapped_all": qstats(rows, "self_join_true_uncapped", every),
            "capped_all_common_support": qstats(rows, "self_join_true_capped", every),
        },
    }


def _table(block: dict) -> list:
    L = ["| estimator | finite | geomean | median | p90 | max | within 2x |",
         "|---|---|---|---|---|---|---|"]
    for name in ARMS:
        a = block["arms"][name]
        L.append(f"| {name} | {a['n_finite']} | {a['q_geomean']:.3f} | "
                 f"{a['q_median']:.3f} | {a['q_p90']:.3f} | "
                 f"{a['q_max']:.3g} | {a['within_2x']:.1%} |")
    return L


def write_report(rows: list, s: dict) -> None:
    p = s["primary"]
    L = ["# Join-size Q-error from a recovered frequency histogram", "",
         f"Paired cases {s['n_cases']}. Primary convention uses the UNCAPPED ground truth",
         f"SUM_v f_v^2 on the {s['n_clean']} cases where capping the truth at j <= 256 changes",
         "it by less than one percent. All spectra still share the estimator support",
         "j <= 256. Q-error is max(est/true, true/est).", "",
         "## Primary convention", ""] + _table(p) + [
         "", "## Paired comparisons against ours_fitted", "",
         "| competitor | n | ours wins | Wilcoxon p |", "|---|---|---|---|"]
    for k, v in p["paired"].items():
        L.append(f"| {k.split('_vs_')[1]} | {v['n']} | {v['win_rate']:.1%} | "
                 f"{v['wilcoxon_p_less']:.3g} |")

    L += ["", "## Reading, including results unfavorable to our framing", "",
          "`catalog_uniform_true` uses the TRUE row count and TRUE NDV, so it is an",
          "ORACLE baseline that no deployed catalog can reach. On the primary cases it",
          f"achieves geomean {p['arms']['catalog_uniform_true']['q_geomean']:.3f} and",
          f"{p['arms']['catalog_uniform_true']['within_2x']:.1%} within 2x, while ours_fitted",
          f"reaches {p['arms']['ours_fitted']['q_geomean']:.3f}. We do NOT beat this oracle.",
          f"The paired win rate is {p['paired']['ours_fitted_vs_catalog_uniform_true']['win_rate']:.1%}.",
          "The 810x example in the introduction is a constructed extreme, not the typical",
          "real column. Most real columns are close enough to uniform that two scalars",
          "already give a good join-size estimate.", "",
          "The honest gain is over DEPLOYABLE alternatives. The sample plug-in catalog",
          f"reaches {p['arms']['catalog_uniform_naive']['q_geomean']:.3f} geomean and ours_fitted",
          f"wins {p['paired']['ours_fitted_vs_catalog_uniform_naive']['win_rate']:.1%} of paired",
          f"cases (p = {p['paired']['ours_fitted_vs_catalog_uniform_naive']['wilcoxon_p_less']:.2g}).",
          f"The raw sample is {p['arms']['raw_sample']['q_geomean']:.3f} geomean.",
          "Scalar NDV estimators remain structurally unable to report entropy, rare-value",
          "mass, or a calibrated interval, which is where the full histogram earns its cost.", "",
          "The oracle prior row is an information ceiling and is not deployable.", ""]

    L += ["## Sensitivity to the ground-truth convention", "",
          f"Of {s['n_cases']} cases, {s['n_support_truncated']} have population counts above",
          "the common support. On those cases a capped denominator produces enormous",
          "spurious Q-errors, so it is not used for the primary claim.", ""]
    for tag, label in [("uncapped_all", "Uncapped truth, all cases"),
                       ("capped_all_common_support", "Capped truth, all cases")]:
        b = s["sensitivity"][tag]
        L += ["", f"### {label} (n = {b['n_cases']})", ""] + _table(b)

    L += ["", "Using the uncapped truth on all cases, every spectrum-based method degrades",
          "sharply because the estimator support stops at j = 256. This is a real",
          "limitation of the common support, reported as limitation (x) in the main",
          "text, not a defect of Q-error as a metric.", "",
          "This is an offline cardinality readout. No plan quality, execution time, or",
          "end-to-end optimizer measurement is claimed.", ""]
    with open(os.path.join(OUT_DIR, "qerror_report.md"), "w") as fh:
        fh.write("\n".join(L))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summarize-only", action="store_true",
                    help="rebuild summary and report from the archived CSV")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.summarize_only:
        with open(os.path.join(OUT_DIR, "qerror_metrics.csv"), newline="") as fh:
            rows = list(csv.DictReader(fh))
        s = summarize(rows)
        with open(os.path.join(OUT_DIR, "qerror_summary.json"), "w") as fh:
            json.dump(s, fh, indent=2)
        write_report(rows, s)
        print(f"rebuilt from {len(rows)} archived rows, "
              f"primary n = {s['n_clean']}", flush=True)
        return

    cases = []
    for ds in DATASETS:
        ds_dir = os.path.join(DATA_DIR, ds)
        attrs = sorted(f for f in os.listdir(ds_dir) if f.endswith(".csv"))
        for attr in attrs:
            f_true = load_attribute(os.path.join(ds_dir, attr))
            if f_true.size < MIN_NDV:
                continue
            for pi, p in enumerate(P_VALUES):
                cases.append((ds, attr, p, stable_seed(ds, attr, "B", pi), f_true))
                if args.limit and len(cases) >= args.limit:
                    break
            if args.limit and len(cases) >= args.limit:
                break
        if args.limit and len(cases) >= args.limit:
            break

    rows, t0 = [], time.time()
    for i, (ds, attr, p, seed, f_true) in enumerate(cases, 1):
        rng = np.random.default_rng(seed)
        try:
            res = evaluate_one(f_true, p, rng)
        except Exception as e:
            print(f"  FAIL {ds}/{attr} p={p}: {type(e).__name__}: {e}", flush=True)
            continue
        res["dataset"], res["attribute"] = ds, attr
        rows.append(res)
        if i % 25 == 0:
            print(f"  [{i}/{len(cases)}] {time.time()-t0:.0f}s", flush=True)
    print(f"ran {len(rows)} cases in {time.time()-t0:.1f}s", flush=True)

    keys = ["dataset", "attribute"] + [k for k in rows[0] if k not in ("dataset", "attribute")]
    with open(os.path.join(OUT_DIR, "qerror_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    s = summarize(rows)
    with open(os.path.join(OUT_DIR, "qerror_summary.json"), "w") as fh:
        json.dump(s, fh, indent=2)
    write_report(rows, s)
    for name in ARMS:
        a = s["primary"]["arms"][name]
        print(f"  {name:24s} geomean {a['q_geomean']:.3f} "
              f"median {a['q_median']:.3f} within2 {a['within_2x']:.1%}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
