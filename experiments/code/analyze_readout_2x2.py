"""Fixed-prior 2x2 readout factorial.

The deployed estimator and the adapted
affine reference differ in two places at once, namely the treatment of observed
types and the allocation of unseen types. This script holds the sample, the
fitted Gamma--Poisson prior, the estimated number of unseen types, the support
j <= 512, and the scoring protocol fixed, then compares all four combinations.

  B1  rounded posterior means  + all unseen types in the singleton bin
  B2  rounded posterior means  + unseen types spread by the posterior profile
  B3  integrated posterior     + all unseen types in the singleton bin
  B4  integrated posterior     + unseen types spread by the posterior profile

B1 is the adapted affine reference and B4 is the deployed readout. The two
intermediate arms isolate the two contributions.

Outputs
  experiments/results/readout_2x2/{metrics.csv,summary.json,report.md}
  experiments/results/tables/tab_readout_2x2.tex
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import d4d_logic as D  # noqa: E402
import d4d_baselines as B  # noqa: E402
import d4d_modern_baselines as M  # noqa: E402
from eb_prior import fitted_prior_eb  # noqa: E402
from validate_real_data import load_attribute  # noqa: E402
from validate_baselines import DATASETS, MIN_NDV, P_VALUES, stable_seed  # noqa: E402

J = 512
DATA_DIR = ROOT / "experiments" / "data" / "real"
OUT_DIR = ROOT / "experiments" / "results" / "readout_2x2"
OUT_TEX = ROOT / "experiments" / "results" / "tables" / "tab_readout_2x2.tex"
ARMS = ("B1", "B2", "B3", "B4")
OBS_MODE = {"B1": "rounded", "B2": "rounded", "B3": "integrated", "B4": "integrated"}
UNSEEN_MODE = {"B1": "singleton", "B2": "profile", "B3": "singleton", "B4": "profile"}


def observed_rounded(g, p, a0, b0):
    obs = g[g >= 1].astype(np.float64)
    if obs.size == 0:
        return np.zeros(J, dtype=np.float64)
    est = np.clip(np.round(obs + (1.0 - p) * (a0 + obs) / (b0 + p)), 1, J)
    return np.bincount(est.astype(np.int64), minlength=J + 1)[1:J + 1].astype(float)


def observed_integrated(g, p, prior_f, logfact, N_max):
    F = np.zeros(J, dtype=np.float64)
    groups = defaultdict(int)
    for gv in g[g >= 1]:
        groups[int(gv)] += 1
    for gv, mult in groups.items():
        fsup, prob = D.analytic_posterior(gv, p, prior_f, logfact, N_max)
        fsup = np.clip(fsup, 1, J)
        np.add.at(F, fsup - 1, mult * prob)
    return F


def build_case(g, p, prior_f, logfact, N_max, d_obs, a0, b0):
    """Return four spectra sharing one sample, prior, and unseen-type count."""
    i = np.arange(1, len(prior_f), dtype=np.float64)
    miss = prior_f[1:len(prior_f)] * np.power(1.0 - p, i)
    w = float(miss.sum())
    n_unseen = max(d_obs / (1.0 - w) - d_obs, 0.0) if 0.0 < w < 1.0 else 0.0
    profile = miss[:J] / float(miss.sum()) if miss.sum() > 0 else np.zeros(J)

    obs = {
        "rounded": observed_rounded(g, p, a0, b0),
        "integrated": observed_integrated(g, p, prior_f, logfact, N_max),
    }
    out = {}
    for arm in ARMS:
        F = obs[OBS_MODE[arm]].copy()
        if UNSEEN_MODE[arm] == "singleton":
            F[0] += n_unseen
        else:
            F[:J] += n_unseen * profile
        out[arm] = np.clip(F, 0.0, None)
    return out, n_unseen, w


def score(F_true, F_hat):
    m = B.distribution_metrics(F_true, F_hat)
    f1_true = float(F_true[0])
    f1_err = (abs(float(F_hat[0]) - f1_true) / f1_true) if f1_true > 0 else float("nan")
    ent = M.functional_relerrs(F_true, F_hat).get("entropy", float("nan"))
    return {
        "w1": m["w1"], "tv": m["tv"], "rmse": m["rmse"],
        "f1_err": f1_err, "entropy_err": ent,
        "total_types": float(F_hat.sum()),
        "total_mass": float(np.dot(np.arange(1, J + 1, dtype=float), F_hat)),
    }


def collect(limit=0):
    rows, timings = [], {"prior_fit": [], **{a: [] for a in ARMS}}
    for ds in DATASETS:
        d = DATA_DIR / ds
        if not d.is_dir():
            continue
        for name in sorted(f for f in __import__("os").listdir(d) if f.endswith(".csv")):
            f_true = load_attribute(str(d / name))
            if f_true.size < MIN_NDV:
                continue
            for pi, p in enumerate(P_VALUES):
                rng = np.random.default_rng(stable_seed(ds, name, "B", pi))
                g = D.thin(f_true, p, rng)
                n = int(g.sum())
                n_cap = int(max(2 * n / p, 4 * int(g.max()) / p)) + 1000
                d_obs = int((g >= 1).sum())
                F_true = D.fofo_spectrum(f_true, j_max=J)
                logfact = np.concatenate(
                    [[0.0], np.cumsum(np.log(np.arange(1, n_cap + 1, dtype=np.float64)))])
                t0 = time.perf_counter()
                prior_f, _ve, a0, b0 = fitted_prior_eb(g, p, n_cap)
                timings["prior_fit"].append(time.perf_counter() - t0)
                built, n_unseen, w = None, None, None
                rec = {"dataset": ds, "attribute": name, "p": p, "a0": a0, "b0": b0,
                       "d_obs": d_obs, "n_unseen": n_unseen, "w": w,
                       "true_types": float(F_true.sum()),
                       "true_mass": float(np.dot(np.arange(1, J + 1, dtype=float), F_true))}
                for arm in ARMS:
                    t1 = time.perf_counter()
                    if built is None:
                        built, n_unseen, w = build_case(g, p, prior_f, logfact, n_cap,
                                                        d_obs, a0, b0)
                        rec["n_unseen"], rec["w"] = n_unseen, w
                    s = score(F_true, built[arm])
                    timings[arm].append(time.perf_counter() - t1)
                    for k, v in s.items():
                        rec[f"{arm}_{k}"] = v
                rows.append(rec)
                if limit and len(rows) >= limit:
                    return rows, timings
    return rows, timings


def finite(rows, key):
    return np.asarray([r[key] for r in rows
                       if r.get(key) is not None and np.isfinite(r[key])], dtype=float)


def paired(rows, a, b, metric):
    pairs = [(r[f"{a}_{metric}"], r[f"{b}_{metric}"]) for r in rows
             if np.isfinite(r.get(f"{a}_{metric}", float("nan")))
             and np.isfinite(r.get(f"{b}_{metric}", float("nan")))]
    if len(pairs) < 3:
        return {"n": len(pairs)}
    x = np.asarray([u for u, _ in pairs]); y = np.asarray([v for _, v in pairs])
    return {"n": len(pairs), "win_rate_a_lower": float(np.mean(x < y)),
            "median_a": float(np.median(x)), "median_b": float(np.median(y)),
            "median_diff": float(np.median(x - y)),
            "wilcoxon_p_two_sided": float(wilcoxon(x, y, alternative="two-sided").pvalue)}


def group_medians(rows, keyfn, arm):
    out = defaultdict(list)
    for r in rows:
        if np.isfinite(r.get(f"{arm}_w1", float("nan"))):
            out[keyfn(r)].append(r[f"{arm}_w1"])
    return {str(k): {"n": len(v), "median_w1": float(np.median(v))}
            for k, v in sorted(out.items(), key=lambda t: str(t[0]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows, timings = collect(args.limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(rows[0].keys())
        with (OUT_DIR / "metrics.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

    summary = {
        "protocol": {
            "cases": len(rows), "J": J,
            "datasets": DATASETS, "rates": P_VALUES,
            "shared": ["thinned sample and seed", "fitted Gamma--Poisson prior (a0,b0)",
                       "support cap n_cap derived from the sample",
                       "estimated unseen-type count d_obs*w/(1-w)",
                       "j<=512 evaluation support", "distribution_metrics scoring"],
            "arms": {a: {"observed": OBS_MODE[a], "unseen": UNSEEN_MODE[a]} for a in ARMS},
        },
        "pooled": {a: {k: (float(np.median(finite(rows, f"{a}_{k}")))
                           if finite(rows, f"{a}_{k}").size else None)
                       for k in ("w1", "tv", "rmse", "f1_err", "entropy_err",
                                 "total_types", "total_mass")} for a in ARMS},
        "truth": {"total_types_median": float(np.median(finite(rows, "true_types")))
                  if finite(rows, "true_types").size else None,
                  "total_mass_median": float(np.median(finite(rows, "true_mass")))
                  if finite(rows, "true_mass").size else None},
        "conservation": {
            "max_abs_total_types_spread": float(max(
                max(r[f"{a}_total_types"] for a in ARMS)
                - min(r[f"{a}_total_types"] for a in ARMS) for r in rows)) if rows else None,
            "median_total_mass_by_arm": {
                a: float(np.median(finite(rows, f"{a}_total_mass")))
                if finite(rows, f"{a}_total_mass").size else None for a in ARMS},
        },
        "paired": {},
        "groups": {},
        "timing_ms_median": {k: float(np.median(v)) * 1000 if v else None
                             for k, v in timings.items()},
    }
    for metric in ("w1", "f1_err", "entropy_err"):
        for a, b in (("B4", "B1"), ("B2", "B1"), ("B3", "B1"),
                     ("B4", "B2"), ("B4", "B3"), ("B2", "B3")):
            summary["paired"][f"{a}_vs_{b}_{metric}"] = paired(rows, a, b, metric)
    for a in ARMS:
        summary["groups"][a] = {
            "by_dataset": group_medians(rows, lambda r: r["dataset"], a),
            "by_rate": group_medians(rows, lambda r: r["p"], a),
            "by_a0_boundary": group_medians(
                rows, lambda r: "boundary" if r["a0"] <= 1.001e-3 else "interior", a),
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    L = ["# Fixed-prior 2x2 readout factorial", "",
         f"{len(rows)} real-column cases. All four arms share one sample, one fitted prior, "
         "one unseen-type count, and one scoring protocol.", "",
         "| arm | observed types | unseen types | median W1 | median F1 rel err | "
         "median entropy rel err | median total types | median total mass |",
         "|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        p = summary["pooled"][a]
        f = lambda x: "---" if x is None else f"{x:.4g}"
        L.append(f"| {a} | {OBS_MODE[a]} | {UNSEEN_MODE[a]} | {f(p['w1'])} | "
                 f"{f(p['f1_err'])} | {f(p['entropy_err'])} | {f(p['total_types'])} | "
                 f"{f(p['total_mass'])} |")
    L += ["", "## Paired comparisons", ""]
    for k, v in summary["paired"].items():
        if "n" in v and "wilcoxon_p_two_sided" in v:
            L.append(f"- {k}: n={v['n']}, arm-a lower on {v['win_rate_a_lower']:.1%}, "
                     f"median diff {v['median_diff']:.4g}, "
                     f"Wilcoxon p={v['wilcoxon_p_two_sided']:.4g}")
    L += ["", "## Conservation", "",
          f"- Max within-case spread of total types across the four arms: "
          f"{summary['conservation']['max_abs_total_types_spread']:.6g}.",
          "- Total mass differs across arms because the profile allocation assigns "
          "unseen types to frequencies above one. W1 and TV are computed on "
          "renormalized spectra, so the shape comparison is unaffected.", "",
          "## Cost (median wall-clock per case, ms)", ""]
    for k, v in summary["timing_ms_median"].items():
        L.append(f"- {k}: {v:.3f}" if v is not None else f"- {k}: ---")
    L += [""]
    (OUT_DIR / "report.md").write_text("\n".join(L))

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    f = lambda x: "---" if x is None else (f"{x:.3g}")
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Fixed-prior $2\times2$ readout factorial on the real-column cases. "
         r"All four arms share one thinned sample, one fitted Gamma--Poisson prior, one "
         r"estimated unseen-type count, and the $j\le512$ evaluation support. B1 is the "
         r"adapted affine reference and B4 is the deployed readout. Lower is better.}",
         r"\label{tab:readout2x2}", r"\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llllll}", r"\toprule",
         r"Arm & observed & unseen & $\Wone$ & $\Fj{1}$ err & entropy err \\",
         r"\midrule"]
    for a in ARMS:
        p = summary["pooled"][a]
        T.append(f"{a} & {OBS_MODE[a]} & {UNSEEN_MODE[a]} & {f(p['w1'])} & "
                 f"{f(p['f1_err'])} & {f(p['entropy_err'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_TEX.write_text("\n".join(T) + "\n")
    print(json.dumps({"cases": len(rows), "pooled": summary["pooled"],
                      "timing_ms": summary["timing_ms_median"]}, indent=2))
    print("wrote", OUT_DIR / "summary.json", "and", OUT_TEX)


if __name__ == "__main__":
    main()
