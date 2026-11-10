"""Wall-clock and peak-memory comparison of the Table V estimators.

Review round 2, Issue 5 asks for a runtime and memory comparison on the real
columns.  This script times each estimator on two representative real cases
(the median and the largest number of observed distinct values at p = 0.3) and
reports the median of repeated runs together with tracemalloc peak allocation.

No estimator is modified.  All inputs come from the archived real data and the
same modules that produced Table V.

Outputs
  experiments/results/runtime_benchmark/summary.json
  experiments/results/runtime_benchmark/report.md
  experiments/results/tables/tab_runtime.tex
"""
from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import numpy as np
import d4d_logic as D
import d4d_baselines as B
import d4d_modern_baselines as M
from d4d_metrics import prior_from_truth
from eb_prior import fitted_prior_eb
from validate_real_data import load_attribute

DATA_DIR = ROOT / "experiments/data/real"
if not DATA_DIR.is_dir():
    DATA_DIR = ROOT / "data/real"
OUT_DIR = ROOT / "experiments/results/runtime_benchmark"
TABLE = ROOT / "experiments/results/tables/tab_runtime.tex"
P = 0.3
REPEAT = 4
K_DRAWS = 120
MAX_N = 500_000  # n_cap = 2n/p must stay small enough to time reliably


def collect_cases():
    cases = []
    for ds in ("Airlines", "kasandr", "NCVoter"):
        d = DATA_DIR / ds
        if not d.is_dir():
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".csv"):
                continue
            f_true = load_attribute(str(d / name))
            n_total = int(f_true.sum())
            if f_true.size < 10 or n_total > MAX_N:
                continue
            rng = np.random.default_rng(20261108)
            g = D.thin(f_true, P, rng)
            d_obs = int((g >= 1).sum())
            cases.append((d_obs, ds, name, f_true, g, n_total))
    cases.sort(key=lambda t: t[0])
    if len(cases) < 2:
        raise RuntimeError("not enough real columns under the size cap")
    return [cases[len(cases) // 2], cases[-1]]


def measure(fn):
    times, peaks = [], []
    for _ in range(REPEAT):
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak / (1024.0 * 1024.0))
    return {"ms_median": median(times), "mb_peak": max(peaks)}


def build(case):
    _d_rank, ds, name, f_true, g, N_total = case
    d_obs_true = int((g >= 1).sum())
    n = int(g.sum())
    n_cap = int(max(2 * n / P, 4 * int(g.max()) / P)) + 1000
    f_samp = D.fofo_spectrum(g)
    logfact = D._logfact(n_cap + 1)
    pi_npmle, p0, F_npmle, K_npmle = M.npmle_prior(f_samp, P, int(g.max()))
    pf_eb, ve, a0, b0 = fitted_prior_eb(g, P, n_cap)
    _, logfact_ours = prior_from_truth(f_true, max(N_total, n_cap))

    runners = {
        "Naive $1/p$": lambda: B.spectrum_naive(g, P),
        "Good--Turing": lambda: B.spectrum_good_turing(f_samp),
        "SVD deconvolution": lambda: B.spectrum_svd(f_samp, P),
        "Chebyshev": lambda: B.spectrum_chebyshev(f_samp),
        "NPMLE prior fit": lambda: M.npmle_prior(f_samp, P, int(g.max())),
        "NPMLE spectrum": lambda: M.spectrum_npmle(f_samp, pi_npmle, P, F_npmle, K_npmle, d_obs_true),
        "SB linear": lambda: M.spectrum_smooth_bayes(g, P, a0, b0, d_obs_true),
        "EB prior fit": lambda: fitted_prior_eb(g, P, n_cap),
        "Ours, spectrum": lambda: B.spectrum_ours(g, P, pf_eb, logfact_ours, n_cap, d_obs_true),
        "Ours, $120$-draw band": lambda: D.plausible_set(
            f_true, g, P, K_DRAWS, pf_eb, logfact_ours, n_cap,
            np.random.default_rng(0)),
    }
    out = {}
    for label, fn in runners.items():
        try:
            out[label] = measure(fn)
        except Exception as exc:  # keep the table honest about failures
            out[label] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  {label:26s} "
              + (out[label]["error"] if "error" in out[label]
                 else f"{out[label]['ms_median']:.1f} ms / "
                      f"{out[label]['mb_peak']:.1f} MB"), flush=True)
    return {"dataset": ds, "attribute": name, "p": P,
            "d_obs": int((g >= 1).sum()), "n_sample": n,
            "N_true": N_total, "results": out}


def write_table(blocks):
    labels = list(blocks[0]["results"].keys())
    head_parts = []
    for b in blocks:
        head_parts.append(r"\multicolumn{2}{c}{" + b["dataset"]
                          + r", $d_{\mathrm{obs}}=" + str(b["d_obs"]) + r"$}")
    heads = " & ".join(head_parts)
    L = [r"\begin{table}[t]", r"\centering",
         rf"\caption{{Wall-clock time and peak allocated memory of the spectrum "
         rf"estimators compared in the main text, on two real columns at $p={P}$. Each entry is the median "
         rf"of {REPEAT} runs. Prior fitting and the posterior-draw band are separate rows.}}",
         r"\label{tab:runtime}", r"\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"& " + heads + r" \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Method & ms & MB & ms & MB \\", r"\midrule"]
    for lab in labels:
        vals = []
        for b in blocks:
            r = b["results"][lab]
            if "error" in r:
                vals.extend(["---", "---"])
            else:
                vals.extend([f"{r['ms_median']:.1f}" if r["ms_median"] >= 0.1 else "<0.1",
                             f"{r['mb_peak']:.1f}" if r["mb_peak"] >= 0.1 else "<0.1"])
        L.append(f"{lab} & " + " & ".join(vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(L) + "\n")


def main():
    cases = collect_cases()
    blocks = [build(c) for c in cases]
    summary = {"p": P, "repeats": REPEAT, "draws": K_DRAWS,
               "cases": [{"dataset": b["dataset"], "attribute": b["attribute"],
                          "d_obs": b["d_obs"], "n_sample": b["n_sample"]}
                         for b in blocks],
               "measurements": [b["results"] for b in blocks]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    L = ["# Runtime and memory of the Table V estimators", "",
         f"Rate $p={P}$, median of {REPEAT} runs, peak from `tracemalloc`.", ""]
    for b in blocks:
        L += [f"## {b['dataset']}/{b['attribute']} "
              f"(d_obs={b['d_obs']}, n={b['n_sample']})", "",
              "| Method | ms | MB |", "|---|---:|---:|"]
        for lab, r in b["results"].items():
            if "error" in r:
                L.append(f"| {lab} | {r['error']} | |")
            else:
                L.append(f"| {lab} | {r['ms_median']:.1f} | {r['mb_peak']:.1f} |")
        L.append("")
    (OUT_DIR / "report.md").write_text("\n".join(L))
    write_table(blocks)
    print(json.dumps(summary, indent=2))
    print("wrote", OUT_DIR / "summary.json", "and", TABLE)


if __name__ == "__main__":
    main()
