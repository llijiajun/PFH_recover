"""Mechanism behind the no-thickening advantage reported in Table II.

This is an independent analytic diagnostic grid. It regenerates its own Zipf
populations with seeds derived from 710000 + 13*seed + 1000*p + 100*alpha and
scores W1 on spectra renormalized over j <= 512. It does NOT reuse the network
grid populations behind the 2.51 value in the main-text tables. Values from
the two protocols must not be compared as repeated measurements.

  1. Metric mechanism. W1 acts on spectra renormalized over j <= 512, so it
     measures shape only. The raw spectrum keeps a shape close to the truth
     for heavy-tailed populations while losing most of its support size. The
     distinct-count error is reported next to W1 to make that trade-off visible.
  2. Prior mechanism. The affine posterior mean has design bias
     (1-p)(a0 - b0 f)/(b0+p). The fitted (a0,b0), the bias scale
     (1-p)a0/(b0+p), and the oracle-prior ceiling are recorded at each rate.
  3. Rate mechanism. The whole grid is swept over p so the crossing point
     between the raw spectrum and the fitted inverse is observed directly.

The oracle row builds its prior from the full population spectrum. It is an
information ceiling, not a deployable estimator, and is labelled as such.

Outputs:
  experiments/results/table2_mechanism/summary.json
  experiments/results/table2_mechanism/report.md
  experiments/results/tables/tab_table2_mechanism.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from d4d_logic import (  # noqa: E402
    zipf_population, thin, fofo_spectrum, w1_distance,
)
from d4d_baselines import spectrum_ours  # noqa: E402
from eb_prior import fitted_prior_eb  # noqa: E402

J = 512
ALPHAS = [1.0, 1.5, 2.0]
RATES = [0.05, 0.1, 0.2, 0.3, 0.5]
SEEDS = [0, 1, 2]
N = 100_000
VOCAB = 30_000

OUT = ROOT / "experiments" / "results" / "table2_mechanism"
OUT_JSON = OUT / "summary.json"
OUT_REPORT = OUT / "report.md"
OUT_TEX = ROOT / "experiments" / "results" / "tables" / "tab_table2_mechanism.tex"


def truth_spectrum(f_true):
    F = np.zeros(J)
    for f in f_true[f_true > 0]:
        if f <= J:
            F[f - 1] += 1.0
    return F


def oracle_prior(F_true, N_max):
    """Prior over clean counts built from the full population spectrum.

    This uses ground truth and is an information ceiling only.
    """
    pf = np.zeros(N_max + 1)
    top = min(N_max, J)
    pf[1:top + 1] = F_true[:top]
    s = pf.sum()
    if s > 0:
        pf /= s
    return pf


def missing_fraction(prior_f, p):
    i = np.arange(1, len(prior_f), dtype=np.float64)
    return float(np.sum(prior_f[1:] * np.power(1.0 - p, i)))


def cdf_of(spec):
    s = float(spec.sum())
    if s <= 0:
        return np.zeros(J)
    return np.cumsum(spec / s)


def pad_to_J(spec):
    s = np.asarray(spec, dtype=np.float64)
    if s.size < J:
        return np.pad(s, (0, J - s.size))
    return s[:J]


def w1_of(spec, F_true):
    return w1_distance(F_true, np.arange(1, J + 1, dtype=np.float64),
                       cdf_of(pad_to_J(spec)), j_eval=J)


def run_one(alpha, p, seed):
    rng = np.random.default_rng(710_000 + seed * 13 + int(p * 1000)
                                + int(alpha * 100))
    f_true = zipf_population(rng, N, VOCAB, alpha)
    F_true = truth_spectrum(f_true)
    g = thin(f_true, p, rng)
    n = int(g.sum())
    N_max = int(2 * n / p) + 1000
    d_obs = int((g >= 1).sum())
    v_true = int((f_true > 0).sum())

    logfact = np.concatenate(
        [[0.0], np.cumsum(np.log(np.arange(1, N_max + 1, dtype=np.float64)))])

    # Raw thinned spectrum, taken at face value.
    raw = fofo_spectrum(g)[:J].astype(np.float64)
    w1_raw = w1_of(raw, F_true)
    ndv_raw = d_obs

    # Deployable state. The prior is fitted from g alone.
    pf, _ve, a0, b0 = fitted_prior_eb(g, p, N_max)
    spec_fit = spectrum_ours(g, p, pf, logfact, N_max, d_obs)[:J]
    w1_fit = w1_of(spec_fit, F_true)
    w_fit = missing_fraction(pf, p)
    ndv_fit = d_obs / (1.0 - w_fit) if 0.0 < w_fit < 1.0 else float(d_obs)

    # Oracle ceiling. The prior comes from the full population spectrum.
    pf_o = oracle_prior(F_true, N_max)
    spec_o = spectrum_ours(g, p, pf_o, logfact, N_max, d_obs)[:J]
    w1_oracle = w1_of(spec_o, F_true)

    return {
        "alpha": alpha, "p": p, "seed": seed,
        "d_obs": d_obs, "ndv_true": v_true,
        "w1_raw": w1_raw, "w1_fit": w1_fit, "w1_oracle": w1_oracle,
        "ratio_fit_raw": w1_fit / w1_raw if w1_raw > 0 else float("nan"),
        "ndv_rel_raw": abs(ndv_raw - v_true) / v_true,
        "ndv_rel_fit": abs(ndv_fit - v_true) / v_true,
        "a0": float(a0), "b0": float(b0),
        "bias_scale": (1.0 - p) * float(a0) / (float(b0) + p),
        "shrink_coef": (1.0 - p) * float(b0) / (float(b0) + p),
        "miss_frac": w_fit,
    }


def aggregate(rows):
    agg = {}
    for a in ALPHAS:
        for p in RATES:
            sub = [r for r in rows
                   if abs(r["alpha"] - a) < 1e-9 and abs(r["p"] - p) < 1e-9]
            if not sub:
                continue
            rec = {"n": len(sub)}
            for k in ("w1_raw", "w1_fit", "w1_oracle", "ratio_fit_raw",
                      "ndv_rel_raw", "ndv_rel_fit", "a0", "b0",
                      "bias_scale", "shrink_coef", "miss_frac"):
                rec[k] = float(np.median([r[k] for r in sub]))
            agg[f"{a}|{p}"] = rec
    return agg


def emit_tex(agg):
    L = []
    L.append("\\begin{table*}[t]")
    L.append("\\centering")
    L.append("\\caption{Mechanism of the raw-spectrum advantage in the synthetic grid of the "
             "main text. Zipf populations, $N=10^{5}$ over $3\\times10^{4}$ candidate values, "
             "three seeds, median per cell. $\\Wone$ is computed on spectra "
             "renormalized over $j\\le512$ and therefore measures shape only. "
             "The NDV columns report relative error in the estimated number of "
             "population types. Oracle builds its prior from the full population "
             "spectrum and is an information ceiling, not a deployable estimator.}")
    L.append("\\label{tab:table2-mechanism}")
    L.append("\\footnotesize")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append("\\begin{tabular}{llcccccc}")
    L.append("\\toprule")
    L.append("$\\alpha$ & $p$ & $\\Wone$ raw & $\\Wone$ fitted & $\\Wone$ oracle "
             "& NDV err raw & NDV err fitted & $(1-p)a_0/(b_0+p)$ \\\\")
    L.append("\\midrule")
    for a in ALPHAS:
        for p in RATES:
            r = agg.get(f"{a}|{p}")
            if r is None:
                continue
            L.append(
                f"{a:.1f} & {p:.2f} & {r['w1_raw']:.2f} & {r['w1_fit']:.2f} "
                f"& {r['w1_oracle']:.2f} & {r['ndv_rel_raw']:.3f} "
                f"& {r['ndv_rel_fit']:.3f} & {r['bias_scale']:.3f} \\\\")
        if a != ALPHAS[-1]:
            L.append("\\midrule")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table*}")
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(L) + "\n")


def emit_report(rows, agg):
    L = []
    L.append("# Table II mechanism analysis")
    L.append("")
    L.append("Generated by `experiments/code/analyze_table2_mechanism.py`.")
    L.append("All numbers below are medians over three seeds of the values in `summary.json`.")
    L.append("")
    L.append("## Reading of the result")
    L.append("")
    L.append("W1 renormalizes the spectrum over j <= 512, so it measures the shape of the")
    L.append("frequency distribution and ignores the number of types. The raw thinned")
    L.append("spectrum is a sparse copy of the truth and keeps a similar normalized shape")
    L.append("for Zipf populations, but it loses most types. The fitted inverse restores")
    L.append("the missing types with a prior, which shifts mass into the singleton bin and")
    L.append("moves the normalized shape away from the truth at low rates. The table")
    L.append("reports both quantities so the trade-off is visible.")
    L.append("")
    L.append("| alpha | p | W1 raw | W1 fitted | W1 oracle | NDV err raw | NDV err fitted | bias scale |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in ALPHAS:
        for p in RATES:
            r = agg.get(f"{a}|{p}")
            if r is None:
                continue
            L.append(f"| {a:.1f} | {p:.2f} | {r['w1_raw']:.3f} | {r['w1_fit']:.3f} "
                     f"| {r['w1_oracle']:.3f} | {r['ndv_rel_raw']:.4f} "
                     f"| {r['ndv_rel_fit']:.4f} | {r['bias_scale']:.3f} |")
    L.append("")
    L.append("## Crossing point")
    L.append("")
    for a in ALPHAS:
        wins = [f"{p:.2f}" for p in RATES
                if agg.get(f"{a}|{p}", {}).get("ratio_fit_raw", 9) < 1.0]
        L.append(f"- alpha={a:.1f}, fitted beats raw at p in "
                 f"{', '.join(wins) if wins else 'none of the tested rates'}.")
    L.append("")
    L.append("The oracle column uses the full population spectrum as its prior. It is an")
    L.append("information ceiling and must never be presented as a recovery result.")
    L.append("")
    OUT_REPORT.write_text("\n".join(L) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in ALPHAS:
        for p in RATES:
            for seed in SEEDS:
                try:
                    rows.append(run_one(a, p, seed))
                    print("ok", a, p, seed, flush=True)
                except Exception as exc:  # noqa: BLE001
                    print("FAIL", a, p, seed, repr(exc), flush=True)
    agg = aggregate(rows)
    OUT_JSON.write_text(json.dumps(
        {"config": {"alphas": ALPHAS, "rates": RATES, "seeds": SEEDS,
                    "N": N, "vocab": VOCAB, "J": J},
         "rows": rows, "agg": agg}, indent=1))
    emit_tex(agg)
    emit_report(rows, agg)
    print("saved", OUT_JSON, OUT_REPORT, OUT_TEX)


if __name__ == "__main__":
    main()
