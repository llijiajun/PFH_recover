"""Data-dependence of the Good--Turing gap (controlled recovery).

Answers the following question. is our fitted estimator's distance to
Good--Turing a property of the data rather than the method? We sweep five
population families (the same five as the unknown-support grid) across four
sampling rates and compare three estimators on the SAME thinned sample:

  raw    -- the observed frequency-of-frequencies spectrum fofo_spectrum(g),
            i.e. what the sample shows if taken at face value;
  ours   -- the closed-form inverse with a Gamma--Poisson prior fitted from the
            sample alone (deployment state);
  gt     -- the Good--Turing spectrum smoother of the observed spectrum.

We also record the fitted abundance-prior shape a0, because the earlier
low-rate FoF study showed a0 collapses to its lower bound when the observed
spectrum carries almost no doubletons, which is the mechanism that pushes the
fitted estimator away from truth.

Outputs:
  experiments/results/data_dependency/summary.json
  experiments/results/tables/tab_data_dependency.tex
  experiments/code/figures/fig_data_dependency.pdf / .png / .svg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "paper" / "figures"))

from d4d_logic import (  # noqa: E402
    zipf_population, w1_distance, analytic_posterior, fofo_spectrum, thin,
)
from d4d_baselines import spectrum_good_turing, spectrum_ours  # noqa: E402
from eb_prior import fitted_prior_eb  # noqa: E402

J = 512
FAMILIES = ["zipf", "mix", "spiked", "concentrated", "geopoor"]
RATES = [0.05, 0.1, 0.2, 0.3]
SEEDS = [0, 1, 2]

OUT_JSON = ROOT / "experiments" / "results" / "data_dependency" / "summary.json"
OUT_TEX = ROOT / "experiments" / "results" / "tables" / "tab_data_dependency.tex"
OUT_FIG = HERE / "figures" / "fig_data_dependency"


# --------------------------------------------------------------------------- #
# population generators (mirror validate_unknown_support.make_population)
# --------------------------------------------------------------------------- #
def make_population(rng, family):
    if family == "zipf":
        V, N = 30_000, 100_000
        w = 1.0 / np.power(np.arange(1, V + 1, dtype=np.float64), 1.5)
    elif family == "mix":
        V, N = 30_000, 100_000
        w = 0.7 / np.power(np.arange(1, V + 1, dtype=np.float64), 1.5)
        w += 0.3 / V
    elif family == "spiked":
        V, N = 30_000, 100_000
        w = 1.0 / np.power(np.arange(1, V + 1, dtype=np.float64), 1.5)
        w[0] += w.sum() * 0.10
    elif family == "concentrated":
        V, N = 300, 1_000_000
        w = rng.dirichlet(np.full(V, 1.0))
        return rng.multinomial(N, w / w.sum())
    elif family == "geopoor":
        V, N = 5_000, 100_000
        w = (1.0 - 0.2) ** np.arange(0, V, dtype=np.float64)
    else:
        raise ValueError(family)
    w = w / w.sum()
    return rng.multinomial(N, w)


def truth_spectrum(f_true):
    F = np.zeros(J)
    for f in f_true[f_true > 0]:
        if f <= J:
            F[f - 1] += 1.0
    return F


def ours_spectrum(g, p, N_max):
    pf, _ve, a0, b0 = fitted_prior_eb(g, p, N_max)
    logfact = np.concatenate(
        [[0.0], np.cumsum(np.log(np.arange(1, N_max + 1, dtype=np.float64)))])
    d_obs = int((g >= 1).sum())
    H = 512
    spec = spectrum_ours(g, p, pf, logfact, N_max, d_obs)
    spec = spec[:H]
    st = spec.sum()
    cdf = np.cumsum(spec / max(st, 1e-12))
    return spec, cdf, a0, b0


def w1_of(cdf_hat, F_true):
    js_hat = np.arange(1, len(cdf_hat) + 1, dtype=np.float64)
    return w1_distance(F_true, js_hat, cdf_hat, j_eval=J)


def run_one(family, p, seed):
    rng = np.random.default_rng(123_000 + seed * 7 + int(p * 1000))
    f_true = make_population(rng, family)
    F_true = truth_spectrum(f_true)
    g = thin(f_true, p, rng)
    n = int(g.sum())
    N_max = int(2 * n / p) + 1000
    d_obs = int((g >= 1).sum())

    # raw: observed FoF at face value
    raw = fofo_spectrum(g)[:J]
    raw_sum = raw.sum()
    raw_cdf = np.cumsum(raw / max(raw_sum, 1e-12)) if raw_sum > 0 else np.zeros(J)
    w1_raw = w1_of(raw_cdf, F_true) if raw_sum > 0 else float("nan")

    # ours fitted
    spec_o, cdf_o, a0, b0 = ours_spectrum(g, p, N_max)
    w1_ours = w1_of(cdf_o, F_true)

    # Good--Turing on the observed spectrum
    gt = spectrum_good_turing(fofo_spectrum(g))[:J]
    gt_sum = gt.sum()
    gt_cdf = np.cumsum(gt / max(gt_sum, 1e-12)) if gt_sum > 0 else np.zeros(J)
    w1_gt = w1_of(gt_cdf, F_true) if gt_sum > 0 else float("nan")

    ratio = (w1_ours / w1_gt) if (w1_gt and np.isfinite(w1_gt) and w1_gt > 0) else float("nan")
    return {
        "family": family, "p": p, "seed": seed,
        "d_obs": d_obs, "ndv_true": float((f_true > 0).sum()),
        "w1_raw": w1_raw, "w1_ours": w1_ours, "w1_gt": w1_gt,
        "ratio_ours_gt": ratio, "a0": a0, "b0": b0,
    }


def main():
    rows = []
    for fam in FAMILIES:
        for p in RATES:
            for seed in SEEDS:
                try:
                    rows.append(run_one(fam, p, seed))
                except Exception as e:  # noqa: BLE001
                    print("FAIL", fam, p, seed, repr(e), flush=True)
    # aggregate: median over seeds per (family, p)
    agg = {}
    for fam in FAMILIES:
        for p in RATES:
            sub = [r for r in rows if r["family"] == fam and abs(r["p"] - p) < 1e-9]
            agg[(fam, p)] = {
                "w1_raw": float(np.median([r["w1_raw"] for r in sub])),
                "w1_ours": float(np.median([r["w1_ours"] for r in sub])),
                "w1_gt": float(np.median([r["w1_gt"] for r in sub])),
                "ratio": float(np.median([r["ratio_ours_gt"] for r in sub])),
                "a0": float(np.median([r["a0"] for r in sub])),
                "d_obs": float(np.median([r["d_obs"] for r in sub])),
                "ndv_true": float(np.median([r["ndv_true"] for r in sub])),
            }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"rows": rows, "agg": {f"{k[0]}|{k[1]}": v for k, v in agg.items()}}, indent=1))
    print("saved", OUT_JSON)

    # ---- emit LaTeX table (family x rate, ratio ours/gt + a0) ----
    emit_tex(agg)
    # ---- emit figure ----
    emit_figure(agg)


def emit_tex(agg):
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Dependence of the recovery gap on the data. Five population")
    lines.append("families, sampling rate $p$, median over 3 seeds. $\\Wone$ is the")
    lines.append("full-spectrum distance ($j\\le 512$). Ours and Good--Turing both act on the")
    lines.append("same thinned sample, and raw is the observed spectrum taken at face value.")
    lines.append("A ratio below one means our fitted inverse is closer to the truth than")
    lines.append("Good--Turing. The fitted abundance-prior shape $a_0$ collapses to its lower")
    lines.append("bound ($10^{-3}$) for the pure Zipf and spiked families at every rate shown,")
    lines.append("which is the regime where the conjugate prior is not identified from the")
    lines.append("sample.}")
    lines.append("\\label{tab:data-dependency}")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l|cccc|cccc}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{4}{c|}{$\\Wone$ ours / GT / raw $\\downarrow$}")
    lines.append("& \\multicolumn{4}{c}{ours $\\div$ GT $\\downarrow$} \\\\")
    lines.append("Family & $p{=}0.05$ & $p{=}0.1$ & $p{=}0.2$ & $p{=}0.3$")
    lines.append("& $p{=}0.05$ & $p{=}0.1$ & $p{=}0.2$ & $p{=}0.3$ \\\\")
    lines.append("\\midrule")
    for fam in FAMILIES:
        def fmt_w1(p):
            a = agg[(fam, p)]
            return f"{a['w1_ours']:.2f}/{a['w1_gt']:.2f}/{a['w1_raw']:.2f}"
        def fmt_ratio(p):
            r = agg[(fam, p)]["ratio"]
            return f"{r:.2f}"
        lines.append(f"{fam} & {fmt_w1(0.05)} & {fmt_w1(0.1)} & {fmt_w1(0.2)} & {fmt_w1(0.3)}"
                     f" & {fmt_ratio(0.05)} & {fmt_ratio(0.1)} & {fmt_ratio(0.2)} & {fmt_ratio(0.3)} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print("saved", OUT_TEX)


def emit_figure(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.0))
    colors = {"zipf": "#D55E00", "mix": "#0072B2", "spiked": "#009E73",
              "concentrated": "#CC79A7", "geopoor": "#6A3D9A"}
    for fam in FAMILIES:
        ratios = [agg[(fam, p)]["ratio"] for p in RATES]
        a0s = [agg[(fam, p)]["a0"] for p in RATES]
        ax1.plot(RATES, ratios, color=colors[fam], lw=2.0, marker="o", label=fam)
        ax2.plot(RATES, a0s, color=colors[fam], lw=2.0, marker="s", label=fam)
    ax1.axhline(1.0, color="#000000", lw=1.4, ls=":")
    ax1.set_xlabel("sampling rate $p$")
    ax1.set_ylabel("ours $\\div$ Good--Turing ($W_1$)")
    ax1.set_title("(a) where the gap to Good--Turing comes from", fontsize=9.5, fontweight="bold")
    ax1.text(0.02, 0.96, "below one, our inverse wins\nabove one, Good--Turing wins",
             transform=ax1.transAxes, fontsize=8, color="#333333", va="top",
             bbox=dict(fc="white", ec="#BBBBBB", lw=0.6, alpha=0.9, pad=2.0))
    ax2.set_yscale("log")
    ax2.axhline(1e-3, color="#000000", lw=1.0, ls=":")
    ax2.set_xlabel("sampling rate $p$")
    ax2.set_ylabel("fitted abundance-prior $a_0$ (log)")
    ax2.set_title("(b) prior shape is identified only for some families", fontsize=9.5, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7.6, frameon=True, framealpha=0.9, ncol=1)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(str(OUT_FIG) + "." + ext, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", str(OUT_FIG) + ".{pdf,png,svg}")


if __name__ == "__main__":
    main()
