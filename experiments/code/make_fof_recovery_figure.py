"""FoF recovery at low retention with an honest, sample-only prior.

The ground truth is F_j, the number of distinct values whose frequency in the
FULL population equals j.  Every plotted curve estimates that same object, so
all curves share one count axis and one metric space.

Two arms of the analytic inverse are shown.  The deployment arm builds its
prior from the thinned sample alone with fitted_prior_eb.  The ceiling arm
builds its prior from the full population with prior_from_truth.  The ceiling
arm sees the answer in its prior and is not a recovery result.  Earlier
versions of this figure showed only the ceiling arm, which produced a nearly
perfect overlap at retention rates down to 0.01.  That overlap was a
self-consistency check rather than evidence of recovery.

The FoF of the thinned sample counts types by SAMPLED frequency g.  Its count
axis is g rather than f and its support is a fraction of the truth.  It is not
an estimator of F_j and is excluded from every panel and every table.

The ground truth is drawn last, on top of every estimate, as a thick black
line with hollow circles.  A thin line underneath an overlapping estimate is
invisible, which was the second defect of the earlier figure.

Outputs are vector first.  PDF, EPS and SVG are the deliverables and PNG is a
preview only.

  experiments/results/figures/fof_recovery_comparison.{pdf,eps,svg,png}
  experiments/results/figures/fof_recovery_deviation.{pdf,eps,svg,png}
  experiments/results/fof_recovery_figure/{summary.json,report.md}

Run from anywhere with the py311 interpreter.  No torch needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from d4d_logic import zipf_population, thin, fofo_spectrum
from d4d_metrics import prior_from_truth
from d4d_baselines import (spectrum_naive, spectrum_good_turing,
                           spectrum_chebyshev, spectrum_ours,
                           distribution_metrics, H_CAP)
from d4d_modern_baselines import smooth_bayes_params, spectrum_smooth_bayes
from eb_prior import fitted_prior_eb

FIG_DIR = ROOT / "experiments" / "results" / "figures"
RES_DIR = ROOT / "experiments" / "results" / "fof_recovery_figure"

CASES = [
    dict(alpha=1.5, n=1_000_000, vocab=100_000, p=0.01, seed=11,
         label="Zipf alpha 1.5, N 1.0M, p 0.01"),
    dict(alpha=1.5, n=300_000, vocab=60_000, p=0.02, seed=3,
         label="Zipf alpha 1.5, N 300k, p 0.02"),
    dict(alpha=1.3, n=100_000, vocab=30_000, p=0.02, seed=0,
         label="Zipf alpha 1.3, N 100k, p 0.02"),
    dict(alpha=1.8, n=100_000, vocab=30_000, p=0.05, seed=0,
         label="Zipf alpha 1.8, N 100k, p 0.05"),
]

FITTED = "Ours (fitted prior)"
ORACLE = "Ours (oracle prior)"
GT = "Good-Turing"
NAIVE = "Naive 1/p"
CHEB = "Chebyshev"
SB = "SB linear (fitted prior)"

METHOD_ORDER = [NAIVE, GT, CHEB, SB, FITTED, ORACLE]
# The oracle arm is drawn in every comparison panel.  It nearly overlaps the
# ground-truth curve, so it is kept above that curve with a thin grey dash.
# Both the curve and its legend entry make the overlap visible to the reader.
ESTIMATE_ORDER = [GT, NAIVE, CHEB, SB, FITTED, ORACLE]

# Hue AND dash pattern differ between every pair.  The oracle prior is grey
# and dashed so it cannot be mistaken for a deployed estimator.
STYLE = {
    FITTED: dict(color="#D55E00", lw=2.6, marker="o", ms=3.0, zorder=12),
    ORACLE: dict(color="#8C8C8C", lw=1.7, ls="--", zorder=32),
    GT: dict(color="#0072B2", lw=1.7, marker="s", ms=2.4, zorder=5),
    NAIVE: dict(color="#CC79A7", lw=1.7, ls="--", marker="^", ms=3.2, zorder=4),
    CHEB: dict(color="#009E73", lw=1.7, ls="-.", zorder=4),
    SB: dict(color="#6A3D9A", lw=1.7, ls=":", marker="D", ms=3.0, zorder=4),
}

LEGEND_NAME = {
    FITTED: "Ours, prior fitted from sample",
    ORACLE: "Ours, oracle prior",
    GT: "Good-Turing",
    NAIVE: "Naive 1/p",
    CHEB: "Chebyshev",
    SB: "SB linear (fitted prior)",
}

TRUTH_LINE = dict(color="#000000", lw=2.4, zorder=30)
TRUTH_MARKER = dict(color="#000000", marker="o", ms=5.6, mfc="white",
                    mec="#000000", mew=1.5, ls="none", zorder=31)

CASE_COLORS = ["#D55E00", "#0072B2", "#009E73", "#6A3D9A"]
SWEEP_P = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
J_SHOW = 32
J_RATIO = 16

VECTOR_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
}


def save_vector(fig, out_base: Path):
    """Write vector files and a PNG preview for local inspection."""
    for ext in ("pdf", "eps", "svg"):
        fig.savefig(out_base.with_suffix("." + ext),
                    bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".png"), dpi=180,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run_case(c: dict) -> tuple[dict, dict, np.ndarray]:
    """Compute the true FoF spectrum and every method's estimate for one case."""
    rng = np.random.default_rng(c["seed"])
    counts = zipf_population(rng, c["n"], c["vocab"], c["alpha"])
    f_true = counts[counts > 0].astype(np.int64)
    g = thin(f_true, c["p"], rng)
    f_samp = fofo_spectrum(g)
    F_true = fofo_spectrum(f_true, j_max=H_CAP)
    d_obs = int((g >= 1).sum())
    N_total = int(f_true.sum())

    # Deployment prior.  Built from the thinned sample only.  No truth input.
    N_max = max(int(g.sum() / c["p"]) + 64, d_obs + 64)
    pf_fit, _, a0_eb, b0_eb = fitted_prior_eb(g, c["p"], N_max)
    lf_fit = gammaln(np.arange(N_max + 513) + 1.0)

    # Ceiling prior.  Built from the full population.  This sees the answer.
    pf_orc, lf_orc = prior_from_truth(f_true, N_total)
    a0_sb, b0_sb = smooth_bayes_params(f_samp, c["p"], int(g.sum()))

    specs = {
        NAIVE: spectrum_naive(g, c["p"]),
        GT: spectrum_good_turing(f_samp),
        CHEB: spectrum_chebyshev(f_samp),
        SB: spectrum_smooth_bayes(g, c["p"], a0_sb, b0_sb, d_obs),
        FITTED: spectrum_ours(g, c["p"], pf_fit, lf_fit, N_max, d_obs),
        ORACLE: spectrum_ours(g, c["p"], pf_orc, lf_orc, N_total, d_obs),
    }
    metrics = {k: distribution_metrics(F_true, v) for k, v in specs.items()}

    out = {
        "label": c["label"], **{k: c[k] for k in ("alpha", "n", "vocab", "p", "seed")},
        "N_total": N_total,
        "n_types_true": int(f_true.size),
        "d_obs": d_obs,
        "observed_type_fraction": float(d_obs / max(F_true.sum(), 1.0)),
        "N_max_fitted_prior": N_max,
        "eb_prior_a0": float(a0_eb),
        "eb_prior_b0": float(b0_eb),
        "sb_prior_a0": float(a0_sb),
        "sb_prior_b0": float(b0_sb),
        "F_true_head": [float(v) for v in F_true[:16]],
        "specs_head": {k: [float(v) for v in s[:16]] for k, s in specs.items()},
        "metrics": metrics,
        "F1_true": float(F_true[0]),
        "ndv_true_capped": float(F_true.sum()),
        "ndv_rel": {k: (float(s.sum()) - float(F_true.sum())) / float(F_true.sum())
                    for k, s in specs.items()},
        "tv_fitted": metrics[FITTED]["tv"],
        "tv_oracle": metrics[ORACLE]["tv"],
    }
    return out, specs, F_true


def run_sweep() -> list[dict]:
    """TV against retention rate on one fixed population."""
    rng = np.random.default_rng(3)
    counts = zipf_population(rng, 300_000, 60_000, 1.5)
    f_true = counts[counts > 0].astype(np.int64)
    F_true = fofo_spectrum(f_true, j_max=H_CAP)
    N_total = int(f_true.sum())
    pf_orc, lf_orc = prior_from_truth(f_true, N_total)
    rows = []
    for i, p in enumerate(SWEEP_P):
        r = np.random.default_rng(3000 + i)
        g = thin(f_true, p, r)
        d_obs = int((g >= 1).sum())
        f_samp = fofo_spectrum(g)
        N_max = max(int(g.sum() / p) + 64, d_obs + 64)
        pf_fit, _, a0_eb, b0_eb = fitted_prior_eb(g, p, N_max)
        lf_fit = gammaln(np.arange(N_max + 513) + 1.0)
        fit = spectrum_ours(g, p, pf_fit, lf_fit, N_max, d_obs)
        orc = spectrum_ours(g, p, pf_orc, lf_orc, N_total, d_obs)
        gt = spectrum_good_turing(f_samp)
        rows.append(dict(
            p=p, d_obs=d_obs,
            observed_type_fraction=float(d_obs / F_true.sum()),
            tv_fitted=distribution_metrics(F_true, fit)["tv"],
            tv_oracle=distribution_metrics(F_true, orc)["tv"],
            tv_good_turing=distribution_metrics(F_true, gt)["tv"],
            eb_prior_a0=float(a0_eb), eb_prior_b0=float(b0_eb),
        ))
    return rows


def _curve(a, J=J_SHOW):
    """Absolute counts, with exact zeros hidden instead of faked as a floor."""
    v = np.asarray(a, dtype=float)[:J].copy()
    v[v <= 0] = np.nan
    return v


def plot_comparison(results, specs_by_case, truth_by_case, out_base: Path):
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.color": "#DDDDDD", "grid.alpha": 1.0,
                         "axes.spines.top": False, "axes.spines.right": False})
    plt.rcParams.update(VECTOR_RC)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)
    x = np.arange(1, J_SHOW + 1)
    for ax, r, specs, F_true in zip(axes.ravel(), results, specs_by_case, truth_by_case):
        for m in ESTIMATE_ORDER:
            ax.plot(x, _curve(specs[m]), label=LEGEND_NAME[m], **STYLE[m])
        # Ground truth is drawn LAST and sits above every estimate.  The thick
        # line and hollow circles stay visible even where a curve overlaps it.
        ax.plot(x, _curve(F_true), label="Ground truth F_j (full population)",
                **TRUTH_LINE)
        ax.plot(x, _curve(F_true), **TRUTH_MARKER)
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
        ax.set_xlim(1, J_SHOW)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 32])
        ax.set_title(r["label"], fontsize=9.5, fontweight="bold")
        ax.text(0.97, 0.95,
                f"observed types {r['observed_type_fraction']:.1%} of truth\n"
                f"fitted TV {r['tv_fitted']:.3f}  |  oracle TV {r['tv_oracle']:.3f}"
                "\nThe oracle estimate nearly overlaps the ground-truth curve.",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.7,
                color="#222222", linespacing=1.35,
                bbox=dict(fc="white", ec="#BBBBBB", lw=0.6, pad=2.0))
    for ax in axes[1]:
        ax.set_xlabel("true frequency j in the full population")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"FoF  $F_j$  (number of types)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles, labels = list(handles), list(labels)
    k = labels.index("Ground truth F_j (full population)")
    handles[k] = Line2D([], [], color="#000000", lw=2.4, marker="o", ms=5.6,
                        mfc="white", mec="#000000", mew=1.5)
    order = [k] + [labels.index(LEGEND_NAME[m])
                   for m in [FITTED, GT, NAIVE, CHEB, SB, ORACLE]]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.957), fontsize=8.6,
               columnspacing=1.6, handlelength=2.6)
    fig.suptitle("FoF recovery at low retention against the full-population ground truth",
                 y=0.987, fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.865))
    save_vector(fig, out_base)


def plot_deviation(results, specs_by_case, truth_by_case, sweep, out_base: Path):
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.color": "#DDDDDD", "grid.alpha": 1.0,
                         "axes.spines.top": False, "axes.spines.right": False})
    plt.rcParams.update(VECTOR_RC)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1))
    x = np.arange(1, J_SHOW + 1)

    ax = axes[0]
    all_ratios = []
    for idx, (r, specs, F_true) in enumerate(zip(results, specs_by_case, truth_by_case)):
        t = np.asarray(F_true, dtype=float)[:J_RATIO]
        ok = t > 0
        ratio = np.full(J_RATIO, np.nan)
        ratio[ok] = np.asarray(specs[FITTED], dtype=float)[:J_RATIO][ok] / t[ok]
        all_ratios.append(ratio)
        ax.plot(x[:J_RATIO], ratio, color=CASE_COLORS[idx], lw=1.9,
                label=r["label"].replace("Zipf ", ""))
        ro = np.full(J_RATIO, np.nan)
        ro[ok] = np.asarray(specs[ORACLE], dtype=float)[:J_RATIO][ok] / t[ok]
        ax.plot(x[:J_RATIO], ro, color="#8C8C8C", lw=1.1, ls="--")
    ax.axhline(1.0, color="#000000", lw=1.6, ls=":")
    hi = float(np.nanmax(np.concatenate(all_ratios)))
    ax.set_ylim(0, min(4.0, max(1.6, hi * 1.05)))
    ax.set_xlim(1, J_RATIO)
    ax.set_xlabel("true frequency j in the full population")
    ax.set_ylabel(r"ratio  $\hat F_j / F_j$")
    ax.set_title("(a) distance from truth, one at perfect recovery", fontsize=9.5,
                 fontweight="bold")
    ax.text(0.02, 0.04, "solid coloured, fitted prior\ngrey dashed, oracle prior\n"
            "j capped at 16, larger j has small F_j",
            transform=ax.transAxes, fontsize=7.6, color="#333333",
            bbox=dict(fc="white", ec="#BBBBBB", lw=0.6, pad=2.0))
    ax.legend(loc="upper right", fontsize=7.2, frameon=True, framealpha=1.0,
              ncol=1, handlelength=1.8)

    ax = axes[1]
    p = np.array([s["p"] for s in sweep])
    ax.plot(p, [s["tv_fitted"] for s in sweep], color="#D55E00", lw=2.4,
            marker="o", ms=4.5, label="Ours, fitted prior")
    ax.plot(p, [s["tv_oracle"] for s in sweep], color="#8C8C8C", lw=1.8,
            ls="--", marker="s", ms=4.0, label="Ours, oracle prior")
    ax.plot(p, [s["tv_good_turing"] for s in sweep], color="#0072B2", lw=1.8,
            ls="-.", marker="^", ms=4.0, label="Good-Turing")
    ax.set_xscale("log")
    ax.set_ylim(0, 0.58)
    ax.set_xlabel("retention rate p of the thinning channel")
    ax.set_ylabel("total variation of the normalized FoF")
    ax.set_title("(b) recovery degrades as the channel gets sparser", fontsize=9.5,
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.6, frameon=True, framealpha=1.0)
    ax2 = ax.twinx()
    ax2.plot(p, [s["observed_type_fraction"] for s in sweep], color="#BBBBBB",
             lw=1.2, ls=":", marker="D", ms=3.2)
    ax2.set_ylim(0, 0.32)
    ax2.set_ylabel("observed types as a fraction of truth", color="#777777",
                   fontsize=8)
    ax2.tick_params(axis="y", labelsize=7.5, colors="#777777")
    ax2.grid(False)

    fig.suptitle("Why a near-perfect overlap at low retention is not a recovery result",
                 y=0.99, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_vector(fig, out_base)


def write_report(results, sweep):
    lines = [
        "# FoF recovery against the full-population ground truth",
        "",
        "## What the ground truth is",
        "",
        "F_j is the number of distinct values whose frequency in the FULL",
        "population equals j.  Every method in the table estimates that same",
        "object, so all curves share one count axis and one metric space.",
        "",
        "The FoF of the thinned sample counts distinct values whose SAMPLED",
        "frequency equals j.  Its count axis is g rather than f and its support",
        "is a fraction of the truth.  It is not an estimator of F_j and is",
        "therefore excluded from both the figure and the table.",
        "",
        "## Why the previous figure was wrong",
        "",
        "The earlier figure showed only an Ours arm that used prior_from_truth.",
        "That function builds the empirical prior from the full-population",
        "frequencies, so the estimator received the true histogram through its",
        "prior.  Its near-perfect overlap with the truth at retention rates down",
        "to 0.01 was a self-consistency check of the posterior formula.  It was",
        "not a recovery from the sample.  The ground truth was also a thin black",
        "line drawn underneath the thick vermillion curve that overlapped it,",
        "which made the truth effectively invisible.",
        "",
        "This version fixes both defects.  The ground truth is the last artist",
        "drawn, a thick black line with hollow circles above every estimate.",
        "The deployed Ours arm uses fitted_prior_eb, which sees the thinned",
        "sample only.  The oracle arm is kept as a clearly labelled ceiling in",
        "the deviation figure, where a ratio axis makes the gap between an",
        "information ceiling and a recovery visible.",
        "",
        "## Setting",
        "",
        "Retention rates are 0.01 to 0.05.  Only 5 to 19 percent of the true",
        "distinct values leave any observation.  W1, TV and RMSE are computed on",
        "normalized truncated spectra.  The NDV relative error uses absolute",
        "counts and is signed as estimate minus truth divided by truth.",
        "",
        "At these rates the Gamma-Poisson fit of fitted_prior_eb reaches its",
        "lower bound on the shape parameter in every case.  The observed",
        "spectrum carries almost no doubletons, so the prior shape is not",
        "identified by the sample.  This is the dominant error source in the",
        "deployment arm and the reason the recovered F_1 sits far below truth.",
        "The paper evaluates real columns at retention rates of 0.1 and above.",
        "These synthetic cases are deliberately sparser and sit outside that",
        "regime, so the numbers here are a boundary study and not the pooled",
        "accuracy reported in the paper.",
        "",
        "| case | method | W1 (lower better) | TV | RMSE | NDV rel err |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        for m in METHOD_ORDER:
            d = r["metrics"][m]
            lines.append(
                f"| {r['label']} | {m} | {d['w1']:.3f} | {d['tv']:.4f} | "
                f"{d['rmse']:.5f} | {r['ndv_rel'][m]:+.1%} |")
    lines += [
        "",
        "## Reading the comparison figure",
        "",
        "The thick black curve with hollow circles is the ground-truth FoF.",
        "The thick vermillion curve is the posterior-mean FoF from the analytic",
        "inverse with a prior fitted from the thinned sample.  It sits well",
        "below the truth at small j and carries too much mass in the tail.",
        "The oracle ceiling is not drawn in these panels because it lies on",
        "top of the truth line and would be an invisible legend entry.  The",
        "deviation figure shows fitted and oracle side by side on a ratio axis.",
        "",
        "Good-Turing underestimates F_1 by a large factor and produces isolated",
        "spikes at higher j.  Chebyshev returns a nearly flat curve.  Naive 1/p",
        "moves observed counts to j about 1/p, which lies outside the displayed",
        "range for p at or below 0.02.  SB linear with a fitted prior collapses",
        "almost all mass into the first bin.",
        "",
        "## Reading the deviation figure",
        "",
        "Panel (a) divides each estimate by the truth at every j.  A value of",
        "one is perfect recovery.  The fitted arm stays far below one at the",
        "head of the spectrum while the oracle ceiling sits on the line.",
        "Panel (b) sweeps the retention rate on one fixed population.  The",
        "fitted arm degrades steadily as p falls while the oracle ceiling stays",
        "near 0.01 TV, which is only possible because it reads the truth prior.",
        "",
        "| p | observed types | TV fitted | TV oracle | TV Good-Turing |",
        "|---|---|---|---|---|",
    ]
    for s in sweep:
        lines.append(
            f"| {s['p']:.3f} | {s['observed_type_fraction']:.1%} | "
            f"{s['tv_fitted']:.4f} | {s['tv_oracle']:.4f} | "
            f"{s['tv_good_turing']:.4f} |")
    lines += [
        "",
        "## Case-level truth and recovery at the head of the spectrum",
        "",
    ]
    for r in results:
        head = r["F_true_head"][:6]
        ours = r["specs_head"][FITTED][:6]
        orc = r["specs_head"][ORACLE][:6]
        lines += [
            f"### {r['label']}",
            "",
            f"Observed distinct values {r['d_obs']} out of "
            f"{r['ndv_true_capped']:.0f} true ones "
            f"({r['observed_type_fraction']:.1%}).",
            "",
            "F_1 to F_6 truth  " + " ".join(f"{v:.0f}" for v in head),
            "",
            "F_1 to F_6 fitted " + " ".join(f"{v:.1f}" for v in ours),
            "",
            "F_1 to F_6 oracle " + " ".join(f"{v:.1f}" for v in orc),
            "",
        ]
    (RES_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    results, specs_by_case, truth_by_case = [], [], []
    for c in CASES:
        r, specs, F_true = run_case(c)
        results.append(r)
        specs_by_case.append(specs)
        truth_by_case.append(F_true)
    sweep = run_sweep()

    plot_comparison(results, specs_by_case, truth_by_case,
                    FIG_DIR / "fof_recovery_comparison")
    plot_deviation(results, specs_by_case, truth_by_case, sweep,
                   FIG_DIR / "fof_recovery_deviation")

    payload = {
        "note": ("ground truth is F_j on the full-population f axis; the "
                 "observed-sample FoF lives on the g axis and is excluded; "
                 "Ours (fitted prior) is the deployment arm and sees the "
                 "sample only; Ours (oracle prior) is a ceiling whose prior "
                 "is built from the full population and is NOT a recovery "
                 "result; p <= 0.05; outputs are PDF EPS SVG vector plus PNG preview"),
        "cases": results,
        "retention_sweep": sweep,
    }
    (RES_DIR / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    write_report(results, sweep)

    for stem in ("fof_recovery_comparison", "fof_recovery_deviation"):
        print("wrote", FIG_DIR / f"{stem}.pdf / .eps / .svg / .png")
    print("wrote", RES_DIR / "summary.json")
    print("wrote", RES_DIR / "report.md")
    for r in results:
        print(f"  {r['label']:<32} obs {r['observed_type_fraction']:>5.1%}"
              f"  fitted TV {r['tv_fitted']:.4f}"
              f"  oracle TV {r['tv_oracle']:.4f}"
              f"  fitted NDV {r['ndv_rel'][FITTED]:+.1%}")


if __name__ == "__main__":
    main()
