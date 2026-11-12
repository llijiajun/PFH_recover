"""Draw spectrum-only diagnostic panels from the archived per-case metrics.

Every panel is a property of the recovered histogram. Output goes to
experiments/results/figures/. Run from any directory.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import d4d_logic as D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE = os.path.join(ROOT, "experiments/results/baseline_validation/metrics.csv")
MOD = os.path.join(ROOT, "experiments/results/modern_baselines/modern_metrics.csv")
OUT = os.path.join(ROOT, "experiments/results/figures")
os.makedirs(OUT, exist_ok=True)


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def col(rows, key):
    out = []
    for r in rows:
        try:
            v = float(r[key])
        except (KeyError, ValueError, TypeError):
            v = np.nan
        out.append(v if np.isfinite(v) else np.nan)
    return np.array(out)


def med(rows, key):
    v = col(rows, key)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def frac_below(rows, key, thr):
    v = col(rows, key)
    v = v[np.isfinite(v)]
    return float(np.mean(v <= thr)) if v.size else np.nan


base = load(BASE)
mod = load(MOD)

fitted = {(r["dataset"], r["attribute"], r["p"]): r for r in mod}
eb_w1 = np.array([
    float(fitted[(r["dataset"], r["attribute"], r["p"])]["eb_fit_w1"])
    if (r["dataset"], r["attribute"], r["p"]) in fitted else np.nan
    for r in base
])
eb_tv = np.array([
    float(fitted[(r["dataset"], r["attribute"], r["p"])]["eb_fit_tv"])
    if (r["dataset"], r["attribute"], r["p"]) in fitted else np.nan
    for r in base
])
eb_rmse = np.array([
    float(fitted[(r["dataset"], r["attribute"], r["p"])]["eb_fit_rmse"])
    if (r["dataset"], r["attribute"], r["p"]) in fitted else np.nan
    for r in base
])

plt.rcParams.update(D.PAPER_RCPARAMS)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "path"})
plt.rcParams.update({"ytick.labelsize": 7.2, "xtick.labelsize": 7.6,
                     "axes.labelsize": 8.5, "axes.titlesize": 8.8,
                     "legend.fontsize": 6.8})


def save_vector(fig, stem):
    """Vector first.  PDF, EPS and SVG are the deliverables.

    No bbox_inches here.  The PDF bounding box is part of the published layout
    and must stay byte-for-byte stable when the script is rerun.
    """
    for ext in ("pdf", "eps", "svg"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"wrote {stem}.pdf / .eps / .svg")

# ----------------------------------------------------------------- fig_baselines
# Method specs: (label, base_key|None, mod_key|None, color).
# A None base_key means the value lives in the modern_baselines file (the fitted variant).
rows = [
    ("Naive $1/p$", "naive_w1", None, D.CB["none"]),
    ("Good--Turing", "good_turing_w1", None, D.CB["anchor"]),
    ("Chebyshev", "chebyshev_w1", None, D.CB["none"]),
    ("Ours (fitted)", None, "eb_fit_w1", D.CB["ours2"]),
    ("Ours (oracle)", "ours_w1", None, D.CB["ours"]),
]
labels = [r[0] for r in rows]
colors = [r[3] for r in rows]


def src_and_key(bk, mk):
    if bk is not None:
        return base, bk
    return mod, mk


def med_in(bk, mk, pval=None):
    src, key = src_and_key(bk, mk)
    if pval is None:
        sub = src
    else:
        sub = [r for r in src if r["p"] == f"{pval}"]
    return med(sub, key)


PVALS = [0.1, 0.3, 0.5]

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.15),
                         gridspec_kw={"height_ratios": [1.15, 1.0]})

# Row 1: spectrum distance across sampling rates (the multi-layer comparison).
for j, pv in enumerate(PVALS):
    ax = axes[0, j]
    vals = [med_in(r[1], r[2], pv) for r in rows]
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("median $W_1$")
    ax.set_title(f"({chr(ord('a') + j)}) $p={pv}$")

# Row 2: per-case profile, total variation, RMSE.
ax = axes[1, 0]
thr = np.logspace(-4, 2, 60)
curves = [
    ("Naive $1/p$", "naive_w1", D.CB["none"], ":"),
    ("Good--Turing", "good_turing_w1", D.CB["anchor"], "--"),
    ("Ours (oracle)", "ours_w1", D.CB["ours"], "-"),
]
for name, key, c, ls in curves:
    y = [frac_below(base, key, t) for t in thr]
    ax.plot(thr, y, color=c, linestyle=ls, label=name, linewidth=1.2)
y_fit = [float(np.mean(eb_w1[np.isfinite(eb_w1)] <= t)) for t in thr]
ax.plot(thr, y_fit, color=D.CB["ours2"], linestyle="-.", linewidth=1.2,
        label="Ours (fitted)")
ax.set_xscale("log")
ax.set_xlabel("threshold on $W_1$")
ax.set_ylabel("fraction of cases")
ax.set_title("(d) per-case profile")
ax.set_ylim(0, 1.02)
ax.legend(frameon=True, framealpha=0.85, loc="upper left", borderpad=0.3,
          handlelength=1.6, fontsize=6.0)

def med_metric(row, suffix):
    """Median of the TV or RMSE column that lives in the same file as row's W1 column."""
    bk, mk = row[1], row[2]
    if bk is not None:
        return med(base, bk.replace("_w1", suffix))
    return med(mod, mk.replace("_w1", suffix))


ax = axes[1, 1]
vals_tv = [med_metric(r, "_tv") for r in rows]
ax.barh(range(len(vals_tv)), vals_tv, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(vals_tv)))
ax.set_yticklabels(labels)
ax.set_xscale("log")
ax.invert_yaxis()
ax.set_xlabel("median TV")
ax.set_title("(e) total variation")

ax = axes[1, 2]
vals_rmse = [med_metric(r, "_rmse") for r in rows]
ax.barh(range(len(vals_rmse)), vals_rmse, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(vals_rmse)))
ax.set_yticklabels(labels)
ax.set_xscale("log")
ax.invert_yaxis()
ax.set_xlabel("median RMSE")
ax.set_title("(f) RMSE")

fig.tight_layout(pad=0.35, w_pad=0.6, h_pad=0.5)
save_vector(fig, "fig_baselines")

# ------------------------------------------------------------------- fig_modern
mod_rows = [
    ("Good--Turing", "good_turing_w1", "good_turing_func_self_join", "good_turing_func_entropy", D.CB["anchor"]),
    ("NPMLE", "npmle_w1", "npmle_func_self_join", "npmle_func_entropy", D.CB["none"]),
    ("SB linear", "smooth_bayes_w1", "smooth_bayes_func_self_join", "smooth_bayes_func_entropy", D.CB["accent"]),
    ("Ours (fitted)", "eb_fit_w1", "eb_fit_func_self_join", "eb_fit_func_entropy", D.CB["ours2"]),
    ("Ours (oracle)", "ours_w1", "ours_func_self_join", "ours_func_entropy", D.CB["ours"]),
    ("Raw sample", "naive_w1", "naive_func_self_join", "naive_func_entropy", "#c8c8c8"),
]
mlabels = [m[0] for m in mod_rows]
mcolors = [m[4] for m in mod_rows]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.15), gridspec_kw={"width_ratios": [1.0, 1.35]})
ax = axes[0]
vals = [med(mod, m[1]) for m in mod_rows]
ax.barh(range(len(vals)), vals, color=mcolors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(mlabels)
ax.set_xscale("log")
ax.invert_yaxis()
ax.set_xlabel("median $W_1$")
ax.set_title("(a) spectrum distance")

ax = axes[1]
x = np.arange(len(mlabels))
w = 0.38
sj = [med(mod, m[2]) for m in mod_rows]
en = [med(mod, m[3]) for m in mod_rows]
ax.bar(x - w / 2, sj, w, color=D.CB["ours"], edgecolor="white", linewidth=0.5,
       label="self-join $\\sum_v f_v^2$")
ax.bar(x + w / 2, en, w, color=D.CB["anchor"], edgecolor="white", linewidth=0.5,
       label="entropy")
ax.set_xticks(x)
ax.set_xticklabels(mlabels, rotation=18, ha="right", fontsize=6.6)
ax.set_yscale("log")
ax.set_ylabel("median rel. error (%)")
ax.set_title("(b) downstream functionals")
ax.legend(frameon=True, framealpha=0.9, loc="lower left", borderpad=0.3, handlelength=1.4,
          fontsize=6.4)

fig.tight_layout(pad=0.35, w_pad=0.8)
save_vector(fig, "fig_modern")

print("wrote figures")
print("W1 medians:", dict(zip(mlabels, [round(v, 5) for v in vals])))
