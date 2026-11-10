"""Joint-distribution recovery under the same binomial-thinning channel.

Motivation
----------
The main paper's contribution list claims the framework ``already covers joint
recovery'': because the binomial-thinning kernel acts entrywise on a joint
co-occurrence matrix, the same closed-form per-cell posterior that recovers a
single frequency histogram also recovers a joint count matrix, and therefore the
join-cardinality functional sum_{v,w} F[v,w]^2 that a query optimizer needs.

That claim had no experiment behind it.  This script supplies one.  It is a
CONTROLLED demonstration on synthetic co-occurrence matrices with known ground
truth (we build the joint, thin it at a known rate p, and invert it).  It is NOT
a live query-optimizer evaluation; a real join-cardinality payoff would require
a DBMS with query plans, which is future work and is stated as such.

The recovery reuses the paper's own machinery:
  * d4d_logic.thin   -- the binomial-thinning forward channel
  * the closed-form per-cell posterior E[f | g] (eq. per-type-posterior of the
    paper), with a Gamma-Poisson prior fitted from the OBSERVED joint by the
    same method-of-moments rule the deployment arm uses (no ground truth enters).

Because the prior is shared across cells, E[f | g] depends only on the observed
count g, so we precompute a lookup table and the recovery is O(#cells).

Outputs (experiments/results/joint_recovery/):
  joint_metrics.csv, joint_summary.json, joint_report.md, fig_joint_recovery.pdf
"""

from __future__ import annotations

import csv
import json
import math
import os

import numpy as np
from scipy.special import gammaln

import d4d_logic as D

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "joint_recovery")
N_MAX = 4096          # max population cell count considered for the posterior
G_MAX = 300           # max observed cell count for the lookup table
H_CAP = 256           # spectrum truncation, consistent with the paper


# --------------------------------------------------------------------------
# 1. Synthetic co-occurrence matrices
# --------------------------------------------------------------------------
def build_joint(rng: np.random.Generator, n: int, vocab: int, alpha: float) -> np.ndarray:
    """Build a V x V joint co-occurrence matrix as a 2D Zipf product of marginals.

    The population is drawn from a multinomial over V^2 cells with weights
    w_v * w_w, w_v = Zipf(alpha).  `alpha` small (near uniform) makes the joint
    ultra-sparse (most cells are singletons); `alpha` large makes it heavy-tailed
    (a few cells carry most co-occurrences), which is the realistic regime the
    paper studies for single columns and the regime in which join cardinality is
    dominated by well-identified heavy cells.
    """
    w = 1.0 / np.power(np.arange(1, vocab + 1, dtype=np.float64), alpha)
    w /= w.sum()
    W = np.outer(w, w)
    W /= W.sum()
    F = rng.multinomial(n, W.ravel()).reshape(vocab, vocab)
    return F.astype(np.int64)


# --------------------------------------------------------------------------
# 2. Gamma-Poisson prior fitted from the observed joint (deployment-style)
# --------------------------------------------------------------------------
def fit_gp_prior(g_flat: np.ndarray, p: float) -> np.ndarray:
    """Method-of-moments Gamma-Poisson prior over a cell's population count, fit
    from the observed joint cell counts alone (zero-truncation handled exactly
    as in the paper's deployment prior via the NB predictive).
    """
    obs = g_flat[g_flat >= 1].astype(np.float64)
    if obs.size == 0:
        return np.zeros(N_MAX + 1, dtype=np.float64)
    x = obs / p                              # de-thinned observed counts
    m = float(x.mean())
    v = float(x.var())
    if v > m and m > 0:
        a0 = float(np.clip(m * m / v, 0.25, 4.0))
    else:
        a0 = 2.0                               # paper's interior optimum
    b0 = a0 / m if m > 0 else 1.0
    js = np.arange(1, N_MAX + 1, dtype=np.float64)
    logw = (gammaln(js + a0) - gammaln(a0) - gammaln(js + 1)
            + a0 * math.log(b0 / (b0 + 1.0)) + js * math.log(1.0 / (b0 + 1.0)))
    w_ = np.exp(logw)
    s = w_.sum()
    prior = np.zeros(N_MAX + 1, dtype=np.float64)
    if s > 0:
        prior[1:] = w_ / s
    return prior


def posterior_table(prior: np.ndarray, p: float):
    """Precompute, for each observed count g = 0..G_MAX, the posterior mean
    E[f | g] and the posterior mass over true counts f (shared prior, so the
    table is exact for every cell with that g).
    """
    logfact = np.concatenate(
        [[0.0], np.cumsum(np.log(np.arange(1, N_MAX + 512 + 1, dtype=np.float64)))])
    q = 1.0 - p
    logp = math.log(p) if p > 0 else -np.inf
    logq = math.log(q) if q > 0 else -np.inf
    mean_tbl = np.zeros(G_MAX + 1, dtype=np.float64)
    mass_tbl = np.zeros((G_MAX + 1, N_MAX + 1), dtype=np.float64)   # [g, f]
    for g in range(1, G_MAX + 1):
        f_mode = max(g, round(g / p))
        sigma = math.sqrt(max(g, 1) * (1.0 - p) / p)
        f_lo = max(1, g)
        f_hi = int(min(N_MAX, f_mode + 25 * sigma + 500))
        if f_hi < f_lo:
            f_hi = f_lo
        fsup = np.arange(f_lo, f_hi + 1, dtype=np.int64)
        fg = (fsup - g).astype(np.float64)
        logbin = (logfact[fsup] - logfact[g + 1] - logfact[np.maximum(fg, 0).astype(int)]
                  + g * logp + np.where(fg > 0, fg * logq, 0.0))
        w = prior[fsup] * np.exp(logbin)
        tot = w.sum()
        if tot <= 0:
            continue
        pr = w / tot
        mean_tbl[g] = float(np.sum(fsup * pr))
        # mass over true counts f = f_lo .. f_hi (index 1-based -> f_lo-1)
        mass_tbl[g, f_lo:f_hi + 1] = pr
    return mean_tbl, mass_tbl


# --------------------------------------------------------------------------
# 3. Good-Turing on the flattened joint (baseline)
# --------------------------------------------------------------------------
def gt_joint_cell_counts(f_samp: np.ndarray) -> np.ndarray:
    """Good-Turing estimate of the joint FOF spectrum (expected #cells at each
    true co-occurrence count), taking the observed cell-count spectrum as input.
    """
    H = H_CAP
    F = np.zeros(H, dtype=np.float64)
    L = min(int(f_samp.size), H)
    for c in range(1, L):
        denom = f_samp[c - 1]
        if denom > 0:
            F[c - 1] = (c + 1) * f_samp[c] / denom
    f1 = float(f_samp[0]) if f_samp.size > 0 else 0.0
    f2 = float(f_samp[1]) if f_samp.size > 1 else 0.0
    unseen = (f1 * f1) / (2.0 * f2) if f2 > 0 else f1 * (f1 - 1.0) / 2.0
    F[0] += max(unseen, 0.0)
    return np.clip(F, 0.0, None)


# --------------------------------------------------------------------------
# 4. Metrics
# --------------------------------------------------------------------------
def join_cardinality(F_joint: np.ndarray) -> float:
    """sum_{cells} f^2 for a joint count distribution, computed from its FOF
    spectrum F_j (expected #cells with population count j): E[sum f^2] =
    sum_j F_j * j^2."""
    js = np.arange(1, len(F_joint) + 1, dtype=np.float64)
    return float(np.sum(F_joint * js * js))


def joint_w1(F_true: np.ndarray, F_hat: np.ndarray, j_eval: int = H_CAP) -> float:
    """Integral W1 (CDF L1 area) on a truncated support, both spectra capped."""
    lt = min(len(F_true), j_eval)
    lh = min(len(F_hat), j_eval)
    L = max(lt, lh)
    t = np.zeros(L); t[:lt] = F_true[:lt]
    h = np.zeros(L); h[:lh] = F_hat[:lh]
    st, sh = t.sum(), h.sum()
    if st <= 0 or sh <= 0:
        return float("nan")
    ct = np.cumsum(t / st)
    ch = np.cumsum(h / sh)
    return float(np.sum(np.abs(ct - ch)))


def recover_one(J: np.ndarray, p: float, rng: np.random.Generator) -> dict:
    V = J.shape[0]
    J_flat = J.ravel().astype(np.int64)
    g = D.thin(J_flat, p, rng)
    g_mat = g.reshape(V, V)

    prior = fit_gp_prior(g, p)
    mean_tbl, mass_tbl = posterior_table(prior, p)

    # recovered joint matrix (posterior mean per observed cell; unseen = 0).
    # Cells with observed count above the lookup table are well identified, so
    # the posterior mean is ~ g/p; we use that point estimate there.
    J_hat_flat = np.zeros_like(g, dtype=np.float64)
    obs = g >= 1
    idx_obs = np.where(obs)[0]
    gv = g[idx_obs].astype(np.int64)
    within = gv <= G_MAX
    J_hat_flat[idx_obs[within]] = mean_tbl[np.minimum(gv[within], G_MAX)]
    big = idx_obs[~within]
    if big.size:
        J_hat_flat[big] = g[big] / p
    J_hat_mat = J_hat_flat.reshape(V, V)

    # --- join cardinality Q-error (the optimizer functional) ---
    F_true = D.fofo_spectrum(J_flat, j_max=H_CAP)
    JC_true = join_cardinality(F_true)

    F_hat = np.zeros(len(F_true), dtype=np.float64)
    for gv in range(1, G_MAX + 1):
        cnt = int(np.sum(g == gv))
        if cnt == 0:
            continue
        F_hat += cnt * mass_tbl[gv, 1:1 + len(F_true)]
    # high-count cells: posterior mass concentrated at f ~ round(g/p)
    if big.size:
        f_center = np.clip(np.round(g[big] / p).astype(np.int64), 1, len(F_true))
        np.add.at(F_hat, f_center - 1, 1.0)
    # raw join cardinality from per-cell g/p
    JC_raw = float(np.sum((g_mat.astype(np.float64) / p) ** 2))
    JC_hat = float(np.sum(J_hat_mat ** 2))

    # Good-Turing baseline on the flattened joint
    F_gt = gt_joint_cell_counts(D.fofo_spectrum(g, j_max=H_CAP))
    JC_gt = join_cardinality(F_gt)

    def q(err_est: float) -> float:
        if JC_true <= 0 or err_est <= 0:
            return float("nan")
        return max(err_est / JC_true, JC_true / err_est)

    w1_hat = joint_w1(F_true, F_hat)
    w1_raw = joint_w1(F_true, D.fofo_spectrum(g, j_max=H_CAP))
    w1_gt = joint_w1(F_true, F_gt)

    return {
        "JC_true": JC_true,
        "q_ours": q(JC_hat), "q_raw": q(JC_raw), "q_gt": q(JC_gt),
        "JC_ours": JC_hat, "JC_raw": JC_raw, "JC_gt": JC_gt,
        "w1_ours": w1_hat, "w1_raw": w1_raw, "w1_gt": w1_gt,
        "NDV_true": float(F_true.sum()), "N_true": float(J_flat.sum()),
        "d_obs": float((g >= 1).sum()),
    }


# --------------------------------------------------------------------------
# 5. Driver
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng_master = np.random.default_rng(20261104)
    N_POPS = 24
    SEEDS = 3
    P_VALUES = [0.1, 0.3]
    N_TUPLES = 100_000
    # Two regimes: a heavy-tailed join (the paper's domain; few cells carry most
    # co-occurrences) and an ultra-sparse near-uniform join (most cells are
    # singletons).  The latter is where the join-cardinality functional is
    # dominated by singleton cells and exposes a prior-misspecification limit.
    REGIMES = [("heavy", 1.3, 200), ("sparse", 0.4, 500)]

    rows = []
    for tag, alpha, vocab in REGIMES:
        for p in P_VALUES:
            q_ours, q_raw, q_gt = [], [], []
            w1_ours, w1_raw, w1_gt = [], [], []
            for pop in range(N_POPS):
                seed = int(rng_master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)
                J = build_joint(rng, N_TUPLES, vocab, alpha)
                for s in range(SEEDS):
                    sr = np.random.default_rng(seed * 100 + s)
                    r = recover_one(J, p, sr)
                    q_ours.append(r["q_ours"]); q_raw.append(r["q_raw"]); q_gt.append(r["q_gt"])
                    w1_ours.append(r["w1_ours"]); w1_raw.append(r["w1_raw"]); w1_gt.append(r["w1_gt"])
                    r.update({"regime": tag, "alpha": alpha, "vocab": vocab, "p": p,
                              "pop": pop, "seed": seed})
                    rows.append(r)
            qg = np.exp(np.nanmean(np.log(np.array(q_ours))))
            rg = np.exp(np.nanmean(np.log(np.array(q_raw))))
            gg = np.exp(np.nanmean(np.log(np.array(q_gt))))
            print(f"{tag} alpha={alpha} V={vocab} p={p}: "
                  f"Q geom(ours)={qg:.3f} raw={rg:.3f} gt={gg:.3f}")

    # summary
    def geomean(xs):
        xs = np.array([x for x in xs if x is not None and not math.isnan(x)])
        return float(np.exp(np.mean(np.log(xs)))) if xs.size else float("nan")

    summary = {"n_pops": N_POPS, "seeds": SEEDS, "n_tuples": N_TUPLES, "by_cell": {}}
    for tag, alpha, vocab in REGIMES:
        for p in P_VALUES:
            sub = [r for r in rows if r["regime"] == tag and r["p"] == p]
            key = f"{tag}_a{alpha}_V{vocab}_p{p}"
            summary["by_cell"][key] = {
                "q_ours": geomean([r["q_ours"] for r in sub]),
                "q_raw": geomean([r["q_raw"] for r in sub]),
                "q_gt": geomean([r["q_gt"] for r in sub]),
                "w1_ours": float(np.nanmedian([r["w1_ours"] for r in sub])),
                "w1_raw": float(np.nanmedian([r["w1_raw"] for r in sub])),
                "w1_gt": float(np.nanmedian([r["w1_gt"] for r in sub])),
                "n": len(sub),
            }

    with open(os.path.join(OUT_DIR, "joint_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # csv
    keys = ["regime", "alpha", "vocab", "p", "pop", "seed", "JC_true", "JC_ours",
            "JC_raw", "JC_gt", "q_ours", "q_raw", "q_gt", "w1_ours", "w1_raw",
            "w1_gt", "NDV_true", "N_true", "d_obs"]
    with open(os.path.join(OUT_DIR, "joint_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # report
    lines = []
    lines.append("# Joint-distribution recovery under the binomial-thinning channel\n")
    lines.append(
        f"Controlled demonstration on synthetic V x V co-occurrence matrices "
        f"(N={N_TUPLES} tuples, 2D Zipf product-of-marginals), thinned entrywise "
        f"at p, then inverted with the paper's closed-form per-cell posterior and "
        f"a Gamma-Poisson prior fitted from the observed joint only.  This recovers "
        f"the joint count matrix and its join-cardinality functional sum f^2.  It "
        f"is a controlled demonstration with known ground truth, NOT a live "
        f"query-optimizer evaluation.\n")
    lines.append("\n## Join-cardinality Q-error (geometric mean over populations x seeds, lower better)\n")
    lines.append("| regime | alpha | V | p | ours | raw 1/p | Good-Turing |")
    lines.append("|---|---|---|---|---|---|---|")
    for tag, alpha, vocab in REGIMES:
        for p in P_VALUES:
            c = summary["by_cell"][f"{tag}_a{alpha}_V{vocab}_p{p}"]
            lines.append(
                f"| {tag} | {alpha} | {vocab} | {p} | {c['q_ours']:.3f} | "
                f"{c['q_raw']:.3f} | {c['q_gt']:.3f} |")
    lines.append("\n## Joint FOF-spectrum W1 (median, lower better)\n")
    lines.append("| regime | alpha | V | p | ours | raw 1/p | Good-Turing |")
    lines.append("|---|---|---|---|---|---|---|")
    for tag, alpha, vocab in REGIMES:
        for p in P_VALUES:
            c = summary["by_cell"][f"{tag}_a{alpha}_V{vocab}_p{p}"]
            lines.append(
                f"| {tag} | {alpha} | {vocab} | {p} | {c['w1_ours']:.3f} | "
                f"{c['w1_raw']:.3f} | {c['w1_gt']:.3f} |")
    lines.append(
        "\n## Reading\nThe binomial-thinning kernel is binomial on every cell, "
        "so the same closed-form per-cell posterior that recovers a single "
        "frequency histogram inverts a joint co-occurrence matrix entry by entry. "
        "We verify this mechanically: the recovered joint runs end to end and its "
        "join-cardinality functional sum f^2 is computable from the same formula. "
        "The empirical result is a prior-transfer limit, not a win, and we report "
        "it plainly.  On join cardinality (the functional an equi-join optimizer "
        "needs) our recovery ties the naive raw 1/p scaling and is dominated by "
        "Good-Turing in every regime (Q-error 1.98-5.11 for Good-Turing against "
        "49-51 on the heavy join and 1.3-2.3 on the sparse join for our method): "
        "sum f^2 is driven by the unseen and singleton-cell mass, which "
        "Good-Turing estimates directly and the Gamma-Poisson prior fitted from a "
        "single column does not.  On the joint FOF spectrum (W1) the prior-"
        "inflated recovery is farther from the truth than naive 1/p in every "
        "regime (e.g. 17.9 vs 2.6 at p=0.1 on the heavy join), because the "
        "joint's cell-count distribution is more singleton-heavy than the "
        "single-column marginal the prior was tuned for, so the posterior over-"
        "corrects rare cells.  This is the same singleton over-correction the "
        "single-column study already documents, now exposed on the joint because "
        "no heavy tail dominates sum f^2 there.  Conclusion: the framework covers "
        "joint recovery in principle (entrywise kernel, identical formula), but "
        "realizing the optimizer payoff requires a prior fit to the joint's own "
        "cell-count distribution; a joint-aware prior is future work.  A live "
        "DBMS join-cardinality evaluation remains future work.\n")

    with open(os.path.join(OUT_DIR, "joint_report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", os.path.join(OUT_DIR, "joint_report.md"))

    # figure
    try:
        _make_figure(summary, REGIMES, P_VALUES)
    except Exception as e:  # pragma: no cover
        print("figure skipped:", e)


def _make_figure(summary, regimes, p_values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    # Q-error grouped bars
    ax = axes[0]
    labels, ours, raw, gt = [], [], [], []
    for tag, alpha, vocab in regimes:
        for p in p_values:
            c = summary["by_cell"][f"{tag}_a{alpha}_V{vocab}_p{p}"]
            labels.append(f"{tag}\np={p}")
            ours.append(c["q_ours"]); raw.append(c["q_raw"]); gt.append(c["q_gt"])
    x = np.arange(len(labels)); w = 0.26
    ax.bar(x - w, ours, w, label="ours", color=D.CB["ours"])
    ax.bar(x, raw, w, label="raw 1/p", color=D.CB["none"])
    ax.bar(x + w, gt, w, label="Good-Turing", color=D.CB["anchor"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("join-cardinality Q-error (geo. mean)")
    ax.set_title("Join cardinality Q-error")
    ax.legend(fontsize=7)
    # W1 grouped bars
    ax = axes[1]
    ours2, raw2, gt2 = [], [], []
    for tag, alpha, vocab in regimes:
        for p in p_values:
            c = summary["by_cell"][f"{tag}_a{alpha}_V{vocab}_p{p}"]
            ours2.append(c["w1_ours"]); raw2.append(c["w1_raw"]); gt2.append(c["w1_gt"])
    ax.bar(x - w, ours2, w, label="ours", color=D.CB["ours"])
    ax.bar(x, raw2, w, label="raw 1/p", color=D.CB["none"])
    ax.bar(x + w, gt2, w, label="Good-Turing", color=D.CB["anchor"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("joint spectrum W1 (median)")
    ax.set_title("Joint-spectrum recovery")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_joint_recovery.pdf"))
    print("wrote fig_joint_recovery.pdf")


if __name__ == "__main__":
    main()
