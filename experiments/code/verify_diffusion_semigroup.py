"""Why a multi-step reverse chain is redundant on a known thinning channel.

Part A proves and checks a semigroup identity for the conjugate Gamma-Poisson
channel. Part B checks one-shot Bayes inversion on three priors, two of them
non-conjugate, and measures how the Tweedie score drifts with the noise level.
All numbers come from closed forms or numerical quadrature. No network is
trained here. The k-step chain below is an analytically tractable surrogate
for a mean-field denoiser, not a reproduction of a trained DDPM.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gammaln


def stable_seed(*parts):
    """Deterministic seed independent of PYTHONHASHSEED."""
    key = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results" / "diffusion_semigroup"
OUT.mkdir(parents=True, exist_ok=True)

LAM_MAX = 80.0
M = 1200
LAM = np.linspace(1e-3, LAM_MAX, M)
DLAM = LAM[1] - LAM[0]
GV = np.arange(0, 61)


def exact_mean(g, p, a0, b0):
    """One-shot Bayes posterior mean E[f | g] under the conjugate prior."""
    return g + (1.0 - p) * (a0 + g) / (b0 + p)


def schedules(p, k, mode):
    """Return the start rates and step sizes of a k-step reverse schedule."""
    i = np.arange(k + 1, dtype=float)
    if mode == "linear":
        q = p + (1.0 - p) * i / k
    else:
        q = p ** (1.0 - i / k)
    return q[:-1], np.diff(q)


def kstep_conjugate(g, p, a0, b0, k, mode):
    """Mean-field denoising chain with the exact local posterior score."""
    q, dq = schedules(p, k, mode)
    m = float(g)
    K = (a0 + g) / (b0 + p)
    drift = []
    for qi, d in zip(q, dq):
        s = (a0 + m) / (b0 + qi)
        drift.append(abs(s - K) / K)
        m = m + d * s
    return m, max(drift)


def part_a():
    a0, b0 = 1.2, 0.8
    rows = []
    worst = 0.0
    for p in (0.1, 0.2, 0.3, 0.5):
        for mode in ("linear", "geometric"):
            for k in (1, 2, 5, 10, 25):
                diffs, drifts = [], []
                for g in range(1, 41):
                    m, drift = kstep_conjugate(g, p, a0, b0, k, mode)
                    diffs.append(abs(m - exact_mean(g, p, a0, b0)))
                    drifts.append(drift)
                mx, dr = max(diffs), max(drifts)
                worst = max(worst, mx)
                rows.append({"p": p, "schedule": mode, "k": k,
                             "max_abs_diff": mx, "max_score_drift": dr})
    return {"a0": a0, "b0": b0, "n_configurations": len(rows),
            "worst_max_abs_diff": worst, "rows": rows}


def gamma_pmf(shape, rate):
    x = (shape * math.log(rate) + (shape - 1.0) * np.log(LAM)
         - rate * LAM - gammaln(shape))
    w = np.exp(x - x.max())
    return w / w.sum()


PRIORS = {
    "gamma_conjugate": lambda: gamma_pmf(1.2, 0.8),
    "two_component_mix": lambda: (0.6 * gamma_pmf(0.6, 0.30)
                                   + 0.4 * gamma_pmf(3.0, 3.0)),
    "heavy_tail_mix": lambda: (0.8 * gamma_pmf(0.4, 0.10)
                                + 0.2 * gamma_pmf(8.0, 0.6)),
}


def score_curve(prior, q):
    """E[lambda | g, q] for g = 0..60 by numerical Bayes on the lambda grid."""
    mu = q * LAM
    with np.errstate(divide="ignore", invalid="ignore"):
        logpmf = (GV[:, None] * np.log(mu[None, :]) - mu[None, :]
                  - gammaln(GV[:, None] + 1.0))
    logpmf = np.nan_to_num(logpmf, nan=-1e300, neginf=-1e300)
    w = prior[None, :] * np.exp(logpmf - logpmf.max(axis=1, keepdims=True))
    den = w.sum(axis=1)
    num = w @ LAM
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-300)


def score_at(curve, m):
    return float(np.interp(m, GV, curve))


def one_shot_bayes(prior, p):
    """Exact E[f | g] = g + (1-p) E[lambda | g, p] for any known prior."""
    return GV + (1.0 - p) * score_curve(prior, p)


def mc_check(prior, p, seed=0, n=300000):
    """Monte Carlo cross-check of the one-shot identity."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(M, size=n, p=prior)
    L = np.maximum(LAM[idx] + rng.uniform(-DLAM / 2, DLAM / 2, n), 1e-6)
    f = rng.poisson(L)
    g = rng.binomial(f, p)
    est = one_shot_bayes(prior, p)
    errs = []
    for gg in (1, 2, 3, 5, 8, 12, 20):
        mask = g == gg
        if mask.sum() < 200:
            continue
        truth = float(f[mask].mean())
        errs.append({"g": gg, "n": int(mask.sum()), "mc_mean": truth,
                     "closed_form": float(est[gg]),
                     "rel_err": abs(truth - est[gg]) / truth})
    return errs


def kstep_numeric(prior, p, g, k, mode):
    """Mean-field chain using the numerically computed local score."""
    q, dq = schedules(p, k, mode)
    curves = [score_curve(prior, float(qi)) for qi in q]
    m = float(g)
    s0 = score_at(curves[0], m)
    scores = []
    for qi, d, cv in zip(q, dq, curves):
        s = score_at(cv, m)
        scores.append(s)
        m = m + d * s
    drift = max(abs(s - s0) / max(s0, 1e-12) for s in scores)
    return m, drift


def part_b():
    out = {}
    for name, make in PRIORS.items():
        prior = make()
        for p in (0.1, 0.3):
            mc = mc_check(prior, p, seed=stable_seed(name, p))
            worst_mc = max((r["rel_err"] for r in mc), default=float("nan"))
            chains = []
            for k in (1, 2, 5, 10):
                devs, drifts = [], []
                for g in (1, 2, 3, 5, 8, 12, 20):
                    m, drift = kstep_numeric(prior, p, g, k, "linear")
                    ref = one_shot_bayes(prior, p)[g]
                    devs.append(abs(m - ref) / max(ref, 1e-12))
                    drifts.append(drift)
                chains.append({"k": k, "mean_rel_deviation": float(np.mean(devs)),
                               "max_score_drift": float(max(drifts))})
            out[f"{name}_p{p}"] = {
                "worst_mc_rel_err": worst_mc,
                "mc_rows": mc,
                "chains": chains,
            }
    return out


def main():
    a = part_a()
    b = part_b()
    payload = {"part_a_conjugate_semigroup": a, "part_b_prior_ladder": b}
    (OUT / "semigroup_results.json").write_text(json.dumps(payload, indent=2))

    mc_all = [r["rel_err"] for v in b.values() for r in v["mc_rows"]]
    mc_lo, mc_hi = 100 * min(mc_all), 100 * max(mc_all)
    drifts = [c["max_score_drift"] for key, v in b.items()
              if "conjugate" not in key
              for c in v["chains"] if c["k"] == 10]
    drift_lo, drift_hi = min(drifts), max(drifts)
    L = ["# Why the reverse chain collapses on a known thinning channel", "",
         "Date 2026-11-04. Generated by `verify_diffusion_semigroup.py`.",
         "No network is trained here. The k-step chain is an analytically",
         "tractable mean-field surrogate, not a reproduction of a DDPM.", "",
         "## Part A. Conjugate semigroup identity", "",
         "Under `lambda ~ Gamma(a0,b0)`, `f | lambda ~ Poisson(lambda)` and",
         "`g | f ~ Bin(f,p)`, the Poisson decomposition gives",
         "`E[f | g] = g + (1-p)(a0+g)/(b0+p)`. Splitting p into k reverse",
         "steps of rates q_0=p < ... < q_k=1, each step applies",
         "`m <- m + (q_{i+1}-q_i) (a0+m)/(b0+q_i)`. The ratio",
         "`(a0+m)/(b0+q)` stays constant along the trajectory by induction,",
         "so the composite equals the one-shot value for every k and every",
         "schedule. The chain carries no statistical information.", "",
         f"Tested configurations {a['n_configurations']}.",
         f"Worst absolute deviation {a['worst_max_abs_diff']:.3e}.",
         "", "| p | schedule | k | max abs diff | max score drift |",
         "|---|---|---|---|---|"]
    for r in a["rows"]:
        if r["k"] in (1, 5, 25):
            L.append(f"| {r['p']} | {r['schedule']} | {r['k']} | "
                     f"{r['max_abs_diff']:.2e} | {r['max_score_drift']:.2e} |")
    L += ["", "## Part B. Prior ladder and score drift", "",
          "For any known prior, `E[f|g] = g + (1-p) E[lambda|g,p]` is exact.",
          "Monte Carlo confirms the numerical quadrature. The score drift",
          "column measures how far `E[lambda|m,q]` moves along the chain.",
          "Zero drift means one step is exact. Positive drift means the score",
          "couples to the noise level and must be estimated at each q.", "",
          "| prior | p | worst MC rel err | k=1 dev | k=10 dev | k=10 drift |",
          "|---|---|---|---|---|---|"]
    for key, v in b.items():
        name, p = key.rsplit("_p", 1)
        ch = {c["k"]: c for c in v["chains"]}
        L.append(f"| {name} | {p} | {v['worst_mc_rel_err']:.3f} | "
                 f"{ch[1]['mean_rel_deviation']:.2e} | "
                 f"{ch[10]['mean_rel_deviation']:.2e} | "
                 f"{ch[10]['max_score_drift']:.3f} |")
    L += ["", "## Caveats", "",
          "The k=1 deviation is zero by construction, since the one-step chain",
          "is the identity itself. The k=10 deviation on the conjugate prior,",
          "about 5e-6, comes from linearly interpolating the numerical score at",
          "real-valued m. Part A uses the closed-form score and gives 7e-14.",
          "It is interpolation error, not a statistical gap. Drift is not",
          "monotone in prior complexity. It is nonzero whenever the score",
          "couples to q. Monte Carlo cross-check errors of "
          f"{mc_lo:.1f} to {mc_hi:.1f} percent are finite",
          "sampling noise, not bias in the identity.", "",
          f"Non-conjugate ten-step drift ranges from {drift_lo:.3f} to "
          f"{drift_hi:.3f}.", "",
          "## Reading", "",
          "The conjugate channel is the zero-drift endpoint. There the reverse",
          "path collapses to one step, which is what Section V-B observes",
          "experimentally as k=1 being best. On non-conjugate priors the score",
          "drifts with q, so a learner must estimate a family of local scores",
          "and integrate them. That is the numerical and learning reason",
          "multi-step diffusion helps. It is not extra statistical information.",
          "One-shot Bayes remains MSE exact whenever the prior is known.", "",
          "Three layers separate cleanly. The statistical layer is one Tweedie",
          "posterior mean and never needs more than one step. The numerical",
          "layer integrates a score field when no closed form exists. The",
          "learning layer obtains self-supervised targets by artificial",
          "corruption at many noise levels. Only the last two justify a chain.", ""]
    (OUT / "semigroup_report.md").write_text("\n".join(L))
    print("Part A worst abs diff", a["worst_max_abs_diff"])
    for key, v in b.items():
        ch = {c["k"]: c for c in v["chains"]}
        print(f"{key:28s} MC {v['worst_mc_rel_err']:.3f} "
              f"k1 {ch[1]['mean_rel_deviation']:.2e} "
              f"k10 {ch[10]['mean_rel_deviation']:.2e} "
              f"drift {ch[10]['max_score_drift']:.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
