"""Numerical capability metrics for the Diffusion4Distribution recovery method.

This module turns the paper's qualitative claims into numbers you can check on
real data.  The method's analytic half (closed-form posterior P(f | g) plus the
90 percent plausible set) is deterministic and needs no training, so it is the
right thing to validate on real frequency histograms.

What an "attribute" is
---------------------
Real NDV datasets ship one CSV per column.  Each row is (distinct_value, count):
column 2 is the TRUE frequency f_i of that distinct value.  That frequency
vector f is exactly the population the method is meant to recover from a thinned
sample g ~ Binomial(f, p).

Oracle prior
------------
The posterior is P(f | g) proportional to Prior(f) times Bin(g | f, p).  As in
the paper's demonstration, we build the prior from the ground-truth frequencies
(best case).  This isolates the test to the inversion math itself.

Metrics produced per (attribute, retention p)
--------------------------------------------
  W1 spectrum recovery      W1(truth_spec, recovered) vs W1(truth_spec, observed)
                            and vs a naive 1/p scaling baseline
  singleton recovery F1     relative error of the singleton count
  total count recovery      observed-type mass recovered toward the truth
  calibration               value-level 90/95 percent interval coverage and a
                            calibration curve at several nominal levels
  band coverage              fraction of j where the true spectrum lies inside
                            the 90 percent credible band

All functions are pure numpy and run without torch.
"""

from __future__ import annotations

import numpy as np

from d4d_logic import thin, fofo_spectrum, analytic_posterior


def _logfact(n: int) -> np.ndarray:
    n = int(n)
    if n < 0:
        n = 0
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1, dtype=np.float64)))])


def prior_from_truth(f_true: np.ndarray, N: int | None = None):
    """Build the oracle empirical prior P(f) from the true frequencies.

    Returns (prior_f, logfact).  prior_f[f] = fraction of present types whose
    true frequency equals f (f >= 1).  f = 0 carries no mass because a present
    type always has f >= 1.
    """
    f_true = np.asarray(f_true, dtype=np.int64)
    if N is None:
        N = int(f_true.sum())
    else:
        N = int(N)
    F = fofo_spectrum(f_true)
    prior_f = np.zeros(N + 1, dtype=np.float64)
    J = min(len(F), N)
    for j in range(1, J + 1):
        prior_f[j] = F[j - 1]
    s = prior_f.sum()
    if s > 0:
        prior_f = prior_f / s
    logfact = _logfact(N + 512)
    return prior_f, logfact


def spectrum_w1(spec_a: np.ndarray, spec_b: np.ndarray, j_eval: int = 512) -> float:
    """Wasserstein-1 distance between two integer-line frequency spectra.

    Both spectra are truncated to j = 1..j_eval, renormalized to a CDF, and the
    L1 area between the two CDFs is returned.  Lower is better.
    """
    a = np.zeros(j_eval, dtype=np.float64)
    b = np.zeros(j_eval, dtype=np.float64)
    sa = np.asarray(spec_a, dtype=np.float64)
    sb = np.asarray(spec_b, dtype=np.float64)
    a[: min(len(sa), j_eval)] = sa[:j_eval]
    b[: min(len(sb), j_eval)] = sb[:j_eval]
    ta, tb = a.sum(), b.sum()
    if ta <= 0 or tb <= 0:
        return float("nan")
    ca = np.cumsum(a / ta)
    cb = np.cumsum(b / tb)
    return float(np.abs(ca - cb).sum())


def posterior_mean_counts(g_obs: np.ndarray, p: float,
                          prior_f: np.ndarray, logfact: np.ndarray,
                          N: int) -> np.ndarray:
    """Posterior-mean frequency for every observed type (collapsed types = 0)."""
    g = np.asarray(g_obs, dtype=np.int64)
    out = np.zeros(len(g), dtype=np.float64)
    for gi in np.where(g >= 1)[0]:
        fsup, prob = analytic_posterior(int(g[gi]), p, prior_f, logfact, N)
        out[gi] = float((fsup * prob).sum())
    return out


# nominal level -> (lower percentile, upper percentile) for a central interval
_LEVELS = {
    0.50: (25.0, 75.0),
    0.80: (10.0, 90.0),
    0.90: (5.0, 95.0),
    0.95: (2.5, 97.5),
    0.99: (0.5, 99.5),
}


def validate_attribute(f_true: np.ndarray, p: float, K: int,
                       rng: np.random.Generator, N: int | None = None) -> dict:
    """Run the full analytic recovery on one real attribute at retention p.

    Draws K posterior population vectors once, then derives every metric from
    that single set of draws.  Returns a flat dict of capability numbers.
    """
    f_true = np.asarray(f_true, dtype=np.int64)
    if N is None:
        N = int(f_true.sum())
    N = int(N)
    prior_f, logfact = prior_from_truth(f_true, N)

    g = thin(f_true, p, rng)
    g_obs = g.astype(np.int64)
    truth_mask = g_obs >= 1
    n_types = len(f_true)

    # group observed types by their observed count g
    groups: dict[int, list[int]] = {}
    for gi in np.where(truth_mask)[0]:
        groups.setdefault(int(g_obs[gi]), []).append(int(gi))

    c_k = np.zeros((K, n_types), dtype=np.int64)
    mean_f = np.zeros(n_types, dtype=np.float64)
    lo = {lv: np.zeros(n_types, dtype=np.float64) for lv in _LEVELS}
    hi = {lv: np.zeros(n_types, dtype=np.float64) for lv in _LEVELS}

    for gg, idxs in groups.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        fsup, prob = analytic_posterior(gg, p, prior_f, logfact, N)
        samp = rng.choice(fsup, size=(K, len(idxs)), p=prob)
        c_k[:, idxs] = samp
        mean_f[idxs] = (fsup * prob).sum()
        for lv, (qlo, qhi) in _LEVELS.items():
            lo[lv][idxs] = np.percentile(samp, qlo, axis=0)
            hi[lv][idxs] = np.percentile(samp, qhi, axis=0)

    true_f = f_true[truth_mask].astype(np.int64)

    # value-level coverage at each nominal level (observed types only)
    cov = {}
    for lv in _LEVELS:
        inside = (true_f >= lo[lv][truth_mask]) & (true_f <= hi[lv][truth_mask])
        cov[lv] = float(np.mean(inside)) if true_f.size else float("nan")

    # spectra
    truth_spec = fofo_spectrum(f_true[truth_mask])
    obs_spec = fofo_spectrum(g)
    specs = [fofo_spectrum(c_k[k]) for k in range(K)]
    Jmax = max((len(s) for s in specs), default=0)
    if Jmax == 0:
        # every type collapsed under thinning: no spectrum to recover
        mean_spec = np.zeros(0, dtype=np.float64)
        band_lo = np.zeros(0, dtype=np.float64)
        band_hi = np.zeros(0, dtype=np.float64)
    else:
        S = np.zeros((K, Jmax))
        for kk, s in enumerate(specs):
            S[kk, : len(s)] = s
        mean_spec = S.mean(0)
        band_lo = np.percentile(S, 5, axis=0)
        band_hi = np.percentile(S, 95, axis=0)

    # naive 1/p scaling baseline (observed types only)
    g_scaled = np.zeros(n_types, dtype=np.int64)
    g_scaled[truth_mask] = np.round(g[truth_mask] / p).astype(np.int64)
    naive_spec = fofo_spectrum(g_scaled)

    # spectrum-level 90 percent band coverage
    Jb = min(len(truth_spec), len(band_lo))
    if Jb > 0:
        in_band = (truth_spec[:Jb] >= band_lo[:Jb]) & (truth_spec[:Jb] <= band_hi[:Jb])
        band_cov = float(np.mean(in_band))
    else:
        band_cov = float("nan")

    # headline numbers
    w1_obs = spectrum_w1(truth_spec, obs_spec)
    w1_rec = spectrum_w1(truth_spec, mean_spec)
    w1_naive = spectrum_w1(truth_spec, naive_spec)

    N_true_obs = float(f_true[truth_mask].sum())
    N_obs = float(g.sum())
    N_rec = float(mean_f[truth_mask].sum())

    f1_true = float(truth_spec[0]) if truth_spec.size else 0.0
    f1_obs = float(obs_spec[0]) if obs_spec.size else 0.0
    f1_rec = float(mean_spec[0]) if mean_spec.size else 0.0

    def _relerr(est, truth):
        return float(abs(est - truth) / truth) if truth > 0 else float("nan")

    return {
        "p": float(p),
        "K": int(K),
        "NDV_true": int(n_types),
        "NDV_obs": int(truth_mask.sum()),
        "NDV_lost_frac": float(1.0 - truth_mask.sum() / max(n_types, 1)),
        "N_true_total": int(f_true.sum()),
        "N_true_obs": N_true_obs,
        "N_obs": N_obs,
        "N_rec": N_rec,
        "N_relerr_obs": _relerr(N_obs, N_true_obs),
        "N_relerr_rec": _relerr(N_rec, N_true_obs),
        "W1_obs": w1_obs,
        "W1_rec": w1_rec,
        "W1_naive": w1_naive,
        "W1_improve_rec": float((w1_obs - w1_rec) / w1_obs) if w1_obs > 0 else float("nan"),
        "W1_improve_naive": float((w1_obs - w1_naive) / w1_obs) if w1_obs > 0 else float("nan"),
        "F1_true": f1_true,
        "F1_obs": f1_obs,
        "F1_rec": f1_rec,
        "F1_relerr_obs": _relerr(f1_obs, f1_true),
        "F1_relerr_rec": _relerr(f1_rec, f1_true),
        "cov90": cov[0.90],
        "cov95": cov[0.95],
        "cal_dev": float(max(abs(cov[0.90] - 0.90), abs(cov[0.95] - 0.95))),
        "band_cov90": band_cov,
        "calib_curve": {str(lv): cov[lv] for lv in _LEVELS},
    }
