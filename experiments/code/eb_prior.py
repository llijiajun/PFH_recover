"""Empirical-Bayes fitted prior matched to the observed spectrum's low-count cells.

Family: Gamma(a0,b0)-Poisson population counts thinned by Bernoulli(p), i.e. the
observed count of an existing type is NegativeBinomial.  The observable, C-free
identifying equations (C = total type mass cancels):

    r_j  =  f_j / d_obs  =  sum_f pi(f) Bin(j | f, p) / sum_f pi(f) [1-(1-p)^f]

for j = 1, 2 (fall back to j = 1, 3 when f2 = 0).  Two equations, two unknowns
(log a0, log b0), solved by least squares.  pi is the NB mass over f >= 1.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln


def _nb_mass(a0: float, b0: float, F: int) -> np.ndarray:
    """NB(a0, b0) pmf over f = 1..F (index 0 unused), normalized over f >= 1."""
    q = b0 / (b0 + 1.0)
    f = np.arange(0, F + 1, dtype=np.float64)
    logpmf = (gammaln(f + a0) - gammaln(a0) - gammaln(f + 1.0)
              + a0 * math.log(q) + f * math.log(1.0 - q))
    pmf = np.exp(logpmf)
    s = pmf[1:].sum()
    pmf[1:] = pmf[1:] / s if s > 0 else 0.0
    pmf[0] = 0.0
    return pmf


def fitted_prior_eb(g: np.ndarray, p: float, N_max: int):
    """Fit (a0, b0) to the singleton/doubleton (or 1/3) mass fractions.
    Returns (prior_f over 0..N_max, ve_chao1, a0, b0)."""
    g = np.asarray(g, dtype=np.int64)
    obs = g[g >= 1].astype(np.float64)
    d_obs = int(obs.size)
    n = int(g.sum())
    f_samp = D_fofo(g)
    if obs.size == 0 or d_obs == 0:
        # nothing survived thinning; fall back to a neutral prior
        f = np.arange(0, N_max + 1, dtype=np.float64)
        from scipy.special import gammaln as _g0
        logpmf = _g0(f + 2.0) - _g0(2.0) - _g0(f + 1.0) + f * math.log(0.5)
        prior_f = np.zeros(N_max + 1)
        s = logpmf[1:].clip(-745).sum()
        pmf = np.exp(logpmf)
        s = pmf[1:].sum()
        if s > 0:
            prior_f[1:] = pmf[1:] / s
        return prior_f, 0.0, 2.0, 1.0
    f1 = float(f_samp[0]) if f_samp.size > 0 else 0.0
    f2 = float(f_samp[1]) if f_samp.size > 1 else 0.0
    f3 = float(f_samp[2]) if f_samp.size > 2 else 0.0
    ve = d_obs + (f1 * f1) / (2.0 * f2) if f2 > 0 else d_obs + f1 * (f1 - 1.0) / 2.0

    js = (1, 2) if f2 > 0 else (1, 3)
    targets = tuple((f1, f2) if f2 > 0 else (f1, f3))

    # truncation: mass beyond f_hi contributes ~0 to observed counts <= 3
    f_hi = int(min(N_max, max(64.0, 12.0 * max(obs.max(), 3.0) / p)))
    from scipy.special import gammaln as _gl
    logfact = _gl(np.arange(f_hi + 2) + 1.0)

    def residuals(params):
        la, lb = params
        a0, b0 = math.exp(la), math.exp(lb)
        pi = _nb_mass(a0, b0, f_hi)
        denom = float(np.sum(pi[1:] * np.power(1.0 - p, np.arange(1, f_hi + 1))))
        denom = 1.0 - denom                      # P(type observed | exists)
        res = []
        if denom <= 1e-12:
            return [1e3, 1e3]
        for j, tgt in zip(js, targets):
            jj = int(j)
            fs = np.arange(jj, f_hi + 1)
            logbin = (logfact[fs] - logfact[jj] - logfact[fs - jj]
                      + jj * math.log(p) + (fs - jj) * math.log(1.0 - p))
            num = float(np.sum(pi[fs] * np.exp(logbin)))
            res.append(num / denom - tgt / d_obs)
        # keep the prior from piling everything on f = 1 (degenerate corner)
        res.append(0.05 * (math.exp(la) - 0.25))
        return res

    sol = least_squares(residuals, x0=[math.log(0.5), math.log(0.5)],
                        bounds=([math.log(1e-3), math.log(1e-4)],
                                [math.log(50.0), math.log(1e4)]), xtol=1e-10)
    a0, b0 = math.exp(sol.x[0]), math.exp(sol.x[1])

    q = b0 / (b0 + 1.0)
    f = np.arange(0, N_max + 1, dtype=np.float64)
    logpmf = (gammaln(f + a0) - gammaln(a0) - gammaln(f + 1.0)
              + a0 * math.log(q) + f * math.log(1.0 - q))
    prior_f = np.zeros(N_max + 1)
    pmf = np.exp(logpmf)
    s = pmf[1:].sum()
    if s > 0:
        prior_f[1:] = pmf[1:] / s
    return prior_f, ve, a0, b0


def D_fofo(g: np.ndarray):
    """FOF spectrum of a count vector, j >= 1."""
    g = np.asarray(g, dtype=np.int64)
    obs = g[g >= 1]
    if obs.size == 0:
        return np.zeros(0)
    jmax = int(obs.max())
    return np.bincount(obs, minlength=jmax + 1)[1:]
