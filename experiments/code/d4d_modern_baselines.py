"""Modern (2016--2025) estimators for NDV and frequency-distribution recovery.

This module exists to answer a deliberately adversarial question about the
paper: is the "diffusion" inverse actually new, or is it classical empirical
Bayes deconvolution written in a different vocabulary?  To decide that with
evidence rather than rhetoric we put the closed-form posterior next to the
strongest modern statistical estimators of the SAME channel.

Methods
-------
NDV scalar (all use only the observed sample):
  sgt          Smoothed Good--Toulmin, binomial smoothing (Orlitsky, Suresh,
               Wu; PNAS 2016).  Near-minimax for extrapolation up to t ~ log n.
  jack1/jack2  First/second-order Burnham--Overton jackknife richness.
  coverage     Good (1953) sample-coverage estimator  d_obs / (1 - f1/n).
  poly         Polynomial (moment-matching) linear estimator in the spirit of
               Wu & Yang (Annals of Statistics 2019) and the SIGMOD 2024
               learning-based polynomial framework.  Coefficients solve
               min_c || A^T c - 1 ||^2 with A[j,f] = Bin(j; f, p).
  npmle        Nonparametric maximum likelihood / profile maximum likelihood
               (Acharya et al. ICML 2017; Wang & Lindsay 2005/2007).  EM fits
               the abundance prior pi_f from the zero-truncated observed FoF.
  smooth_bayes Smoothed-Bayesian linear estimator (Beraha, Favaro, Sesia;
               JASA 2025).  Under a Gamma--Poisson abundance prior the reverse
               posterior mean is LINEAR in the observed count, namely
                   E[f | g] = g + (1-p) (a0+g)/(b0+p).
               This is the exact class of linear estimators that work proves
               optimal within.  It is also our own EB-fitted prior, so it is
               the sharpest possible test of "is this just ordinary statistics".

Full spectrum:
  npmle        The SAME closed-form posterior P(f|g) used by the paper, but
               with the NPMLE prior instead of the oracle / EB prior.
  smooth_bayes The same posterior with the Gamma--Poisson EB prior.
  eb_fit       Alias of smooth_bayes for the deployment row already in the paper.

All methods see the identical binomial-thinned sample, the same support
j = 1..H_CAP, and the same metrics as d4d_baselines.evaluate_all.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.special import gammaln

H_CAP = 512  # maximum true frequency bin used in the spectrum comparison (paper J)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _binom_kernel(J: int, F: int, p: float) -> np.ndarray:
    """K[j-1, f-1] = P(g = j | true count f), j = 1..J, f = 1..F."""
    j = np.arange(1, J + 1, dtype=np.float64)[:, None]
    f = np.arange(1, F + 1, dtype=np.float64)[None, :]
    ok = f >= j
    fj = np.maximum(f - j, 0.0)
    logK = (gammaln(f + 1.0) - gammaln(j + 1.0) - gammaln(fj + 1.0)
            + j * math.log(p) + fj * math.log(1.0 - p))
    K = np.exp(np.clip(np.where(ok, logK, -np.inf), -745.0, 0.0))
    K[~ok] = 0.0
    return K


def _support_size(p: float, max_obs: int) -> int:
    """Largest true frequency we model.  Beyond H_CAP every bin is folded into
    the last reported bin, so a truncated support is sufficient and keeps the
    EM kernel small."""
    need = int(math.ceil(2.0 * H_CAP / max(p, 1e-3)))
    return int(min(2048, max(512, need, 2 * min(max_obs, 256))))


def _fold(F_hat: np.ndarray, extra: float = 0.0) -> np.ndarray:
    """Fold all frequency bins > H_CAP into the last reported bin, mirroring
    spectrum_ours' clipping convention."""
    F_hat = np.asarray(F_hat, dtype=np.float64)
    out = np.zeros(H_CAP, dtype=np.float64)
    out[:H_CAP - 1] = F_hat[:H_CAP - 1]
    out[H_CAP - 1] = float(F_hat[H_CAP - 1:].sum()) + extra
    return np.clip(out, 0.0, None)


def _relerr(est: float, truth: float) -> float:
    return float(abs(est - truth) / truth) if truth > 0 else float("nan")


# ---------------------------------------------------------------------------
# NDV scalar: modern statistical estimators
# ---------------------------------------------------------------------------
def ndv_sgt(f_samp: np.ndarray, p: float, n: int) -> float:
    """Smoothed Good--Toulmin with binomial (Efron--Thisted) smoothing.

    Orlitsky, Suresh & Wu (PNAS 2016), eq. for U_L:
        U = sum_{i>=1} (-1)^{i+1} t^i P(L >= i) f_i,
        t = (1-p)/p,  P(L>=i) = P{Bin(k, q) >= i},  q = 1/(1+t),
        k = floor( 0.5 log2( n t^2 / (t-1) ) ).
    For t <= 1 the unsmoothed Good--Toulmin series converges, so P(L>=i)=1.
    """
    f_samp = np.asarray(f_samp, dtype=np.float64)
    d_obs = float(f_samp.sum())
    if d_obs <= 0 or n <= 0:
        return float("nan")
    t = (1.0 - p) / p
    if t <= 1.0:
        # plain Good--Toulmin, truncated at the largest observed frequency
        i = np.arange(1, len(f_samp) + 1, dtype=np.float64)
        U = float(np.sum((-1.0) ** (i + 1.0) * np.power(t, i) * f_samp))
        return max(d_obs + U, d_obs)

    k = int(math.floor(0.5 * math.log2(max(n * t * t / (t - 1.0), 2.0))))
    k = max(1, min(k, 40))
    q = 1.0 / (1.0 + t)
    # binomial survival P(Bin(k,q) >= i) for i = 1..k
    jj = np.arange(0, k + 1, dtype=np.float64)
    logpmf = (gammaln(k + 1.0) - gammaln(jj + 1.0) - gammaln(k - jj + 1.0)
              + jj * math.log(q) + (k - jj) * math.log(1.0 - q))
    pmf = np.exp(np.clip(logpmf, -745.0, 0.0))
    surv = np.array([pmf[i:].sum() for i in range(0, k + 1)])

    L = min(int(f_samp.size), k)
    i = np.arange(1, L + 1, dtype=np.float64)
    U = float(np.sum((-1.0) ** (i + 1.0) * np.power(t, i) * surv[1:L + 1] * f_samp[:L]))
    # clamp to a defensible range, the extrapolation is unreliable far out
    U = float(np.clip(U, 0.0, d_obs * max(t, 1.0)))
    return d_obs + U


def ndv_jack1(f_samp: np.ndarray, n: int) -> float:
    """First-order Burnham--Overton jackknife richness estimator."""
    f1 = float(f_samp[0]) if f_samp.size > 0 else 0.0
    return float(f_samp.sum()) + f1 * (n - 1.0) / max(n, 1.0)


def ndv_jack2(f_samp: np.ndarray, n: int) -> float:
    """Second-order Burnham--Overton jackknife richness estimator."""
    f1 = float(f_samp[0]) if f_samp.size > 0 else 0.0
    f2 = float(f_samp[1]) if f_samp.size > 1 else 0.0
    if n < 2:
        return ndv_jack1(f_samp, n)
    return (float(f_samp.sum())
            + (2.0 * n - 3.0) / n * f1
            - (n - 2.0) ** 2 / (n * (n - 1.0)) * f2)


def ndv_good_coverage(f_samp: np.ndarray, n: int) -> float:
    """Good (1953) coverage-based estimator: d_obs / C, C = 1 - f1/n."""
    f1 = float(f_samp[0]) if f_samp.size > 0 else 0.0
    C = 1.0 - f1 / max(n, 1.0)
    if C <= 1e-6:
        return float("nan")
    return float(f_samp.sum()) / C


def ndv_poly(f_samp: np.ndarray, p: float, max_obs: int) -> float:
    """Polynomial / moment-matching linear estimator (Wu & Yang, Ann. Statist.
    2019; Li et al., SIGMOD 2024 polynomial framework).

    We seek coefficients c_j such that sum_j c_j Bin(j; f, p) ~ 1 for every
    true frequency f.  Then sum_j c_j f_j is an (approximately) unbiased
    estimator of the number of distinct values, because
        E[f_j] = sum_f F_f Bin(j; f, p).
    Coefficients are fitted by ridge-regularized least squares on a log-spaced
    frequency grid, which is the Chebyshev / best-polynomial-approximation idea
    in a numerically stable form.
    """
    f_samp = np.asarray(f_samp, dtype=np.float64)
    J = min(int(f_samp.size), 64)
    if J < 2:
        return float("nan")
    F = _support_size(p, max_obs)
    grid = np.unique(np.round(np.geomspace(1.0, float(F), 120)).astype(int))
    grid = grid[grid >= 1]
    A = _binom_kernel(J, F, p)[:, grid - 1]        # J x |grid|
    # solve min ||A^T c - 1||^2 + lam ||c||^2
    G = A @ A.T
    lam = 1e-6 * max(float(np.trace(G)) / J, 1e-12)
    rhs = A.sum(axis=1)
    try:
        c = np.linalg.solve(G + lam * np.eye(J), rhs)
    except np.linalg.LinAlgError:
        return float("nan")
    est = float(np.dot(c, f_samp[:J]))
    return max(est, float(f_samp.sum()))


# ---------------------------------------------------------------------------
# NPMLE / profile maximum likelihood abundance prior
# ---------------------------------------------------------------------------
def npmle_prior(f_samp: np.ndarray, p: float, max_obs: int,
                iters: int = 100):
    """Fit pi_f (f = 1..F) by EM on the zero-truncated observed FoF profile.

    Returns (pi, p0, F, K) where p0 = P(g=0) = sum_f pi_f (1-p)^f and K is the
    binomial kernel used for the posterior deconvolution.  This is the
    nonparametric maximum likelihood / PML backbone of Acharya et al. (ICML
    2017) and of the SIGMOD 2024 polynomial property-estimation framework,
    adapted to the known-rate binomial thinning channel.
    """
    F = _support_size(p, max_obs)
    J = min(int(len(f_samp)), F, 384)
    fs = np.asarray(f_samp[:J], dtype=np.float64)
    K = _binom_kernel(J, F, p)
    z = np.power(1.0 - p, np.arange(1, F + 1, dtype=np.float64))
    pi = np.ones(F, dtype=np.float64) / F
    d_obs = float(fs.sum())
    p0 = float(z @ pi)
    if d_obs <= 0:
        return pi, p0, F, K

    for _ in range(iters):
        obs_mass = K @ pi                       # P(g=j), j = 1..J
        p0 = float(z @ pi)
        p_obs = max(1.0 - p0, 1e-12)           # exact under the truncated model
        R = K * pi[None, :] / np.maximum(obs_mass[:, None], 1e-300)
        n0 = d_obs * p0 / p_obs                # expected unseen species
        rz = z * pi / max(p0, 1e-300)
        new = fs @ R + n0 * rz
        s = float(new.sum())
        if s <= 0 or not np.isfinite(s):
            break
        new /= s
        delta = float(np.max(np.abs(new - pi)))
        pi = 0.5 * pi + 0.5 * new              # damping for stability
        pi /= pi.sum()
        if delta < 1e-8:
            break
    return pi, float(z @ pi), F, K


def ndv_npmle(pi: np.ndarray, p: float, d_obs: int) -> float:
    p0 = float(np.dot(pi, np.power(1.0 - p, np.arange(1, len(pi) + 1))))
    return float(d_obs) / max(1.0 - p0, 1e-9)


def spectrum_npmle(f_samp: np.ndarray, pi: np.ndarray, p: float,
                   F: int, K: np.ndarray, d_obs: int) -> np.ndarray:
    """The paper's own closed-form posterior deconvolution, with the NPMLE
    prior in place of the oracle prior.  If this matches ours, the 'diffusion'
    inverse IS empirical Bayes deconvolution and the novelty is elsewhere."""
    J = K.shape[0]
    fs = np.asarray(f_samp[:J], dtype=np.float64)
    obs_mass = K @ pi
    post = K * pi[None, :] / np.maximum(obs_mass[:, None], 1e-300)
    F_hat = fs @ post                          # length F
    z = np.power(1.0 - p, np.arange(1, F + 1, dtype=np.float64))
    p0 = float(z @ pi)
    if 0.0 < p0 < 1.0:
        n0 = float(d_obs) * p0 / (1.0 - p0)
        F_hat = F_hat + n0 * (z * pi) / p0
    extra = float(np.asarray(f_samp[J:]).sum())   # observed j beyond J -> last bin
    return _fold(F_hat, extra)


# ---------------------------------------------------------------------------
# Smoothed-Bayesian linear estimator (JASA 2025)
# ---------------------------------------------------------------------------
def smooth_bayes_params(f_samp: np.ndarray, p: float, n: int):
    """Fit Gamma(a0, b0) abundance prior by moment matching on the observed
    singleton/doubleton ratio.  Returns (a0, b0).

    Under Gamma(a0,b0)-Poisson, g ~ NegBin(a0, b0/(b0+p)), so
        E[g] = a0/b0 * p,   Var[g] = a0/b0 * p (1 + p/b0).
    We match the observed mean and variance of the POSITIVE counts with the
    zero-truncated NB, solved by a short fixed-point iteration.
    """
    f_samp = np.asarray(f_samp, dtype=np.float64)
    d_obs = float(f_samp.sum())
    if d_obs <= 0:
        return 1.0, 1.0
    j = np.arange(1, len(f_samp) + 1, dtype=np.float64)
    mu = float(np.dot(j, f_samp) / d_obs)
    var = float(np.dot((j - mu) ** 2, f_samp) / d_obs)
    # start from the untruncated moment estimates
    b0 = max(p * mu / max(var - mu, 1e-6), 1e-3) if var > mu else 1.0
    a0 = max(mu * b0 / p, 0.2)
    for _ in range(40):
        q0 = (b0 / (b0 + p)) ** a0             # P(g=0)
        # zero-truncated NB moments (numeric over a bounded support)
        jj = np.arange(1, 512, dtype=np.float64)
        logpmf = (gammaln(jj + a0) - gammaln(a0) - gammaln(jj + 1.0)
                  + a0 * math.log(b0 / (b0 + p)) + jj * math.log(p / (b0 + p)))
        w = np.exp(np.clip(logpmf, -745.0, 0.0))
        s = float(w.sum())
        if s <= 0:
            break
        w /= s
        mt = float(np.dot(jj, w))
        vt = float(np.dot((jj - mt) ** 2, w))
        # adjust b0 to move the truncated mean toward the observed mean
        if abs(mt - mu) < 1e-6:
            break
        b0 *= max(min(mt / max(mu, 1e-6), 2.0), 0.5)
        b0 = float(np.clip(b0, 1e-3, 1e5))
        a0 = float(np.clip(a0, 0.2, 50.0))
    return a0, b0


def _truncated_p0(a0: float, b0: float, p: float) -> float:
    """P(g = 0 | type exists) under a Gamma(a0,b0)-Poisson prior TRUNCATED to
    f >= 1 (the paper's prior excludes zero-count types by definition).

    For the untruncated NB, G(s) = (b0/(b0+1-s))^a0, so
        G(1-p) = (b0/(b0+p))^a0,   NB(0) = (b0/(b0+1))^a0,
        P0_trunc = (G(1-p) - NB(0)) / (1 - NB(0)).
    Using G(1-p) directly is wrong: it counts types that do not exist.
    """
    g_unobs = (b0 / (b0 + p)) ** a0
    nb_zero = (b0 / (b0 + 1.0)) ** a0
    den = 1.0 - nb_zero
    if den <= 1e-12:
        return 0.0
    return float(np.clip((g_unobs - nb_zero) / den, 0.0, 1.0 - 1e-9))


def ndv_smooth_bayes(a0: float, b0: float, p: float, d_obs: int) -> float:
    return float(d_obs) / max(1.0 - _truncated_p0(a0, b0, p), 1e-9)


def spectrum_smooth_bayes(g: np.ndarray, p: float, a0: float,
                          b0: float, d_obs: int) -> np.ndarray:
    """Linear reverse posterior mean under the Gamma--Poisson prior.

        E[f | g] = g + (1-p) * (a0 + g) / (b0 + p)

    This is exactly the smoothed-Bayesian linear estimator of Beraha et al.
    (JASA 2025) specialized to the binomial thinning channel.  It uses the
    SAME posterior the paper calls a 'reverse diffusion step', which makes it
    the decisive comparator for the necessity question.
    """
    g = np.asarray(g, dtype=np.int64)
    obs = g[g >= 1]
    if obs.size == 0:
        return np.zeros(H_CAP, dtype=np.float64)
    mean_f = obs.astype(np.float64) + (1.0 - p) * (a0 + obs) / (b0 + p)
    est = np.clip(np.round(mean_f), 1, None)
    F_hat = np.bincount(np.clip(est, 1, H_CAP).astype(np.int64),
                        minlength=H_CAP + 1)[1:H_CAP + 1].astype(float)
    # unseen mass from the zero-truncated NB prior
    p0 = _truncated_p0(a0, b0, p)
    if 0.0 < p0 < 1.0:
        n0 = max(float(d_obs) * p0 / (1.0 - p0), 0.0)
        F_hat[0] += n0
    return np.clip(F_hat, 0.0, None)


# ---------------------------------------------------------------------------
# downstream functionals: what a full histogram buys that a scalar cannot
# ---------------------------------------------------------------------------
def downstream_functionals(F: np.ndarray) -> dict:
    """Functionals of a FoF spectrum that a scalar NDV estimate cannot give."""
    F = np.asarray(F, dtype=np.float64)
    j = np.arange(1, len(F) + 1, dtype=np.float64)
    ndv = float(F.sum())
    N = float(np.dot(j, F))
    out = {
        "ndv": ndv,
        "self_join": float(np.dot(j ** 2, F)),       # SUM f_v^2, join / duplicate mass
        "rare_frac": float(F[:5].sum() / ndv) if ndv > 0 else float("nan"),
    }
    if N > 0:
        prob = (j / N)
        out["entropy"] = float(-np.dot(F * prob, np.log(np.maximum(prob, 1e-300))))
    else:
        out["entropy"] = float("nan")
    return out


def functional_relerrs(F_true: np.ndarray, F_hat: np.ndarray) -> dict:
    t = downstream_functionals(F_true)
    h = downstream_functionals(F_hat)
    out = {}
    for k in ("ndv", "self_join", "rare_frac", "entropy"):
        out[k] = _relerr(h[k], t[k])
    return out
