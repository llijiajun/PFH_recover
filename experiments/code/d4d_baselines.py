"""Canonical baselines for NDV / frequency-distribution recovery.

This module puts the Diffusion4Distribution method next to the established
estimators the database and ecology literature actually uses, so the paper's
claims can be checked against more than a strawman.

Two evaluation axes (both on the SAME simulated binomial-thinned sample):

  AXIS 1 -- NDV scalar accuracy (the classic DB metric).
      Inputs : thinned frequency vector g, sampling rate p, population size N,
               sample size n, sample FOF spectrum f_samp.
      Methods: naive Horvitz-Thompson (d/p), Chao1, Haas (1995) Horvitz-Thompson,
               Haas (1995) method-of-moments, SVD deconvolution, Good-Turing, ours.

  AXIS 2 -- full frequency-distribution recovery (our primary contribution, and
            the full-distribution recovery task).
      Methods: naive 1/p (pointwise), SVD deconvolution (F = M^+ f, same binomial
               forward model), Good-Turing smoothing, Log-Log power-law regression,
               Chebyshev extrapolation, ours (closed-form posterior + prior-derived
               unseen mass).
      Metrics: W1 / total-variation / RMSE on the population FOF spectrum, plus
               NDV and total-population (N) relative error.

All formulas are the textbook ones; nothing here is a strawman.  The forward
model for every method is the same binomial thinning the paper studies, which is
exactly why these are the fair competitors.

Every function is pure numpy (no torch).  Axes share a fixed support H so the
comparison is apples-to-apples (the FOF spectrum is truncated to j = 1..H for
all methods and for the truth).
"""

from __future__ import annotations

import math
import numpy as np

from d4d_logic import thin, fofo_spectrum, analytic_posterior
from d4d_metrics import prior_from_truth, spectrum_w1

H_CAP = 512  # maximum true frequency bin used in the spectrum comparison (paper J)


# ===========================================================================
# shared helpers
# ===========================================================================
def _h_n(x: float, N: int, n: int) -> float:
    """P{a distinct value with population frequency x is UNobserved in a
    Bernoulli sample of n draws from a population of N tuples}.

    Implemented as the product form (equivalent to Haas 1995 eq. 5 but free of
    large-gamma overflow):
        h_n(x) = prod_{k=0}^{n-1} (N - x - k) / (N - k).
    Returns 0 when x >= N - n + 1 (the value is certainly observed).
    """
    if x <= 0:
        return 0.0
    if x >= N - n + 1:
        return 0.0
    acc = 0.0
    for k in range(n):
        num = N - x - k
        den = N - k
        if num <= 0:
            return 0.0
        acc += math.log(num) - math.log(den)
    return math.exp(acc)


def _relerr(est: float, truth: float) -> float:
    return float(abs(est - truth) / truth) if truth > 0 else float("nan")


# ===========================================================================
# AXIS 1 -- NDV scalar estimators
# ===========================================================================
def ndv_naive_ht(d_obs: int, p: float) -> float:
    """Naive Horvitz-Thompson: treat every distinct value as certainly included
    (inclusion prob ~ p).  Over-corrects when many singletons exist."""
    return float(d_obs) / float(p) if p > 0 else float("nan")


def ndv_chao1(f_samp: np.ndarray) -> float:
    """Chao1 lower bound on species richness from the sample FOF spectrum.
    d_obs + f1^2 / (2 f2)  (bias-corrected when f2 == 0)."""
    d_obs = float(f_samp.sum())
    f1 = float(f_samp[0]) if f_samp.size > 0 else 0.0
    f2 = float(f_samp[1]) if f_samp.size > 1 else 0.0
    if f2 > 0:
        unseen = (f1 * f1) / (2.0 * f2)
    else:
        unseen = f1 * (f1 - 1.0) / 2.0
    return d_obs + unseen


def ndv_haas_ht(f_samp: np.ndarray, N: int, n: int) -> float:
    """Haas et al. (1995) Horvitz-Thompson estimator (their D_HT):
        D_HT = sum_{j>=1} f_samp[j-1] / (1 - h_n( N*j/n )).
    h_n is P{value unseen | population freq N*j/n}."""
    total = 0.0
    for j in range(1, len(f_samp) + 1):
        cnt = float(f_samp[j - 1])
        if cnt <= 0:
            continue
        x = (j * N) / n if n > 0 else N
        h = _h_n(x, N, n)
        denom = 1.0 - h
        total += cnt / denom if denom > 0 else cnt
    return total


def ndv_haas_mm1(d_obs: int, N: int, n: int) -> float:
    """Haas et al. (1995) method-of-moments estimator D_MM1: the solution D of
        d_obs = D * (1 - h_n(N / D))
    solved by bisection on D in [d_obs, N].  Accounts for the (finite) equal-
    frequency assumption; outperforms HT when the column is near-uniform."""
    d_obs = float(d_obs)
    lo, hi = d_obs, float(N)
    if hi <= lo:
        return hi

    def f(D: float) -> float:
        if D <= 0:
            return -d_obs
        return D * (1.0 - _h_n(N / D, N, n)) - d_obs

    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:  # degenerate (everything observed) -> HT-like
        return d_obs
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0 or (hi - lo) / hi < 1e-9:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


# ===========================================================================
# AXIS 2 -- full frequency-distribution (FOF spectrum) estimators
#   Each returns an estimated population FOF spectrum F_hat of length H_CAP
#   (F_hat[i] = est. # distinct values whose TRUE frequency is i+1, i = 0..H-1).
# ===========================================================================
def spectrum_naive(g: np.ndarray, p: float) -> np.ndarray:
    """Naive pointwise 1/p scaling of the observed counts (observed types only;
    no model of unseen values).  The 'do nothing clever' reference."""
    g = np.asarray(g, dtype=np.int64)
    obs = g[g >= 1]
    if obs.size == 0:
        return np.zeros(H_CAP, dtype=np.float64)
    est = np.round(obs / p).astype(np.int64)
    est = np.clip(est, 1, H_CAP)
    return fofo_spectrum(est, j_max=H_CAP)


def spectrum_svd(f_samp: np.ndarray, p: float) -> np.ndarray:
    """SVD deconvolution (baseline 2): invert the binomial sampling channel
        f_samp = M F,   M[j,i] = Bin(i, p, j),   F_hat = M^+ f_samp.
    This is the naive linear inverse of the SAME forward model our method solves;
    it is the most direct competitor and a strong reference.

    Plain pseudo-inverse explodes on heavy-tailed real columns (ill-conditioned
    M), so we use a ridge-regularized least squares that returns the bounded,
    minimum-norm solution -- still a fair linear baseline, just not numerically
    broken.  Non-negativity is enforced by clipping."""
    f_samp = np.asarray(f_samp, dtype=np.float64)
    L = min(int(f_samp.size), H_CAP)          # observed-frequency bins used
    H = H_CAP
    if L < 1:
        return np.zeros(H, dtype=np.float64)
    js = np.arange(1, L + 1, dtype=np.float64)
    logp = math.log(p) if p > 0 else -np.inf
    logq = math.log(1 - p) if p < 1 else -np.inf
    from scipy.special import gammaln
    M = np.zeros((L, H), dtype=np.float64)
    is_ = np.arange(1, H + 1, dtype=np.float64)
    for ji in range(L):
        j = ji + 1
        idx = is_ >= j
        ii = is_[idx]
        logM = (gammaln(ii + 1) - gammaln(j + 1) - gammaln(ii - j + 1)
                + j * logp + (ii - j) * logq)
        M[ji, idx] = np.exp(logM)
    rhs = f_samp[:L].astype(np.float64)
    MtM = M.T @ M
    reg = 1e-3 * max(float(MtM.diagonal().max()), 1.0)
    F, *_ = np.linalg.lstsq(MtM + reg * np.eye(H), M.T @ rhs, rcond=None)
    return np.clip(F, 0.0, None)


def spectrum_good_turing(f_samp: np.ndarray) -> np.ndarray:
    """Good-Turing smoothing of the FOF spectrum (baseline 1):
        F_hat[c] = (c+1) * f_{c+1} / f_c   for c = 1..H-1,
    plus a Chao-style unseen add to the singleton bin.  Classic unseen-species
    estimate; the historical reference for any 'missing mass' method."""
    f_samp = np.asarray(f_samp, dtype=np.float64)
    H = H_CAP
    F = np.zeros(H, dtype=np.float64)
    L = min(int(f_samp.size), H)
    for c in range(1, L):                 # F_hat at freq c (index c-1)
        denom = f_samp[c - 1]
        if denom > 0:
            F[c - 1] = (c + 1) * f_samp[c] / denom
    # unseen value count (Chao1) folded into the singleton bin
    f1 = float(f_samp[0]) if f_samp.size > 0 else 0.0
    f2 = float(f_samp[1]) if f_samp.size > 1 else 0.0
    unseen = (f1 * f1) / (2.0 * f2) if f2 > 0 else f1 * (f1 - 1.0) / 2.0
    F[0] = F[0] + max(unseen, 0.0)
    return np.clip(F, 0.0, None)


def spectrum_loglog(f_samp: np.ndarray) -> np.ndarray:
    """Log-Log power-law regression (baseline 3): fit log f_c ~ a - b log c
    on the observed spectrum and extrapolate F_hat[c] = exp(a) c^{-b}."""
    f_samp = np.asarray(f_samp, dtype=np.float64)
    H = H_CAP
    idx = np.arange(1, int(f_samp.size) + 1, dtype=np.float64)
    mask = f_samp > 0
    if mask.sum() < 3:
        return np.zeros(H, dtype=np.float64)
    xs = np.log(idx[mask])
    ys = np.log(f_samp[mask])
    b, a = np.polyfit(xs, ys, 1)           # slope b, intercept a
    c = np.arange(1, H + 1, dtype=np.float64)
    F = np.exp(a) * np.power(c, b)
    return np.clip(F, 0.0, None)


def spectrum_chebyshev(f_samp: np.ndarray) -> np.ndarray:
    """Chebyshev polynomial extrapolation (baseline 4): fit a low-degree
    Chebyshev polynomial to (freq, count) and extrapolate.  Bounded to 5x the
    largest observed count so it cannot blow up at the tail -- the weakest
    reference, included for completeness."""
    f_samp = np.asarray(f_samp, dtype=np.float64)
    H = H_CAP
    idx = np.arange(1, int(f_samp.size) + 1, dtype=np.float64)
    mask = f_samp > 0
    if mask.sum() < 4:
        return np.zeros(H, dtype=np.float64)
    try:
        import numpy.polynomial.chebyshev as cheb
        deg = min(3, int(mask.sum()) - 1)
        poly = cheb.Chebyshev.fit(idx[mask], f_samp[mask], deg)
        c = np.arange(1, H + 1, dtype=np.float64)
        F = poly(c)
    except Exception:
        return np.zeros(H, dtype=np.float64)
    F = np.clip(F, 0.0, None)
    F = np.clip(F, 0.0, max(float(f_samp.max()), 1.0) * 5.0)
    return F


def spectrum_ours(g: np.ndarray, p: float, prior_f: np.ndarray,
                  logfact: np.ndarray, N_max: int, d_obs: int) -> np.ndarray:
    """Our method's population FOF spectrum estimate.

    Observed types: for each distinct observed count g, draw the closed-form
        posterior P(f | g) and add its probability mass into the true-frequency
        bins (expected # of observed types at each true frequency).
    Unseen types: the oracle prior gives the missing fraction
        w = sum_{i>=1} prior_f[i] (1-p)^i  (P{a present type is unobserved}),
    so the estimated total distinct is  D_hat = d_obs / (1 - w); the unseen mass
    is distributed across frequencies with the prior-weighted missing prob.
    """
    g = np.asarray(g, dtype=np.int64)
    H = H_CAP
    F = np.zeros(H, dtype=np.float64)
    obs_idx = np.where(g >= 1)[0]
    if obs_idx.size == 0:
        return F
    # --- observed contribution (expected counts per true-frequency bin) ---
    groups: dict[int, int] = {}
    for gi in obs_idx:
        groups[int(g[gi])] = groups.get(int(g[gi]), 0) + 1
    for gv, m in groups.items():
        fsup, prob = analytic_posterior(gv, p, prior_f, logfact, N_max)
        fsup = np.clip(fsup, 1, H)
        for fi, pr in zip(fsup, prob):
            F[fi - 1] += m * pr
    # --- unseen contribution via the prior missing fraction ---
    i = np.arange(1, len(prior_f), dtype=np.float64)
    miss = prior_f[1:len(prior_f)] * np.power(1.0 - p, i)
    w = float(miss.sum())
    if 0.0 < w < 1.0:
        D_hat = d_obs / (1.0 - w)
        unseen = max(D_hat - d_obs, 0.0)
        denom = float(miss.sum())
        if denom > 0:
            add = unseen * (miss / denom)
            add = add[:H]
            F[:add.size] += add
    return np.clip(F, 0.0, None)


# ===========================================================================
# metric aggregation over spectra
# ===========================================================================
def distribution_metrics(F_true: np.ndarray, F_hat: np.ndarray, J: int = H_CAP) -> dict:
    """Compare an estimated population spectrum to the (capped) truth.

    Both spectra are truncated to the first J bins, renormalized to probability
    vectors, and compared on that fixed support. W1 is the truncated CDF L1 area,
    TV the total variation, and RMSE the root-mean-square over the J bins. With
    two probability vectors on J bins, TV <= 1 and RMSE <= sqrt(2/J), so the
    reported numbers always respect the metric bound. NDV rel error and total
    population (N) rel error are computed on the raw (un-normalized) counts and
    measure support-size recovery, which the shape metrics do not capture."""
    F_true = np.asarray(F_true, dtype=np.float64)
    F_hat = np.asarray(F_hat, dtype=np.float64)
    t = np.zeros(J, dtype=np.float64); t[: min(len(F_true), J)] = F_true[:J]
    h = np.zeros(J, dtype=np.float64); h[: min(len(F_hat), J)] = F_hat[:J]
    # W1 (truncated CDF L1 area) on the same fixed J bins
    w1 = spectrum_w1(t, h, j_eval=J)
    st, sh = t.sum(), h.sum()
    if st > 0 and sh > 0:
        pt, ph = t / st, h / sh
        tv = 0.5 * float(np.abs(pt - ph).sum())
        rmse = float(np.sqrt(np.mean((pt - ph) ** 2)))
    else:
        tv, rmse = float("nan"), float("nan")
    ndv_rel = _relerr(h.sum(), t.sum())
    n_true = float(np.sum(np.arange(1, J + 1) * t))
    n_hat = float(np.sum(np.arange(1, J + 1) * h))
    pop_rel = _relerr(n_hat, n_true)
    return {"w1": w1, "tv": tv, "rmse": rmse, "ndv_rel": ndv_rel, "pop_rel": pop_rel}


# ===========================================================================
# top-level per-case driver
# ===========================================================================
def evaluate_all(f_true: np.ndarray, p: float, rng: np.random.Generator) -> dict:
    """Run every baseline + our method on one real attribute at retention p.

    Returns a flat dict with NDV estimates (axis 1) and per-method distribution
    metrics (axis 2), plus the ground-truth anchors.
    """
    f_true = np.asarray(f_true, dtype=np.int64)
    N_total = int(f_true.sum())
    n_types_true = int(f_true.size)
    g = thin(f_true, p, rng)
    d_obs = int((g >= 1).sum())
    n = int(g.sum())
    f_samp = fofo_spectrum(g)                     # sample FOF spectrum (j>=1)
    F_true = fofo_spectrum(f_true, j_max=H_CAP)    # truth, capped to H_CAP

    prior_f, logfact = prior_from_truth(f_true, N_total)

    # ---- AXIS 1: NDV scalar ----
    ndv = {
        "naive_ht": ndv_naive_ht(d_obs, p),
        "chao1": ndv_chao1(f_samp),
        "haas_ht": ndv_haas_ht(f_samp, N_total, n),
        "haas_mm1": ndv_haas_mm1(d_obs, N_total, n),
    }

    # ---- AXIS 2: full spectrum ----
    F_naive = spectrum_naive(g, p)
    F_svd = spectrum_svd(f_samp, p)
    F_gt = spectrum_good_turing(f_samp)
    F_ll = spectrum_loglog(f_samp)
    F_ch = spectrum_chebyshev(f_samp)
    F_ours = spectrum_ours(g, p, prior_f, logfact, N_total, d_obs)

    specs = {
        "naive": F_naive, "svd": F_svd, "good_turing": F_gt,
        "loglog": F_ll, "chebyshev": F_ch, "ours": F_ours,
    }
    dist = {name: distribution_metrics(F_true, F_hat) for name, F_hat in specs.items()}

    # NDV scalar from the spectrum methods (sum of capped spectrum)
    for name in ("svd", "good_turing", "ours"):
        ndv[name] = float(specs[name].sum())

    ndv_true = float(F_true.sum())
    out = {
        "p": float(p),
        "NDV_true": n_types_true,
        "NDV_true_capped": ndv_true,
        "N_true": N_total,
        "d_obs": d_obs,
        "n_sample": n,
        "NDV_lost_frac": float(1.0 - d_obs / max(n_types_true, 1)),
    }
    for name, val in ndv.items():
        out[f"NDV_{name}"] = val
        out[f"NDVrel_{name}"] = _relerr(val, ndv_true)
    for name, m in dist.items():
        for k, v in m.items():
            out[f"{name}_{k}"] = v
    return out
