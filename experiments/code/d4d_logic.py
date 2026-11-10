"""Diffusion4Distribution — consolidated, organized core logic.

This module is the single, readable home for the math behind the paper
"Sampling Is Noising: Learning to Invert the Sampling Channel for
Frequency-Histogram Recovery".  It pulls the verified implementations out of the
scattered experimental scripts (thickening.py, sampler.py,
plausible_set_validation.py) and lays them out in dependency order so the logic
can be read top-to-bottom and verified (see the companion notebook
Diffusion4Distribution_logic_walkthrough.ipynb).

Sections
--------
1. Populations            -- Zipf / lognormal / uniform / custom
2. Forward channel        -- binomial thinning  (sampling == noising)
3. FoF spectrum F_j       -- j >= 1 only, F_0 is excluded by definition
4. Bucketing + W1         -- the readout space and the evaluation metric
5. Analytic posterior      -- closed-form P(f | g)  (no training)
6. 90% plausible set       -- sample the posterior, rebucket, take quantiles
7. Learned Thickener       -- set-transformer denoiser over FoF buckets (torch)
8. Count-space sampler     -- ancestral reverse with closed-form per-coordinate draws
9. Schedules              -- geometric / linear / adaptive retention ladders
10. Paper plot style       -- one consistent publication-quality theme

Torch is imported lazily (only sections 7-8 need it) so the analytic half of the
story (the part the paper's plausible-set claim rests on) runs with numpy alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Torch is required only for the learned models (sections 7-8).  We import it
# lazily/guarded so the analytic half of this module (the plausible-set story)
# can be imported and reasoned about even where torch is unavailable.
try:
    import torch
    import torch.nn as nn  # noqa: F401
    _TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    _TORCH = False

_Base = nn.Module if _TORCH else object  # nn.Module when torch present, else plain object

__all__ = [
    "zipf_population", "lognormal_population", "uniform_population",
    "custom_from_frequencies", "thin", "fofo_spectrum", "Bucketizer",
    "w1_distance", "fourier_feats", "analytic_posterior", "build_empirical_prior",
    "plausible_set", "Thickener", "train_thickener", "chain_thickener",
    "baseline_no_thickening", "PointwiseEndpoint", "reverse_draw", "train_endpoint",
    "chain_sample", "make_schedule", "set_style", "CB",
]


# ===========================================================================
# 1. Populations
# ===========================================================================
def zipf_population(rng: np.random.Generator, n: int, vocab: int, alpha: float) -> np.ndarray:
    """Draw `n` elements over `vocab` categories with Zipf(alpha) weights."""
    w = 1.0 / np.power(np.arange(1, vocab + 1, dtype=np.float64), alpha)
    w /= w.sum()
    return rng.multinomial(n, w)


def lognormal_population(rng: np.random.Generator, n: int, vocab: int,
                         sigma: float = 2.0) -> np.ndarray:
    w = rng.lognormal(mean=0.0, sigma=sigma, size=vocab)
    w /= w.sum()
    return rng.multinomial(n, w)


def uniform_population(rng: np.random.Generator, n: int, vocab: int) -> np.ndarray:
    w = np.ones(vocab, dtype=np.float64) / vocab
    return rng.multinomial(n, w)


def custom_from_frequencies(freqs: dict[int, int], vocab: int | None = None) -> np.ndarray:
    """Build a per-type count vector from a {j: F_j} frequency table.

    Only types that actually appear are materialised; the population therefore
    contains NO zero-count types (consistent with the rule `f_i >= 1`, so the
    true spectrum has no F_0 entry by construction).
    """
    if vocab is None:
        vocab = sum(freqs.values())
    c = np.zeros(vocab, dtype=np.int64)
    idx = 0
    for j, Fj in freqs.items():
        for _ in range(int(Fj)):
            if idx >= vocab:
                break
            c[idx] = j
            idx += 1
    return c[:idx] if idx else c


# ===========================================================================
# 2. Forward channel: binomial thinning  (sampling == noising)
# ===========================================================================
def thin(counts: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Binomial thinning: each of c_i elements survives independently with prob p.

    This is the discrete analogue of Gaussian diffusion.  p = 1 recovers the
    population; p = p_obs yields the observed sample.  Types whose every element
    is thinned out become 0 (the "unseen" types) -- they vanish from the sample
    but are NOT part of any spectrum (see fofo_spectrum).
    """
    return rng.binomial(np.asarray(counts, dtype=np.int64), p)


# ===========================================================================
# 3. FoF spectrum F_j  (j >= 1 only, no F_0)
# ===========================================================================
def fofo_spectrum(counts: np.ndarray, j_max: int | None = None) -> np.ndarray:
    """F_j = #{types occurring exactly j times}, for j = 1 .. j_max.

    F_0 (the unseen mass) is EXCLUDED by definition: a frequency-of-frequencies
    spectrum describes the types that are present.  Types with count 0 -- the
    thinned-out species -- never enter the spectrum.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0:
        return np.array([], dtype=np.float64)
    top = int(counts.max()) if j_max is None else j_max
    top = max(top, 1)
    hist = np.bincount(counts, minlength=top + 1)  # 0..top
    return hist[1: top + 1].astype(np.float64)


# ===========================================================================
# 4. Bucketing + W1
# ===========================================================================
@dataclass
class Bucketizer:
    """Singleton bins for j = 1..j_single, then log-spaced bins up to j_cap.

    Singleton bins keep F_1/F_2 directly readable (the tail-error hypothesis);
    the log bins compress the long body into a fixed budget so the denoiser has a
    tractable, permutation-invariant input.
    """

    j_single: int = 16
    n_log_bins: int = 48
    j_cap: int = 4096

    @property
    def n_buckets(self) -> int:
        return self.j_single + self.n_log_bins

    def edges(self) -> np.ndarray:
        edges = list(range(1, self.j_single + 2))          # singleton bins
        lo = float(self.j_single + 1)
        for k in range(1, self.n_log_bins + 1):
            hi = lo * math.exp(math.log(self.j_cap / lo) / self.n_log_bins)
            edges.append(hi)
            lo = hi
        return np.asarray(edges[: self.j_single + self.n_log_bins + 1])

    def bucketize(self, spec: np.ndarray) -> np.ndarray:
        edges = self.edges()
        j = np.arange(1, len(spec) + 1, dtype=np.float64)
        out = np.zeros(self.n_buckets, dtype=np.float64)
        for b in range(self.j_single):                     # exact singleton bins
            if b < len(spec):
                out[b] = spec[b]
        idx = np.searchsorted(edges[self.j_single:], j[self.j_single:],
                             side="right") - 1
        np.add.at(out, self.j_single + np.clip(idx, 0, self.n_log_bins - 1),
                  spec[self.j_single:])
        return out

    def unbucketize_cdf(self, bucket_mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Approximate the integer-line CDF from bucket masses (singletons exact,
        log-bin mass spread geometrically inside the bin)."""
        edges = self.edges()
        js, pmf = [], []
        for b in range(self.j_single):
            js.append(float(b + 1))
            pmf.append(float(bucket_mass[b]))
        for b in range(self.n_log_bins):
            lo, hi = edges[self.j_single + b], edges[self.j_single + b + 1]
            m = float(bucket_mass[self.j_single + b])
            if m <= 0:
                continue
            k_lo, k_hi = int(math.ceil(lo)), int(math.floor(hi))
            if k_hi < k_lo:
                k_hi = k_lo
            ks = np.arange(k_lo, k_hi + 1)
            w = np.exp(-0.35 * (ks - lo))
            w = w / w.sum() * m
            js.extend(ks.tolist())
            pmf.extend(w.tolist())
        js_a = np.asarray(js, dtype=np.float64)
        pmf_a = np.asarray(pmf, dtype=np.float64)
        order = np.argsort(js_a)
        js_a, pmf_a = js_a[order], pmf_a[order]
        cdf = np.cumsum(pmf_a)
        total = cdf[-1] if cdf.size else 1.0
        if total > 0:
            cdf = cdf / total
        return js_a, cdf


def w1_distance(spec_true: np.ndarray, js_hat: np.ndarray, cdf_hat: np.ndarray,
                j_eval: int = 512) -> float:
    """Integral W1 on a truncated support j = 1..j_eval (both CDFs renormalized).

    Truncation keeps the metric meaningful: Zipf-type spectra carry negligible
    mass beyond a few hundred, and an unbounded grid lets reconstruction artifacts
    in the empty tail dominate the integral.
    """
    j_top = int(min(len(spec_true), j_eval))
    jt = np.arange(1, j_top + 1, dtype=np.float64)
    mass = spec_true[:j_top]
    cdf_t = np.cumsum(mass / max(mass.sum(), 1e-12))
    interp = np.interp(jt, js_hat, cdf_hat, left=0.0, right=1.0)
    return float(np.sum(np.abs(interp - cdf_t)))


# ===========================================================================
# 5. Analytic posterior  P(f | g)  (closed form, no training)
# ===========================================================================
def _logfact(n: int) -> np.ndarray:
    """log(k!) for k = 0..n."""
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1, dtype=np.float64)))])


def build_empirical_prior(alpha: float, N: int, vocab: int, n_pops: int = 24,
                          seed: int = 20_000, rng: np.random.Generator | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Oracle empirical prior over the population count f (0..N), pooled.

    In the paper this is the *best-case* prior (it is built from the true
    population shape).  It is used only to demonstrate that the posterior + set
    are well calibrated; the deployed model replaces it with a fitted prior and
    the posterior formula is unchanged.
    """
    rng = rng or np.random.default_rng(seed)
    hist = np.zeros(N + 1, dtype=np.float64)
    for _ in range(n_pops):
        hist += np.bincount(zipf_population(rng, N, vocab, alpha), minlength=N + 1)
    s = hist.sum()
    return hist / s if s > 0 else hist, _logfact(N + 512)


def analytic_posterior(g: int, p: float, prior_f: np.ndarray, logfact: np.ndarray,
                       N: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form posterior P(f | g) with LOCAL support around the mode.

        P(f | g)  ∝  Prior(f) · Bin(g | f, p),   f >= g.

    Support is built locally around its mode (f ≈ g/p) because f spans 0..N and a
    global table is infeasible; the local support is exact for sampling and for
    coverage.  Returns (f_support:int[], prob:float[]) with prob summing to 1.
    """
    p = float(p)
    q = 1.0 - p
    logp, logq = math.log(p), (math.log(q) if q > 0 else -np.inf)
    g = int(g)
    f_mode = max(g, round(g / p))
    sigma = math.sqrt(max(g, 1) * (1 - p) / p)
    f_lo = max(1, g)                                   # need f >= g for Bin(g|f,p)>0
    f_hi = int(min(N, f_mode + 25 * sigma + 500))
    if f_hi < f_lo:
        f_hi = f_lo
    fsup = np.arange(f_lo, f_hi + 1, dtype=np.int64)
    fg = (fsup - g).astype(np.float64)
    # log Bin(g | f, p) = log(f!) - log(g!) - log((f-g)!) + g·log p + (f-g)·log q
    logbin = (logfact[fsup] - logfact[g + 1] - logfact[np.maximum(fg, 0).astype(int)]
              + g * logp + np.where(fg > 0, fg * logq, 0.0))
    w = prior_f[fsup] * np.exp(logbin)
    tot = w.sum()
    prob = (w / tot) if tot > 0 else (np.ones_like(w) / w.size)
    return fsup, prob


# ===========================================================================
# 6. 90% plausible set
# ===========================================================================
def plausible_set(c_true: np.ndarray, c_obs: np.ndarray, p: float, K: int,
                  prior_f: np.ndarray, logfact: np.ndarray, N: int,
                  rng: np.random.Generator):
    """Sample K full posterior population vectors, rebucket into spectra, and
    return the band + calibration diagnostics.

    Returns a dict with:
        band_lo / band_hi : (J,) 5% and 95% spectrum quantiles (the 90% set)
        mean_spec        : (J,) posterior-mean spectrum
        truth_spec       : (J,) spectrum of the *observed* types in the truth
        obs_spec         : (J,) observed spectrum
        coverage95       : empirical fraction of observed types whose true f lies
                           inside the per-type 95% credible interval
        renoise_w1       : W1( mean re-noised spectrum, observed spectrum )
        renoise_base     : W1 between two independent thinnings of the truth
        cov95_line       : dict nom->emp for several nominal levels (calibration)
    """
    g_obs = c_obs.astype(np.int64)
    groups: dict[int, list[int]] = {}
    for gi in np.where(g_obs >= 1)[0]:
        groups.setdefault(int(g_obs[gi]), []).append(int(gi))

    c_k = np.zeros((K, len(c_obs)), dtype=np.int64)
    covered = 0
    n_obs = 0
    for g, idxs in groups.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        fsup, prob = analytic_posterior(g, p, prior_f, logfact, N)
        samp = rng.choice(fsup, size=(K, len(idxs)), p=prob)
        c_k[:, idxs] = samp
        true_f = c_true[idxs].astype(np.int64)
        lo = np.percentile(samp, 2.5, axis=0)
        hi = np.percentile(samp, 97.5, axis=0)
        covered += int(np.sum((true_f >= lo) & (true_f <= hi)))
        n_obs += len(idxs)

    # per-sample spectra -> per-j quantiles
    specs = [fofo_spectrum(c_k[k]) for k in range(K)]
    Jmax = max(len(s) for s in specs)
    S = np.zeros((K, Jmax))
    for k, s in enumerate(specs):
        S[k, : len(s)] = s
    band_lo = np.percentile(S, 5, axis=0)
    band_hi = np.percentile(S, 95, axis=0)
    mean_spec = S.mean(0)

    truth_spec = fofo_spectrum(c_true[g_obs >= 1])
    obs_spec = fofo_spectrum(c_obs)

    # self-consistency: re-noise the plausible populations, compare to observed
    s_obs = fofo_spectrum(c_obs)
    cdf_o = np.cumsum(s_obs / max(s_obs.sum(), 1e-12))
    specs_rn = [fofo_spectrum(thin(c_k[kk], p, rng)) for kk in range(K)]
    maxlen = max(len(s) for s in specs_rn)
    Srn = np.zeros((K, maxlen))
    for kk, s in enumerate(specs_rn):
        Srn[kk, : len(s)] = s
    mean_rn = Srn.mean(0)
    js_m = np.arange(1, maxlen + 1, dtype=np.float64)
    cdf_m = np.cumsum(mean_rn / max(mean_rn.sum(), 1e-12))
    renoise_w1 = w1_distance(s_obs, js_m, cdf_m)

    base = []
    for _ in range(8):
        sa = fofo_spectrum(thin(c_true, p, rng))
        sb = fofo_spectrum(thin(c_true, p, rng))
        jsb = np.arange(1, len(sb) + 1, dtype=np.float64)
        cdfb = np.cumsum(sb / max(sb.sum(), 1e-12))
        base.append(w1_distance(sa, jsb, cdfb))
    renoise_base = float(np.mean(base))

    # calibration line
    levels = np.array([0.5, 0.8, 0.9, 0.95, 0.99])
    emp = []
    for lv in levels:
        lo_q, hi_q = (1 - lv) / 2 * 100, (1 + lv) / 2 * 100
        cov = 0.0
        tot = 0
        for g, idxs in groups.items():
            fsup, prob = analytic_posterior(g, p, prior_f, logfact, N)
            samp = rng.choice(fsup, size=(K, len(idxs)), p=prob)
            true_f = c_true[idxs].astype(np.int64)
            lo = np.percentile(samp, lo_q, axis=0)
            hi = np.percentile(samp, hi_q, axis=0)
            cov += int(np.sum((true_f >= lo) & (true_f <= hi)))
            tot += len(idxs)
        emp.append(cov / max(tot, 1))
    cov95_line = {float(lv): float(e) for lv, e in zip(levels, emp)}

    return {
        "band_lo": band_lo, "band_hi": band_hi, "mean_spec": mean_spec,
        "truth_spec": truth_spec, "obs_spec": obs_spec,
        "coverage95": covered / max(n_obs, 1),
        "renoise_w1": renoise_w1, "renoise_base": renoise_base,
        "cov95_line": cov95_line, "K": K, "p": p,
    }


# ===========================================================================
# 7. Learned Thickener  (set-transformer denoiser over FoF buckets)
# ===========================================================================
def fourier_feats(x: float, k: int = 6) -> np.ndarray:
    ws = np.array([2**i for i in range(k)], dtype=np.float32)
    return np.concatenate([np.sin(ws * x), np.cos(ws * x)]).astype(np.float32)


class _Torch:
    """Lazy torch handle so importing this module never requires torch."""
    _mod = None

    @classmethod
    def get(cls):
        if cls._mod is None:
            import torch
            cls._mod = torch
        return cls._mod


class BucketToken:
    """Precomputed static features of a bucket (independent of the value)."""

    def __init__(self, bucketizer: Bucketizer):
        edges = bucketizer.edges()
        feats = []
        for b in range(bucketizer.n_buckets):
            lo, hi = edges[b], edges[b + 1]
            center_log = math.log1p((lo + hi) / 2 - 1)
            width_log = math.log1p(hi - lo)
            is_single = 1.0 if b < bucketizer.j_single else 0.0
            feats.append([center_log, width_log, is_single])
        self.static = np.asarray(feats, dtype=np.float32)


class Thickener(_Base):
    """Set-transformer over buckets -> per-bucket log1p count at the target level.

    Predicts the FoF spectrum one thickening step ahead from the current
    bucketized spectrum plus the (p_from, p_to) schedule pair.
    """

    def __init__(self, width: int = 128, depth: int = 6, n_buckets: int = 64,
                 p_fourier: int = 4):
        super().__init__()
        self.p_fourier = p_fourier
        in_dim = 3 + 2 * (2 * p_fourier) + 1
        self.in_proj = nn.Linear(in_dim, width)
        layer = nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                  nn.Linear(width, 1))

    def forward(self, bucket_log_counts, p_from, p_to, static):
        torch = _Torch.get()
        pf = torch.tensor(fourier_feats(math.log(p_from + 1e-6), self.p_fourier)).expand(
            bucket_log_counts.shape[0], -1)
        pt = torch.tensor(fourier_feats(math.log(p_to + 1e-6), self.p_fourier)).expand(
            bucket_log_counts.shape[0], -1)
        x = torch.cat([static, pf, pt, bucket_log_counts], dim=-1)
        h = self.in_proj(x)
        h = self.encoder(h.unsqueeze(0)).squeeze(0)
        return self.head(h).squeeze(-1)


@dataclass
class TrainConfig:
    alpha: float = 1.5
    n: int = 100_000
    vocab: int = 30_000
    j_single: int = 16
    n_log_bins: int = 48
    j_cap: int = 4096
    width: int = 128
    depth: int = 6
    steps_train: int = 1500
    batch_pops: int = 8
    lr: float = 3e-4
    p_obs: float = 0.1
    schedule: str = "geometric"
    steps_infer: int = 50
    seed: int = 0
    device: str = "cpu"
    huber_beta: float = 1.0


def _refresh_pairs(cfg: TrainConfig, rng: np.random.Generator, bucketizer: Bucketizer):
    pops = [zipf_population(rng, cfg.n, cfg.vocab, cfg.alpha) for _ in range(cfg.batch_pops)]
    x_b, y_b, pfb, ptb = [], [], [], []
    for c0 in pops:
        p_from = float(rng.uniform(0.01, 0.9))
        p_to = float(rng.uniform(p_from + 0.01, 1.0))
        c1 = thin(c0, p_from, rng)
        s_target = fofo_spectrum(thin(c0, p_to, rng))
        s_from = fofo_spectrum(c1)
        x_b.append(bucketizer.bucketize(s_from))
        y_b.append(bucketizer.bucketize(s_target))
        pfb.append(p_from)
        ptb.append(p_to)
    return x_b, y_b, pfb, ptb


def train_thickener(cfg: TrainConfig):
    """Train one thickener; return (model, metrics)."""
    torch = _Torch.get()
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    bucketizer = Bucketizer(cfg.j_single, cfg.n_log_bins, cfg.j_cap)
    static = torch.tensor(BucketToken(bucketizer).static, device=device)
    model = Thickener(cfg.width, cfg.depth, bucketizer.n_buckets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    huber = torch.nn.SmoothL1Loss(beta=cfg.huber_beta)

    for it in range(1, cfg.steps_train + 1):
        x_b, y_b, pfb, ptb = _refresh_pairs(cfg, rng, bucketizer)
        idx = int(rng.integers(0, len(x_b)))
        x = torch.log1p(torch.tensor(x_b[idx], dtype=torch.float32, device=device)).unsqueeze(-1)
        y = torch.log1p(torch.tensor(y_b[idx], dtype=torch.float32, device=device))
        pred = model(x, pfb[idx], ptb[idx], static)
        loss = huber(pred, y)
        opt.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        loss.backward()
        opt.step()

    # evaluation: chain thickening from p_obs to 1
    eval_rng = np.random.default_rng(10_000 + cfg.seed)
    w1s, f1_errs = [], []
    for _ in range(16):
        c_true = zipf_population(eval_rng, cfg.n, cfg.vocab, cfg.alpha)
        s_true = fofo_spectrum(c_true)
        c_obs = thin(c_true, cfg.p_obs, eval_rng)
        levels = make_schedule(cfg.schedule, cfg.p_obs, cfg.steps_infer, eval_rng)
        cur = fofo_spectrum(c_obs)
        with torch.no_grad():
            for p_from, p_to in zip(levels[:-1], levels[1:]):
                x = torch.log1p(torch.tensor(
                    bucketizer.bucketize(cur), dtype=torch.float32, device=device)).unsqueeze(-1)
                pred = model(x, p_from, p_to, static)
                cur = np.maximum(np.expm1(pred.cpu().numpy()), 0.0)   # invert log1p, keep counts >= 0
        js_hat, cdf_hat = bucketizer.unbucketize_cdf(cur)
        w1s.append(w1_distance(s_true, js_hat, cdf_hat))
        f1_hat = float(np.exp(cur[0])) if cur[0] < 20 else float(cur[0])
        f1_errs.append(abs(f1_hat - s_true[0]) / max(s_true[0], 1e-9))
    return model, {
        "w1_mean": float(np.mean(w1s)), "w1_std": float(np.std(w1s)),
        "f1_relerr": float(np.mean(f1_errs)),
    }


def chain_thickener(model, bucketizer, c_obs, p_obs, schedule_family, steps, device):
    torch = _Torch.get()
    s_true = fofo_spectrum(c_obs)
    levels = make_schedule(schedule_family, p_obs, steps)
    cur = s_true
    static = torch.tensor(BucketToken(bucketizer).static, device=device)
    with torch.no_grad():
        for p_from, p_to in zip(levels[:-1], levels[1:]):
            x = torch.log1p(torch.tensor(
                bucketizer.bucketize(cur), dtype=torch.float32, device=device)).unsqueeze(-1)
            pred = model(x, p_from, p_to, static)
            cur = np.maximum(np.expm1(pred.cpu().numpy()), 0.0)   # invert log1p, keep counts >= 0
    return cur


def baseline_no_thickening(alpha=1.5, n=100_000, vocab=30_000, p_obs=0.1, seed=0):
    """W1 of the raw thinned spectrum (no thickening) vs the true spectrum."""
    rng = np.random.default_rng(seed)
    w1s = []
    for _ in range(16):
        c_true = zipf_population(rng, n, vocab, alpha)
        s_true = fofo_spectrum(c_true)
        s_obs = fofo_spectrum(thin(c_true, p_obs, rng))
        js = np.arange(1, len(s_obs) + 1, dtype=np.float64)
        cdf = np.cumsum(s_obs / max(s_obs.sum(), 1e-12))
        w1s.append(w1_distance(s_true, js, cdf))
    return {"w1_mean": float(np.mean(w1s))}


# ===========================================================================
# 8. Count-space sampler  (Algorithm 1 of the paper: ancestral reverse)
# ===========================================================================
class PointwiseEndpoint(_Base):
    """Pointwise MLP:  E[c0 | k, pi]  via (log1p count, fourier(pi))."""

    def __init__(self, width=128, depth=3, pf=4):
        super().__init__()
        self.pf = pf
        in_dim = 1 + 2 * pf
        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, logk, pi):
        torch = _Torch.get()
        n = logk.shape[0]
        f = torch.tensor(np.stack([fourier_feats(math.log(pi + 1e-6), self.pf)] * n))
        h = self.net(torch.cat([logk, f], dim=-1))
        return self.head(h).squeeze(-1)


def _logbin_table(j_max, n_max):
    lg = np.array([math.lgamma(x + 1) for x in range(int(n_max) + 2)], dtype=np.float64)
    return lg


def reverse_draw(k_arr, m_arr, r, q, rng):
    """Sample j ~ q(j | k, m) ∝ Bin(k; j, r)·Bin(j; m, q), r = pi_t/pi_s <= 1.

    Support j in [k, m]; vectorized by width-bucketing (wide supports handled in
    their own group).  This is the closed-form reverse kernel of Algorithm 1.
    """
    k_arr = np.asarray(k_arr, dtype=np.int64)
    m_arr = np.asarray(m_arr, dtype=np.int64)
    widths = m_arr - k_arr + 1
    out = k_arr.copy()

    lg = _logbin_table(int(np.max(m_arr)) + 64, int(np.max(m_arr)) + 64)
    klg = lg[np.maximum(k_arr, 0)]
    mlg = lg[np.maximum(m_arr, 0)]
    lg_top = len(lg) - 1

    def draw_group(idx, W):
        if len(idx) == 0:
            return
        kn, mn = k_arr[idx], m_arr[idx]
        s = np.arange(W, dtype=np.float64)[None, :]
        jmat = kn[:, None] + s
        valid = s < widths[idx, None]
        # clip the f-index to the table bounds: invalid (padded) entries are
        # masked to -inf below, so their exact index value is irrelevant.
        jmat_idx = np.minimum(np.maximum(jmat, 0), lg_top).astype(int)
        logp = np.where(
            valid,
            (lg[jmat_idx]
             - klg[idx, None] - lg[np.maximum(jmat_idx - kn[:, None], 0)]
             + kn[:, None] * math.log(max(r, 1e-12))
             + jmat * math.log(max(q, 1e-12))
             + mlg[idx, None] - lg[np.maximum(mn[:, None] - jmat_idx, 0)]),
            -np.inf)
        logp -= logp.max(axis=1, keepdims=True)
        pmf = np.exp(logp)
        cdf = np.cumsum(pmf / pmf.sum(axis=1, keepdims=True), axis=1)
        u = rng.random((len(idx), 1))
        out[idx] = kn + np.minimum((u > cdf).sum(axis=1), W - 1)

    WMAX = 32
    draw_group(np.where(widths <= WMAX)[0], WMAX)
    prev = WMAX
    for b in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 1 << 22):
        idx = np.where((widths > prev) & (widths <= b))[0]
        if len(idx):
            draw_group(idx, int(widths[idx].max()))   # pad only to the real support
        prev = b
    return out


def train_endpoint(alpha=1.5, n=100_000, vocab=30_000, steps=1500, seed=0,
                   device="cpu", lr=3e-4, batch=4096):
    torch = _Torch.get()
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    net = PointwiseEndpoint().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    huber = nn.SmoothL1Loss()
    for it in range(1, steps + 1):
        c0 = zipf_population(rng, n, vocab, alpha)
        pi = float(rng.uniform(0.01, 0.95))
        c_t = thin(c0, pi, rng)
        sel = rng.integers(0, vocab, size=batch)
        k = torch.log1p(torch.tensor(c_t[sel], dtype=torch.float32, device=device)).unsqueeze(-1)
        y = torch.log1p(torch.tensor(c0[sel], dtype=torch.float32, device=device))
        loss = huber(net(k, pi), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net, float(loss.item())


def chain_sample(net, c_obs, p_obs, schedule, rng, device, n=None):
    c = c_obs.astype(np.int64)
    if n is None:
        n = int(c.sum())
    for p_from, p_to in zip(schedule[:-1], schedule[1:]):
        k = torch_log1p(c, device)
        with torch.no_grad():
            m = torch_expm1(net(k, p_from)).cpu().numpy()
        m = np.maximum(m, c).astype(np.float64)
        # clamp the predicted population count to a sane support (the endpoint can
        # extrapolate wildly for high-frequency types); this keeps the reverse
        # kernel finite without biasing the bulk of the distribution.
        cap = c + 25.0 * np.sqrt(np.maximum(c, 1) * (1.0 - p_from) / p_from) + 500.0
        m = np.minimum(m, np.maximum(cap, c))
        m = np.minimum(m, n)                       # cannot exceed the population total
        m = np.maximum(m, c).astype(np.int64)
        r = p_from / max(p_to, 1e-9)
        c = reverse_draw(c, m, r, p_to, rng)
    return c


# small torch tensor helpers (kept here so the sampler section stays torch-free at import)
def torch_log1p(arr, device):
    torch = _Torch.get()
    return torch.log1p(torch.tensor(arr, dtype=torch.float32, device=device)).unsqueeze(-1)


def torch_expm1(t):
    torch = _Torch.get()
    return torch.expm1(t)


# ===========================================================================
# 9. Schedules
# ===========================================================================
def make_schedule(family: str, p_obs: float, steps: int,
                  rng: np.random.Generator | None = None):
    """Return list of p levels p_0=p_obs < ... < p_T=1 (thickening direction)."""
    if family == "geometric":
        rho = math.exp(math.log(1e-6) / steps)
        gaps = ((1 - p_obs) * np.power(rho, np.arange(steps)) * (1 - rho)
                / (1 - rho**steps) * steps)
        gaps = np.clip(gaps, 1e-6, None)
        gaps = gaps / gaps.sum() * (1 - p_obs)
    elif family == "linear":
        gaps = np.full(steps, (1 - p_obs) / steps)
    elif family == "adaptive":
        levels, p = [p_obs], p_obs
        for _ in range(steps):
            frac_unseen = (1 - p) ** 1.5
            step = (1 - p) * (0.15 + 0.85 * (1 - frac_unseen)) / steps * 2
            p = min(p + max(step, 1e-6), 1.0)
            levels.append(p)
        return levels
    else:
        raise ValueError(family)
    levels = [p_obs]
    for g in gaps:
        levels.append(min(levels[-1] + float(g), 1.0))
    levels[-1] = 1.0
    return levels


# ===========================================================================
# 10. Paper plot style  (one consistent, publication-quality theme)
# ===========================================================================
CB = {  # colorblind-safe, used everywhere
    "ours": "#0072B2", "ours2": "#56B4E9", "anchor": "#E69F00",
    "none": "#999999", "truth": "#111111", "set": "#2c7fb8",
    "mean": "#d95f02", "point": "#7a0177", "accent": "#009E73",
}

PAPER_RCPARAMS = {
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 160, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
    "axes.titlepad": 4.0, "figure.autolayout": False,
}


def set_style():
    """Apply the paper's matplotlib theme.  Call before plotting."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update(PAPER_RCPARAMS)
