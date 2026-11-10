"""Stratified per-frequency band coverage on an independent grid.

Question
--------
The camera-ready text reports a pooled 90% per-frequency band coverage of
0.913--0.936 on three seeds and says the check is conditional empirical Bayes.
The analysis validates actual coverage.  Pooled coverage can hide
systematic failure on populated bins if most evaluated bins carry zero true
mass, and the archived summary explicitly excludes prior-parameter estimation
error.  This script re-runs the posterior draws, reports coverage separately by
true bin mass and by frequency index, and perturbs the fitted prior parameters
to quantify how much coverage moves when the prior is misspecified.

Protocol (declared in the outputs)
----------------------------------
  * synthetic Zipf population, n = 100000, vocab = 30000
  * alpha in {1.0, 1.5, 2.0}, seed in {0, 1, 2}, retention p = 0.1
  * prior fitted once from the thinned sample by fitted_prior_eb
  * K = 120 independent posterior population vectors per configuration
  * evaluation support j = 1..512, matching the W1 support in the paper
  * band = pointwise 5% and 95% quantiles of the K spectra
  * target = spectrum of the observed types in the true population

This is an independent grid.  It is NOT a recomputation of
results/thickening-plausible-set-validation/intervals_three_layer.json, whose
population construction is not archived as a script.  Coverage values from the
two grids must not be presented as one pooled estimate.

Outputs
-------
  experiments/results/band_stratified_coverage/summary.json
  experiments/results/band_stratified_coverage/bins.csv
  experiments/results/band_stratified_coverage/report.md

Run with:
  python3 \
      experiments/code/analyze_band_stratified.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from d4d_logic import analytic_posterior, fofo_spectrum, thin, zipf_population, _logfact
from eb_prior import _nb_mass, fitted_prior_eb

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(ROOT, "experiments", "results", "band_stratified_coverage")

N_ROWS = 100000
VOCAB = 30000
P = 0.1
K = 120
J = 512
ALPHAS = (1.0, 1.5, 2.0)
SEEDS = (0, 1, 2)
DELTAS = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)


def mass_layer(t):
    if t <= 0:
        return "zero_mass"
    if t <= 10:
        return "rare_1_10"
    if t <= 100:
        return "mid_11_100"
    return "head_gt_100"


def freq_layer(j):
    if j == 1:
        return "j_eq_1"
    if j <= 5:
        return "j_2_5"
    if j <= 20:
        return "j_6_20"
    return "j_21_512"


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan"), float("nan")]
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def draw_spectra(c_obs, p, prior_f, logfact, N, rng, k=K):
    """K posterior population vectors, returned as spectra on j = 1..J.

    Uses inverse-CDF sampling, which is distributionally identical to
    rng.choice(fsup, p=prob) but avoids repeating the same categorical draw.
    """
    g_obs = c_obs.astype(np.int64)
    c_k = np.zeros((k, c_obs.size), dtype=np.int64)
    groups = {}
    for gi in np.where(g_obs >= 1)[0]:
        groups.setdefault(int(g_obs[gi]), []).append(int(gi))
    for g, idxs in groups.items():
        fsup, prob = analytic_posterior(g, p, prior_f, logfact, N)
        cum = np.cumsum(prob)
        cum[-1] = 1.0
        u = rng.random((k, len(idxs)))
        c_k[:, np.asarray(idxs, dtype=np.int64)] = fsup[np.searchsorted(cum, u)]
    S = np.zeros((k, J), dtype=np.float64)
    for i in range(k):
        s = fofo_spectrum(c_k[i], j_max=J)
        S[i, :] = s
    return S


def evaluate(c_true, c_obs, prior_f, logfact, N, rng, tag, bins_writer, with_arrays=False):
    S = draw_spectra(c_obs, P, prior_f, logfact, N, rng)
    lo = np.percentile(S, 5, axis=0)
    hi = np.percentile(S, 95, axis=0)
    truth = fofo_spectrum(c_true[c_obs >= 1], j_max=J)
    covered = (truth >= lo) & (truth <= hi)

    by_mass = {}
    by_freq = {}
    for j in range(J):
        lay_m, lay_f = mass_layer(truth[j]), freq_layer(j + 1)
        for d, lay in ((by_mass, lay_m), (by_freq, lay_f)):
            d.setdefault(lay, [0, 0])
            d[lay][1] += 1
            d[lay][0] += int(covered[j])
        if bins_writer is not None:
            bins_writer.writerow({
                "config": tag, "j": j + 1, "true_count": int(truth[j]),
                "band_lo": float(lo[j]), "band_hi": float(hi[j]),
                "covered": int(covered[j]), "mass_layer": lay_m, "freq_layer": lay_f,
            })

    def pack(d):
        return {
            lay: {
                "n_bins": v[1], "n_covered": v[0],
                "coverage": float(v[0] / v[1]) if v[1] else None,
                "wilson_ci95": wilson(v[0], v[1]),
            }
            for lay, v in sorted(d.items())
        }

    nz = [int(covered[j]) for j in range(J) if truth[j] > 0]
    z = [int(covered[j]) for j in range(J) if truth[j] <= 0]
    out = {
        "n_bins": J,
        "n_covered": int(covered.sum()),
        "pooled_coverage": float(covered.mean()),
        "populated_bins_n": len(nz),
        "populated_bins_coverage": float(np.mean(nz)) if nz else None,
        "zero_mass_bins_n": len(z),
        "zero_mass_bins_coverage": float(np.mean(z)) if z else None,
        "by_true_bin_mass": pack(by_mass),
        "by_frequency_index": pack(by_freq),
        "median_band_width": float(np.median(hi - lo)),
    }
    if with_arrays:
        return out, (truth, lo, hi)
    return out


def prior_from_params(a0, b0, N):
    prior_f = np.zeros(N + 1, dtype=np.float64)
    prior_f[:] = _nb_mass(a0, b0, N)
    return prior_f


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(20261110)

    grid = {}
    pooled_mass = {}
    pooled_freq = {}
    arrays = {}
    fbin = open(os.path.join(OUT_DIR, "bins.csv"), "w", newline="")
    bw = csv.DictWriter(fbin, fieldnames=[
        "config", "j", "true_count", "band_lo", "band_hi", "covered",
        "mass_layer", "freq_layer"])
    bw.writeheader()

    for alpha in ALPHAS:
        for seed in SEEDS:
            rng_conf = np.random.default_rng(1000 * int(alpha * 10) + seed + 7)
            c_true = zipf_population(rng_conf, N_ROWS, VOCAB, alpha)
            c_obs = thin(c_true, P, rng_conf)
            N = max(20000, 2 * int(c_true.max()))
            logfact = _logfact(N + 512)
            prior_f, ve, a0, b0 = fitted_prior_eb(c_obs, P, N)
            tag = "alpha%.1f_seed%d" % (alpha, seed)
            r, arr = evaluate(c_true, c_obs, prior_f, logfact, N, rng, tag, bw,
                              with_arrays=True)
            arrays[tag] = arr
            r["alpha"] = alpha
            r["seed"] = seed
            r["fitted_a0"] = float(a0)
            r["fitted_b0"] = float(b0)
            r["n_observed_types"] = int((c_obs >= 1).sum())
            grid[tag] = r
            for lay, v in r["by_true_bin_mass"].items():
                pooled_mass.setdefault(lay, [0, 0])
                pooled_mass[lay][0] += v["n_covered"]
                pooled_mass[lay][1] += v["n_bins"]
            for lay, v in r["by_frequency_index"].items():
                pooled_freq.setdefault(lay, [0, 0])
                pooled_freq[lay][0] += v["n_covered"]
                pooled_freq[lay][1] += v["n_bins"]
            print("%-18s pooled %.4f populated %.4f zero %.4f a0 %.4g" % (
                tag, r["pooled_coverage"], r["populated_bins_coverage"],
                r["zero_mass_bins_coverage"], a0))
    fbin.close()

    # ---- prior-parameter sensitivity on one fixed configuration ----------
    alpha, seed = 1.0, 0
    rng_conf = np.random.default_rng(1000 * int(alpha * 10) + seed + 7)
    c_true = zipf_population(rng_conf, N_ROWS, VOCAB, alpha)
    c_obs = thin(c_true, P, rng_conf)
    N = max(20000, 2 * int(c_true.max()))
    logfact = _logfact(N + 512)
    prior_f, ve, a0, b0 = fitted_prior_eb(c_obs, P, N)
    sens = {}
    for delta in DELTAS:
        p2 = prior_from_params(a0 * (1 + delta), b0 * (1 + delta), N)
        rr = evaluate(c_true, c_obs, p2, logfact, N,
                      np.random.default_rng(4242), "delta%+.2f" % delta, None)
        sens["%+.2f" % delta] = {
            "a0": float(a0 * (1 + delta)),
            "b0": float(b0 * (1 + delta)),
            "pooled_coverage": rr["pooled_coverage"],
            "populated_bins_coverage": rr["populated_bins_coverage"],
            "zero_mass_bins_coverage": rr["zero_mass_bins_coverage"],
            "median_band_width": rr["median_band_width"],
        }
        print("delta %+.2f pooled %.4f populated %.4f" % (
            delta, rr["pooled_coverage"], rr["populated_bins_coverage"]))

    # ---- split-conformal calibration of the populated bins ---------------
    # Diagnosis above: pooled coverage is carried by zero-mass bins.  We therefore
    # fit one additive inflation q on the populated bins of six calibration
    # configurations (seeds 0 and 1) and evaluate it, unchanged, on the three
    # held-out configurations (seed 2).  Configurations are the exchangeable
    # unit. Bins inside one configuration are not independent, so this is an
    # empirical holdout result rather than a per-bin theoretical guarantee.
    calib_tags = ["alpha%.1f_seed%d" % (a, s) for a in ALPHAS for s in (0, 1)]
    hold_tags = ["alpha%.1f_seed%d" % (a, 2) for a in ALPHAS]
    scores = []
    for tag in calib_tags:
        truth, lo, hi = arrays[tag]
        pop = truth > 0
        scores.append(np.maximum(lo[pop] - truth[pop], truth[pop] - hi[pop]))
    scores = np.concatenate(scores)
    n_cal = scores.size
    k = int(math.ceil(0.9 * (n_cal + 1)))
    q = float(np.sort(scores)[min(k, n_cal) - 1])

    def calibrated_stats(tags, shift):
        n_all = n_cov = n_pop = n_pop_cov = 0
        layers = {}
        for tag in tags:
            truth, lo, hi = arrays[tag]
            cov = (truth >= lo - shift) & (truth <= hi + shift)
            n_all += J
            n_cov += int(cov.sum())
            pop = truth > 0
            n_pop += int(pop.sum())
            n_pop_cov += int(cov[pop].sum())
            for j in range(J):
                lay = mass_layer(truth[j])
                layers.setdefault(lay, [0, 0])
                layers[lay][1] += 1
                layers[lay][0] += int(cov[j])
        return {
            "n_configurations": len(tags),
            "n_bins": n_all,
            "pooled_coverage": float(n_cov / n_all),
            "populated_bins_n": n_pop,
            "populated_bins_coverage": float(n_pop_cov / n_pop) if n_pop else None,
            "by_true_bin_mass": {
                lay: {"n_bins": v[1], "coverage": float(v[0] / v[1]) if v[1] else None}
                for lay, v in sorted(layers.items())},
        }

    # Layer-wise variant. One inflation per true-mass layer, fitted only on the
    # calibration configurations. This tests whether the residual failure of
    # the single-scalar band on high-count bins is a scale problem.
    layer_scores = {}
    for tag in calib_tags:
        truth, lo, hi = arrays[tag]
        for j in range(J):
            lay = mass_layer(truth[j])
            layer_scores.setdefault(lay, []).append(
                max(float(lo[j] - truth[j]), float(truth[j] - hi[j])))
    q_layer = {}
    for lay, ss in layer_scores.items():
        ss = np.sort(np.asarray(ss, dtype=float))
        kk = int(math.ceil(0.9 * (ss.size + 1)))
        q_layer[lay] = float(ss[min(kk, ss.size) - 1])

    def layerwise_stats(tags):
        layers = {}
        n_pop = n_pop_cov = 0
        widths = []
        for tag in tags:
            truth, lo, hi = arrays[tag]
            shift = np.array([q_layer[mass_layer(t)] for t in truth])
            cov = (truth >= lo - shift) & (truth <= hi + shift)
            pop = truth > 0
            n_pop += int(pop.sum())
            n_pop_cov += int(cov[pop].sum())
            widths.append(float(np.median((hi + shift) - (lo - shift))))
            for j in range(J):
                lay = mass_layer(truth[j])
                layers.setdefault(lay, [0, 0])
                layers[lay][1] += 1
                layers[lay][0] += int(cov[j])
        return {
            "n_configurations": len(tags),
            "populated_bins_n": n_pop,
            "populated_bins_coverage": float(n_pop_cov / n_pop) if n_pop else None,
            "median_band_width": float(np.median(widths)),
            "by_true_bin_mass": {
                lay: {"n_bins": v[1], "coverage": float(v[0] / v[1]) if v[1] else None}
                for lay, v in sorted(layers.items())},
        }

    layerwise = {
        "q_by_layer": q_layer,
        "n_calibration_scores_by_layer": {k: len(v) for k, v in layer_scores.items()},
        "calibrated_band": "[lo - q_layer, hi + q_layer]",
        "holdout": layerwise_stats(hold_tags),
        "calibration_in_sample": layerwise_stats(calib_tags),
        "scope": (
            "Layer-wise inflations are fitted on six configurations. Layers with "
            "few calibration bins have unstable quantiles, so this is a "
            "diagnostic of scale sensitivity rather than a deployable guarantee."),
    }
    print("layerwise q =", {k: round(v, 2) for k, v in q_layer.items()})
    print("layerwise holdout populated %.4f" % layerwise["holdout"]["populated_bins_coverage"])

    conformal = {
        "calibration_configurations": calib_tags,
        "holdout_configurations": hold_tags,
        "target_level": 0.9,
        "score": "max(lo - true, true - hi) on populated bins only",
        "n_calibration_scores": int(n_cal),
        "conformal_quantile_q": q,
        "calibrated_band": "[lo - q, hi + q]",
        "width_note": "one scalar additive inflation, no per-bin refitting",
        "holdout_raw": calibrated_stats(hold_tags, 0.0),
        "holdout_calibrated": calibrated_stats(hold_tags, q),
        "calibration_in_sample_calibrated": calibrated_stats(calib_tags, q),
        "exchangeability_unit": "configuration, not frequency bin",
        "guarantee_scope": (
            "Empirical holdout coverage on three unseen configurations. "
            "Bins within a configuration are dependent, so no simultaneous "
            "per-bin guarantee is claimed."),
        "layerwise_variant": layerwise,
    }
    print("\nconformal q=%.3f | holdout populated raw %.4f -> calibrated %.4f" % (
        q, conformal["holdout_raw"]["populated_bins_coverage"],
        conformal["holdout_calibrated"]["populated_bins_coverage"]))

    def pack_pooled(d):
        return {
            lay: {"n_bins": v[1], "n_covered": v[0],
                  "coverage": float(v[0] / v[1]) if v[1] else None,
                  "wilson_ci95": wilson(v[0], v[1])}
            for lay, v in sorted(d.items())
        }

    pooled_all_cov = sum(r["n_covered"] for r in grid.values())
    pooled_all_n = sum(r["n_bins"] for r in grid.values())
    summary = {
        "protocol": {
            "question": "stratified validation of the 90% per-frequency credible band",
            "population": "synthetic Zipf, n=100000, vocab=30000",
            "alphas": list(ALPHAS), "seeds": list(SEEDS),
            "retention_p": P, "posterior_draws_K": K,
            "evaluation_support": "j=1..512, the same truncation as the W1 metric",
            "band": "pointwise 5th and 95th percentiles of the K posterior spectra",
            "target": "spectrum of observed types in the true population",
            "prior": "fitted_prior_eb from the thinned sample, fitted once per configuration",
            "independent_grid": (
                "This grid is not a recomputation of intervals_three_layer.json. "
                "The two populations differ, so the coverage values are not pooled."),
            "non_independence_caveat": (
                "Frequency bins inside one configuration are dependent. Wilson "
                "intervals are descriptive and are not a simultaneous guarantee."),
            "uncertainty_included": "posterior count draws conditional on the fitted prior",
            "uncertainty_excluded": (
                "unknown abundance of types absent from the sample, and prior "
                "misspecification beyond the explicit parameter perturbation below"),
            "object_isolation": (
                "This is per-frequency band coverage on j<=512. It does not "
                "validate the 95% per-type count interval or the L1 model-content figure."),
        },
        "grid": grid,
        "pooled_over_9_configurations": {
            "n_bins": pooled_all_n,
            "n_covered": pooled_all_cov,
            "coverage": float(pooled_all_cov / pooled_all_n),
            "by_true_bin_mass": pack_pooled(pooled_mass),
            "by_frequency_index": pack_pooled(pooled_freq),
            "per_config_pooled_coverage": [r["pooled_coverage"] for r in grid.values()],
            "per_config_populated_coverage": [
                r["populated_bins_coverage"] for r in grid.values()],
        },
        "prior_parameter_sensitivity": {
            "base_config": "alpha1.0_seed0",
            "perturbation": "a0 and b0 jointly scaled by 1+delta, same sample and same draw seed",
            "results": sens,
        },
        "conformal_calibration": conformal,
        "headline": (
            "Pooled coverage is near nominal, but populated frequency bins are "
            "under-covered. A single conformal inflation fitted on six "
            "configurations is evaluated on three held-out configurations."),
    }

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    L = []
    L.append("# Stratified per-frequency band coverage (new R3 evidence)\n")
    L.append("Independent grid. Population Zipf n=100000, vocab=30000, p=0.1, K=120, j<=512.\n")
    L.append("## Per configuration\n")
    L.append("| alpha | seed | pooled | populated bins | zero-mass bins | n populated | a0 |")
    L.append("|---|---|---|---|---|---|---|")
    for tag, r in grid.items():
        L.append("| %.1f | %d | %.4f | %.4f | %.4f | %d | %.4g |" % (
            r["alpha"], r["seed"], r["pooled_coverage"],
            r["populated_bins_coverage"], r["zero_mass_bins_coverage"],
            r["populated_bins_n"], r["fitted_a0"]))
    L.append("")
    pp = summary["pooled_over_9_configurations"]
    L.append("## Pooled over %d bins (%d configurations)\n" % (pp["n_bins"], len(grid)))
    L.append("Overall coverage %.4f.\n" % pp["coverage"])
    L.append("### By true bin mass\n")
    L.append("| layer | n bins | covered | coverage | Wilson CI95 |")
    L.append("|---|---|---|---|---|")
    for lay, v in pp["by_true_bin_mass"].items():
        L.append("| %s | %d | %d | %.4f | [%.4f, %.4f] |" % (
            lay, v["n_bins"], v["n_covered"], v["coverage"], *v["wilson_ci95"]))
    L.append("")
    L.append("### By frequency index\n")
    L.append("| layer | n bins | covered | coverage | Wilson CI95 |")
    L.append("|---|---|---|---|---|")
    for lay, v in pp["by_frequency_index"].items():
        L.append("| %s | %d | %d | %.4f | [%.4f, %.4f] |" % (
            lay, v["n_bins"], v["n_covered"], v["coverage"], *v["wilson_ci95"]))
    L.append("")
    L.append("## Prior parameter sensitivity\n")
    L.append("| delta | a0 | b0 | pooled | populated | median width |")
    L.append("|---|---|---|---|---|---|")
    for d, v in sens.items():
        L.append("| %s | %.4g | %.4g | %.4f | %.4f | %.3f |" % (
            d, v["a0"], v["b0"], v["pooled_coverage"],
            v["populated_bins_coverage"], v["median_band_width"]))
    L.append("")
    L.append("## Split-conformal calibration of populated bins\n")
    L.append("Calibration on %d configurations, holdout on %d, q = %.3f.\n" % (
        len(conformal["calibration_configurations"]),
        len(conformal["holdout_configurations"]), q))
    L.append("| holdout | pooled | populated | zero_mass | rare | mid | head |")
    L.append("|---|---|---|---|---|---|---|")
    for lab in ("holdout_raw", "holdout_calibrated"):
        v = conformal[lab]
        lm = v["by_true_bin_mass"]
        def g(k):
            return ("%.4f" % lm[k]["coverage"]) if k in lm else "n/a"
        L.append("| %s | %.4f | %.4f | %s | %s | %s | %s |" % (
            lab, v["pooled_coverage"], v["populated_bins_coverage"],
            g("zero_mass"), g("rare_1_10"), g("mid_11_100"), g("head_gt_100")))
    L.append("")
    L.append("Fitted in-sample populated coverage %.4f.\n" %
             conformal["calibration_in_sample_calibrated"]["populated_bins_coverage"])
    L.append("")
    lw = conformal["layerwise_variant"]
    L.append("### Layer-wise inflations\n")
    L.append("q by layer: %s\n" % ", ".join(
        "%s=%.2f" % (k, v) for k, v in sorted(lw["q_by_layer"].items())))
    L.append("| split | populated | median width | zero_mass | rare | mid | head |")
    L.append("|---|---|---|---|---|---|---|")
    for lab in ("calibration_in_sample", "holdout"):
        v = lw[lab]
        lm = v["by_true_bin_mass"]
        def g2(k):
            return ("%.4f" % lm[k]["coverage"]) if k in lm else "n/a"
        L.append("| %s | %.4f | %.2f | %s | %s | %s | %s |" % (
            lab, v["populated_bins_coverage"], v["median_band_width"],
            g2("zero_mass"), g2("rare_1_10"), g2("mid_11_100"), g2("head_gt_100")))
    L.append("")
    L.append(lw["scope"])
    L.append("")
    L.append("Headline: " + summary["headline"])
    L.append("")
    L.append("Caveats: " + summary["protocol"]["non_independence_caveat"])
    L.append("")
    L.append(conformal["guarantee_scope"])
    L.append("")
    L.append(summary["protocol"]["object_isolation"])
    with open(os.path.join(OUT_DIR, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    print("\npooled coverage %.4f" % pp["coverage"])
    for lay, v in pp["by_true_bin_mass"].items():
        print("  %-14s n=%4d cov=%.4f CI=%s" % (lay, v["n_bins"], v["coverage"], v["wilson_ci95"]))
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
