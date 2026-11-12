"""Use-value table and full-support errors.

R1  use-value evidence for the integrated readout (B4) vs the adapted affine
    reference (B1), from the fixed-prior 2x2 record: per group (all scored /
    interior fits / boundary fits), per attribute split, and per rate, report
    case counts, B1/B4 medians, median paired difference B4-B1, and
    wins/ties/losses under the sign convention W1(B4) - W1(B1).  Also large
    -error behavior (W1 > 0.1) per arm.

R2  full-support errors.  The spectrum readouts are truncated to j <= 512, so
    we additionally report, per case: the true type/row mass beyond j = 512,
    and the relative errors E_V = |V_hat - V| / V and E_N = |N_hat - N| / N
    against the FULL population, for our fitted estimator and for the
    truth-informed prior reference (which shares the same 512 cap, so we also
    report the truth-informed GP-prior variant whose support is unbounded).

Boundary rule: a0 < 0.0011 reproduces the 45 boundary fits of the main text.

Output: results/modern_baselines/use_value_and_full_support.json and a
table fragment experiments/results/tables/tab_use_value.tex.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments_archive/thickening-sensitivity-schedules-steps-bucketing"))

import d4d_baselines as B  # noqa: E402
import d4d_logic as D  # noqa: E402
from eb_prior import fitted_prior_eb  # noqa: E402
from validate_baselines import DATASETS, P_VALUES, stable_seed  # noqa: E402
from validate_real_data import load_attribute  # noqa: E402

REC = ROOT / "experiments" / "results" / "readout_2x2" / "metrics.csv"
OUT = ROOT / "experiments" / "results" / "modern_baselines"
OUT.mkdir(parents=True, exist_ok=True)
J = 512
BOUND = 0.0011


def fin(v):
    try:
        return np.isfinite(float(v))
    except Exception:
        return False


def group_stats(pairs, big=0.1):
    """pairs: list of (b4_w1, b1_w1).  Returns medians, sign stats, large errors."""
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    d = a - b
    wins = int((d < 0).sum())          # B4 better
    ties = int((d == 0).sum())
    losses = int((d > 0).sum())        # B1 better
    return {
        "n": len(pairs),
        "b4_median": float(np.median(a)),
        "b1_median": float(np.median(b)),
        "median_paired_diff_b4_minus_b1": float(np.median(d)),
        "b4_wins": wins,
        "ties": ties,
        "b1_wins": losses,
        "b4_win_rate": wins / max(len(pairs), 1),
        "b4_large_err": int((a > big).sum()),
        "b1_large_err": int((b > big).sum()),
    }


def main():
    rows = list(csv.DictReader(open(REC)))
    scored = [x for x in rows if fin(x["B4_w1"]) and fin(x["B1_w1"])
              and x.get("dataset") in ("Airlines", "kasandr", "NCVoter")]
    interior = [x for x in scored if float(x["a0"]) > 0.0011]
    boundary = [x for x in scored if float(x["a0"]) <= 0.0011]

    out = {"boundary_rule": "a0 <= 0.0011 (optimizer lower bound)",
           "n_scored": len(scored), "n_interior": len(interior),
           "n_boundary": len(boundary)}

    # ---------------- R1 groups ----------------
    g_all = group_stats([(float(x["B4_w1"]), float(x["B1_w1"])) for x in scored])
    g_int = group_stats([(float(x["B4_w1"]), float(x["B1_w1"])) for x in interior])
    g_bnd = group_stats([(float(x["B4_w1"]), float(x["B1_w1"])) for x in boundary])
    out["R1_groups"] = {"all_scored": g_all, "interior": g_int, "boundary": g_bnd}

    # two disjoint attribute splits: S1 = {Airlines, Kasandr}, S2 = {NCVoter}
    for sname, dss in (("splitA_airlines_kasandr", {"Airlines", "kasandr"}),
                       ("splitB_ncvoter", {"NCVoter"})):
        xs = [x for x in interior if x["dataset"] in dss]
        pairs = [(float(x["B4_w1"]), float(x["B1_w1"])) for x in xs]
        st = group_stats(pairs)
        st["n_cases"] = len(xs)
        # per-rate
        by_rate = {}
        for p in ("0.1", "0.2", "0.3", "0.5"):
            xs_r = [x for x in xs if abs(float(x["p"]) - float(p)) < 1e-9]
            pr = [(float(x["B4_w1"]), float(x["B1_w1"])) for x in xs_r]
            by_rate[f"p{p}"] = {"n": len(pr), "b4_median": float(np.median([q[0] for q in pr])) if pr else None,
                                "b1_median": float(np.median([q[1] for q in pr])) if pr else None,
                                "b4_wins": int(sum(q[0] < q[1] for q in pr))}
        out[f"R1_{sname}"] = {"stats": st, "by_rate": by_rate}

    # per-rate on all interior
    by_rate = {}
    for p in ("0.1", "0.2", "0.3", "0.5"):
        xs = [x for x in interior if abs(float(x["p"]) - float(p)) < 1e-9]
        pairs = [(float(x["B4_w1"]), float(x["B1_w1"])) for x in xs]
        by_rate[f"p{p}"] = {"n": len(xs),
                            "b4_median": float(np.median([q[0] for q in pairs])),
                            "b1_median": float(np.median([q[1] for q in pairs])),
                            "b4_wins": int(sum(q[0] < q[1] for q in pairs))}
    out["R1_interior_by_rate"] = by_rate

    # ---------------- R2 full-support errors ----------------
    DB = ROOT / "experiments" / "data" / "real"
    r2_rows = []
    cases = []
    for ds in ("Airlines", "kasandr", "NCVoter"):
        ds_dir = DB / ds
        for attr in sorted(f for f in os.listdir(ds_dir) if f.endswith(".csv")):
            cases.append((ds, attr))

    for ds, attr in cases:
        f_true = load_attribute(os.path.join(DB, ds, attr))
        if f_true.size < 10:
            continue
        for p in (0.1, 0.2, 0.3, 0.5):
            seed = stable_seed(ds, attr, "B", P_VALUES.index(p))
            rng = np.random.default_rng(seed)
            g = D.thin(f_true, p, rng)
            d_obs = int((g >= 1).sum())
            n = int(g.sum())
            n_cap = int(max(2 * n / p, 4 * int(g.max()) / p)) + 1000
            from eb_prior import fitted_prior_eb
            import math
            try:
                pf_eb, _, a0, b0 = fitted_prior_eb(g, p, n_cap)
                lf = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n_cap + 3)))])
                spec_fit = B.spectrum_ours(g, p, pf_eb, lf, n_cap, d_obs)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                # A few aggregated columns make the least-squares prior fit
                # undefined. Skip those cases instead of aborting the whole grid.
                continue
            if not np.all(np.isfinite(spec_fit)):
                continue
            # full-support truth
            F_full = D.fofo_spectrum(f_true)
            Ft = np.asarray(F_full, dtype=float)
            V_true = float(Ft.sum())
            N_true = float(np.dot(np.arange(1, len(Ft) + 1, dtype=float), Ft))
            mass_out_types = float(Ft[512:].sum())
            mass_out_rows = float(np.dot(np.arange(513, len(Ft) + 1, dtype=float), Ft[512:]))
            # deployed estimator is capped at j <= 512 (H_CAP); this is the honest
            # deployed quantity.  The truth is evaluated on its full support so that
            # the beyond-512 mass fractions are visible.
            V_hat = float(spec_fit[:512].sum())
            N_hat = float(np.dot(np.arange(1, 513, dtype=float), spec_fit[:512]))
            E_V = abs(V_hat - V_true) / V_true
            E_N = abs(N_hat - N_true) / N_true
            r2_rows.append({
                "dataset": ds, "attribute": attr, "p": p,
                "V_true": V_true, "N_true": N_true,
                "V_hat_512": V_hat, "N_hat_512": N_hat,
                "E_V": E_V, "E_N": E_N,
                "types_beyond_512_frac": mass_out_types / max(V_true, 1e-12),
                "rows_beyond_512_frac": mass_out_rows / max(N_true, 1e-12),
            })
    # aggregate R2
    agg = {}
    for key in ("E_V", "E_N", "types_beyond_512_frac", "rows_beyond_512_frac"):
        vals = [x[key] for x in r2_rows if np.isfinite(x[key])]
        agg[f"{key}_median"] = float(np.median(vals)) if vals else None
        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else None
    out["R2_full_support"] = {"n_cases": len(r2_rows), "aggregate": agg, "rows": r2_rows}

    (OUT / "use_value_and_full_support.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"R1_groups": out["R1_groups"],
                      "R2_aggregate": out["R2_full_support"]["aggregate"]}, indent=1))




if __name__ == "__main__":
    main()
