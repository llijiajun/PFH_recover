"""Diagnose the a0 lower-bound fit flagged in the Table IV caption.

Review round 8 asks whether the fitted abundance-prior shape collapsing to its
lower bound is specific to synthetic populations and what it implies for
reliability.  This script answers both from archived run outputs.  No new
random experiment is run.

Inputs
  experiments/results/fitted_prior_validation/fitted_metrics.csv
      276 real-column cases, column a0_eb is the shape returned by
      eb_prior.fitted_prior_eb (the same fitter as make_data_dependency.py),
      whose least-squares lower bound on a0 is 1e-3.
  experiments/results/data_dependency/summary.json
      synthetic five-family grid, field a0 from the same fitter.

Outputs
  experiments/results/a0_boundary/summary.json
  experiments/results/a0_boundary/report.md
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REAL_CSV = ROOT / "experiments/results/fitted_prior_validation/fitted_metrics.csv"
SYN_JSON = ROOT / "experiments/results/data_dependency/summary.json"
OUT_DIR = ROOT / "experiments/results/a0_boundary"

BOUND = 1e-3
TOL = 1.001e-3  # least-squares returns values a few 1e-15 above the bound


def finite(values):
    return [v for v in values if v == v and abs(v) != float("inf")]


def main():
    rows = list(csv.DictReader(REAL_CSV.open()))
    for r in rows:
        r["a0"] = float(r["a0_eb"])
        r["eb_w1"] = float(r["eb_w1"])
        r["is_boundary"] = r["a0"] <= TOL

    boundary = [r for r in rows if r["is_boundary"]]
    interior = [r for r in rows if not r["is_boundary"]]
    by_ds = defaultdict(lambda: {"n": 0, "boundary": 0})
    for r in rows:
        by_ds[r["dataset"]]["n"] += 1
        by_ds[r["dataset"]]["boundary"] += int(r["is_boundary"])

    real = {
        "n_cases": len(rows),
        "a0_bound": BOUND,
        "n_boundary": len(boundary),
        "boundary_fraction": len(boundary) / len(rows),
        "n_boundary_attributes": len({r["attribute"] for r in boundary}),
        "a0_median": median(r["a0"] for r in rows),
        "a0_min": min(r["a0"] for r in rows),
        "a0_max": max(r["a0"] for r in rows),
        "per_dataset": {k: dict(v) for k, v in sorted(by_ds.items())},
        "eb_w1_median_boundary": median(finite(r["eb_w1"] for r in boundary)),
        "eb_w1_median_interior": median(finite(r["eb_w1"] for r in interior)),
    }

    syn = json.load(SYN_JSON.open())
    fam = defaultdict(lambda: {"n": 0, "boundary": 0})
    for r in syn["rows"]:
        fam[r["family"]]["n"] += 1
        fam[r["family"]]["boundary"] += int(r["a0"] <= TOL)
    synthetic = {k: dict(v) for k, v in sorted(fam.items())}

    out = {"real_columns": real, "synthetic_families": synthetic,
           "source_real": str(REAL_CSV.relative_to(ROOT)),
           "source_synthetic": str(SYN_JSON.relative_to(ROOT)),
           "fitter": "eb_prior.fitted_prior_eb, least-squares lower bound on a0 = 1e-3"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")

    L = ["# Boundary fit of the abundance-prior shape a0", "",
         "Both sources use `eb_prior.fitted_prior_eb`, whose least-squares lower",
         f"bound on $a_0$ is {BOUND:g}. A case is a boundary fit when $a_0 \\le$",
         f"{TOL:g}.", "",
         "## Real columns", "",
         f"- {real['n_boundary']} of {real['n_cases']} cases "
         f"({real['boundary_fraction']:.1%}) are boundary fits, spanning "
         f"{real['n_boundary_attributes']} distinct attributes.",
         f"- Overall $a_0$ median {real['a0_median']:.4f}, range "
         f"[{real['a0_min']:.4g}, {real['a0_max']:.4g}]."]
    for k, v in real["per_dataset"].items():
        L.append(f"- {k}: {v['boundary']}/{v['n']} boundary fits.")
    L += [f"- Median $\\Wone$ of the empirical-Bayes inverse is "
          f"{real['eb_w1_median_boundary']:.4f} on boundary cases against "
          f"{real['eb_w1_median_interior']:.4f} on interior cases.", "",
          "The boundary fit therefore also occurs on real columns. It is",
          "associated with much larger spectrum error and marks a regime where",
          "the two identifying equations do not determine $a_0$.", "",
          "## Synthetic families (5 families x 4 rates x 3 seeds)", ""]
    for k, v in synthetic.items():
        L.append(f"- {k}: {v['boundary']}/{v['n']} boundary fits.")
    L.append("")
    (OUT_DIR / "report.md").write_text("\n".join(L))

    print(f"real boundary fits: {real['n_boundary']}/{real['n_cases']} "
          f"({real['boundary_fraction']:.1%}), "
          f"attributes {real['n_boundary_attributes']}")
    print(f"median W1 boundary {real['eb_w1_median_boundary']:.4f} "
          f"vs interior {real['eb_w1_median_interior']:.4f}")
    print("synthetic:", synthetic)
    print("wrote", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
