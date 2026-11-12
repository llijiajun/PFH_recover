"""Emit the unknown-support grid as a table fragment.

Reads the archived results only. No experiment is re-run.
Run: python3 make_unknown_support_table.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(
    ROOT, "experiments", "results", "archived_inputs",
    "unknown_support.json",
)
OUT = os.path.join(ROOT, "experiments", "results", "tables", "tab_unknown_support.tex")

FAMILIES = [
    ("zipf", "Zipf"),
    ("mix", "Zipf--uniform mix"),
    ("spiked", "Spiked heavy tail"),
    ("concentrated", "Concentrated Dirichlet"),
    ("geopoor", "Singleton-poor geometric"),
]
PRIORS = [
    ("matched", "Matched"),
    ("fitted", "Fitted"),
    ("misspecified", "Misspecified"),
]


def fnum(v):
    if v is None:
        return "--"
    if v < 0.01:
        return "%.2E" % v
    if v < 10:
        return "%.3f" % v
    return "%.2f" % v


def main():
    with open(SRC) as f:
        data = json.load(f)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Unknown-support and prior-misspecification grid. "
        r"Five population families, three seeds each, medians over seeds. "
        r"Matched uses the true family predictive as an information ceiling, fitted uses the "
        r"two-equation empirical-Bayes fit from the sample alone, and misspecified fits on a "
        r"sample of the opposite family. $\Wone$ is the full-spectrum distance, unseen $\Wone$ "
        r"is the distance on the spectrum of types absent from the sample, $F_1$ rel.\ is the "
        r"singleton-mass relative error, NDV rel.\ and $U$ rel.\ are the relative errors of the "
        r"distinct count and missing row mass. NDV appears here as a robustness diagnostic only "
        r"and is not an estimand of the paper. A dash means the relative error is undefined "
        r"because the concentrated family has no true singletons, the same condition that "
        r"$\mathrm{Err}_1$ requires in the main text. Lower is better.}",
        r"\label{tab:unknown-support}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Family & Prior & $\Wone$ & unseen $\Wone$ & $F_1$ rel. & NDV rel. & $U$ rel. \\",
        r"\midrule",
    ]
    for key, name in FAMILIES:
        s = data[key]["summary"]
        for j, (pkey, pname) in enumerate(PRIORS):
            fam = name if j == 0 else ""
            lines.append(
                "%s & %s & %s & %s & %s & %s & %s \\\\"
                % (
                    fam, pname,
                    fnum(s[pkey + "_w1_median"]),
                    fnum(s[pkey + "_unseen_w1_median"]),
                    fnum(s[pkey + "_f1_rel_median"]),
                    fnum(s[pkey + "_ndv_rel_median"]),
                    fnum(s[pkey + "_U_rel_median"]),
                )
            )
        if key != FAMILIES[-1][0]:
            lines.append(r"\cmidrule(lr){1-7}")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    n = sum(len(v["rows"]) for v in data.values())
    fails = sum(int(v["summary"]["fit_failures"]) for v in data.values())
    print("wrote", OUT)
    print("rows in archive:", n, "prior conditions:", len(PRIORS),
          "total runs:", n * len(PRIORS), "fit failures:", fails)


if __name__ == "__main__":
    main()
