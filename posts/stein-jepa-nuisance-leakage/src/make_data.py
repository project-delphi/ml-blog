"""Write widget-data/data.js from the study's two CSVs.

Usage: .venv/bin/python posts/stein-jepa-nuisance-leakage/src/make_data.py STUDY_BLOG_DATA_DIR

Keeps only what the two widgets draw: the 3DShapes rows with a uniform content
law and a Gaussian target (the explorer's grid) and the lambda sweep.
"""

import csv
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(__file__).resolve().parent.parent / "widget-data" / "data.js"

NUM = ("d", "sigma", "alpha", "lam", "ema_tau", "n_seeds", "collapsed_frac")
for m in ("content_r2", "leak_r2", "leak_knn_r2"):
    NUM += (m, m + "_lo", m + "_hi")


def num(v):
    return None if v == "" else round(float(v), 4)


summary = []
for r in csv.DictReader(open(src / "summary.csv")):
    if (r["tier"], r["law"], r["prior"]) != ("shapes", "uniform", "gaussian"):
        continue
    row = {"arm": r["arm_name"]}
    row.update({k: num(r[k]) for k in NUM})
    summary.append(row)

lam = []
for r in csv.DictReader(open(src / "lam_sens.csv")):
    row = {"arm": r["arm_name"]}
    for k in ("lam", "lam_rel"):
        row[k] = num(r[k])
    for m in ("leak_r2", "content_r2"):
        for s in ("", "_lo", "_hi"):
            row[m + s] = num(r[m + s])
    lam.append(row)

payload = {"summary": summary, "lam": lam}
out.write_text("window.WK_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")
print(len(summary), "summary rows,", len(lam), "lambda rows ->", out)
