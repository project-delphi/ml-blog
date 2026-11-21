"""Print every I-94 traffic number the post quotes, from the caches (run from the post dir).

    cd posts/two-cultures-breiman-2001
    OMP_NUM_THREADS=1 ../../.venv-two-cultures/bin/python src/check_traffic.py

A missing cache is computed first (`twocultures.traffic`; roughly 15 minutes from
cold on ten cores). Ranges are over the five seeds; a single number means the model
is unseeded or the seeds agree to the printed precision. Intervals are 95%, from
circular moving blocks of 168 test hours (seven days), per seed; the printed interval runs
from the lowest lower end to the highest upper end over the seeds.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if not Path("index.qmd").exists():
    sys.exit("run from posts/two-cultures-breiman-2001")
sys.path.insert(0, str(Path("src").resolve()))

from twocultures import traffic as T  # noqa: E402

LADDER = [
    "seasonal_naive",
    "persistence",
    "ols",
    "cl",
    "cl_plus",
    "rf",
    "et",
    "hgb",
    "hgb_log",
    "blend",
]


def rng(v, fmt="{:.1f}"):
    lo, hi = fmt.format(np.min(v)), fmt.format(np.max(v))
    return lo if lo == hi else f"{lo} to {hi}"


def seeds(res, m):
    return sorted(set(res.loc[(res["model"] == m) & (res["split"] == "test"), "seed"]))


def gap_str(gs, key="pct"):
    """Gap as a seed range; interval as the widest over seeds (lowest lower end,
    highest upper end).
    """
    lo, hi = min(g[key + "_lo"] for g in gs), max(g[key + "_hi"] for g in gs)
    return f"{rng([g[key] for g in gs], '{:+.1f}')} [{lo:+.1f}, {hi:+.1f}]"


def main():
    res, pr = T.run(), T.probe()
    df = T.load()
    folds, pre, te = T.splits(df)
    y = te["y"].to_numpy()
    cong = T.cong_threshold(df)
    top_cut = float(np.quantile(y, T.TOP_Q))
    P = {
        m: {s: T.pred(res, m, s).reindex(te.index).to_numpy() for s in seeds(res, m)}
        for m in LADDER
    }
    P["cl_pp"] = {0: T.pred(pr, "cl_pp").reindex(te.index).to_numpy()}
    cl, clp = P["cl"][0], P["cl_plus"][0]

    print(
        f"I-94 W, ATR 301. {len(df):,} observed hours from {T.START}; pre-test "
        f"{len(pre):,}, validation folds {[len(v) for _, v in folds]}, test {len(y):,} "
        f"({te.index.min():%Y-%m-%d} to {te.index.max():%Y-%m-%d})."
    )
    print(
        f"Congestion hour: volume >= {cong:.0f} veh/h, the 80th percentile of pre-test "
        f"hours, fixed before scoring.\nBusiest 10% of test hours: volume >= {top_cut:.0f}."
    )

    print("\n== Settings chosen on pooled 2017 walk-forward RMSE (seed 0) ==")
    for m in LADDER + ["cl_pp"]:
        g = (pr if m == "cl_pp" else res).query("model == @m").iloc[0]
        print(f"  {m:15s} val {g['cv_rmse']:6.1f}  {g['params']}")

    print("\n== Test, Jan-Sep 2018 (gaps: % of RMSE, [95% CI]) ==")
    print(
        f"{'model':14s} {'RMSE':>13s} {'MAE':>11s} {'top10':>11s} {'pk15':>9s} "
        f"{'short':>11s} {'congF1':>11s}  {'vs CL':>24s}  {'vs CL+':>24s}"
    )
    rows = {}
    for m in LADDER + ["cl_pp"]:
        sc = [T.scores(p, y, cong) for p in P[m].values()]
        g1 = [T.gap(p, cl, y) for p in P[m].values()]
        g2 = [T.gap(p, clp, y) for p in P[m].values()]
        rows[m] = sc
        col = lambda k, f="{:.1f}": rng([s[k] for s in sc], f)  # noqa: E731
        print(
            f"{m:14s} {col('rmse'):>13s} {col('mae'):>11s} {col('rmse_top10'):>11s} "
            f"{col('peak_within15', '{:.2f}'):>9s} {col('peak_shortfall'):>11s} "
            f"{col('cong_f1', '{:.3f}'):>11s}  "
            f"{gap_str(g1) if m != 'cl' else '':>24s}  "
            f"{gap_str(g2) if m != 'cl_plus' else '':>24s}"
        )
    print(
        "  (pk15: share of busiest-10% hours within 15%; short: mean under-forecast "
        "there, veh/h; cl_pp is post-hoc)"
    )
    mean = {m: np.mean([s["rmse"] for s in rows[m]]) for m in rows}
    best = min(T.TREES + ("blend",), key=mean.get)
    o = mean["ols"]
    print(
        f"Best tree model on test: {best}. Share of the OLS-to-{best} RMSE gap closed: "
        f"CL {(o - mean['cl']) / (o - mean[best]):.0%}, CL+ "
        f"{(o - mean['cl_plus']) / (o - mean[best]):.0%}, CL++ (post-hoc) "
        f"{(o - mean['cl_pp']) / (o - mean[best]):.0%}."
    )
    print(
        f"Congestion recall/precision, CL {rows['cl'][0]['cong_recall']:.3f}/"
        f"{rows['cl'][0]['cong_precision']:.3f}; {best} "
        f"{rng([s['cong_recall'] for s in rows[best]], '{:.3f}')}/"
        f"{rng([s['cong_precision'] for s in rows[best]], '{:.3f}')}."
    )
    gb = [T.gap(p, P["cl_pp"][0], y) for p in P[best].values()]
    print(f"{best} vs CL++ (post-hoc): {gap_str(gb)}")

    print(
        "\n== Learning curve: test RMSE, training on the last k months (settings fixed) =="
    )
    lc = T.learning_curve()
    models = list(dict.fromkeys(lc["model"]))
    print(f"{'months':>7s} {'n_train':>8s} " + " ".join(f"{m:>12s}" for m in models))
    for k, g in lc.groupby("months", sort=False):
        cells = [rng(g.loc[g["model"] == m, "rmse"]) for m in models]
        print(
            f"{'all' if k == -1 else k:>7} {g['n_train'].iloc[0]:>8,} "
            + " ".join(f"{c:>12s}" for c in cells)
        )

    print("\n== Test RMSE by day type (seeded models: seed mean of RMSE) ==")
    dt = T.day_type(te.index)
    show = ["cl", "cl_plus", "cl_pp", "et", "hgb", "blend"]
    print(
        f"{'day type':12s} {'hours':>6s} "
        + " ".join(f"{m:>8s}" for m in show)
        + "  share of CL sq. error"
    )
    tot = np.sum((y - cl) ** 2)
    for name in ["working day", "weekend", "holiday", "all"]:
        k = np.ones(len(y), bool) if name == "all" else dt == name
        cells = [np.mean([T.rmse(y[k], p[k]) for p in P[m].values()]) for m in show]
        print(
            f"{name:12s} {k.sum():>6,} "
            + " ".join(f"{c:8.1f}" for c in cells)
            + f"  {np.sum((y[k] - cl[k]) ** 2) / tot:.2f}"
        )


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
