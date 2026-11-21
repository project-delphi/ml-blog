"""Print every New Delhi number Part 1 quotes, from the caches (run from the post dir).

    cd posts/two-cultures-breiman-2001
    OMP_NUM_THREADS=1 ../../.venv-two-cultures/bin/python src/check_delhi.py

A missing cache is computed first (`twocultures.delhi_eval`; about half an hour
from cold on ten cores). Ranges are over the five seeds; a single number means
the model is unseeded or the seeds agree to the printed precision.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if not Path("index.qmd").exists():
    sys.exit("run from posts/two-cultures-breiman-2001")
sys.path.insert(0, str(Path("src").resolve()))

from twocultures import cache, delhi, modern  # noqa: E402
from twocultures import delhi_eval as E  # noqa: E402

pd.set_option("display.width", 200, "display.max_columns", 40)


def rng(v, fmt="{:.1f}"):
    lo, hi = fmt.format(np.min(v)), fmt.format(np.max(v))
    return lo if lo == hi else f"{lo}-{hi}"


def ci(g):
    """A per-seed gap table as 'diff [lo, hi]', each part a seed range."""
    return f"{rng(g['diff'])} [{rng(g['lo'])}, {rng(g['hi'])}]"


def show(df):
    print(df.to_string())


y = E.truth()
z = E.zoo()
fa = cache.frame(delhi.CACHE_TAG, delhi.run)
ch = delhi.chronos()
d = delhi.daily()
test_days = E.pred(z, "persistence", 0, "test").index
yt = y.loc[test_days]
print(
    f"days {d['date'].min().date()}..{d['date'].max().date()}  with pm25 "
    f"{int(d['pm25'].notna().sum())}/{len(d)}  design rows {len(y)}  test days "
    f"{len(yt)}  Severe test days {int((yt >= E.ALERT).sum())}  top-10% cut "
    f"{np.quantile(yt, 0.9):.0f}"
)
diffs = {
    m: float(
        np.max(
            [
                np.abs(
                    E.pred(z, m, s, "test") - E.pred(fa, m, s, "test").loc[test_days]
                ).max()
                for s in E.SEEDS
            ]
        )
    )
    for m in [
        "persistence",
        "linear_quad",
        "lasso",
        "forest",
        "forest_tuned",
        "extra_trees",
        "boosting",
    ]
}
print("zoo vs delhi.run, max |test diff|:", {k: round(v, 6) for k, v in diffs.items()})

# ---------------------------------------------------------------- A. scores
extra = pd.concat([fa[fa["model"] == "quantile_boost"], ch], ignore_index=True)[
    ["model", "seed", "split", "day", "pred"]
]
T = E.score_table(pd.concat([z, extra], ignore_index=True), y)
cv = z.groupby("model")["cv_rmse"].first()
order = T.groupby("model", sort=False)["rmse"].mean().sort_values().index
g = T.groupby("model")


def col(c, fmt):
    return [rng(g.get_group(m)[c], fmt) for m in order]


print("\nA1. Test error, Nov 2023-Oct 2026 (CV = walk-forward mean fold RMSE, seed 0)")
show(
    pd.DataFrame(
        {
            "cv": [
                f"{cv[m]:.1f}" if m in cv and pd.notna(cv[m]) else "" for m in order
            ],
            "oof": col("oof_rmse", "{:.1f}"),
            "rmse": col("rmse", "{:.1f}"),
            "mae": col("mae", "{:.1f}"),
            "log_rmse": col("log_rmse", "{:.3f}"),
            "top10": col("top10_rmse", "{:.1f}"),
            "skill": col("skill", "{:.3f}"),
            "aqi_acc": col("aqi_acc", "{:.3f}"),
        },
        index=order,
    )
)
print(
    "\nA2. Severe (>= 250) decisions on test; tau and csi_tau tuned on each model's OOF"
)
show(
    pd.DataFrame(
        {
            "hits": col("hits", "{:.0f}"),
            "false_al": col("false_alarms", "{:.0f}"),
            "csi": col("csi", "{:.3f}"),
            "tau": col("tau", "{:.0f}"),
            "csi_tau": col("csi_tau", "{:.3f}"),
            "ap": col("ap", "{:.3f}"),
            f"hits@{E.MATCHED_FA}fa": col("hits_at_fa", "{:.0f}"),
        },
        index=order,
    )
)

# ---------------------------------------------------------------- settings
print("\nB. Walk-forward settings and stack weights (seed 0)")
for m in [*E.MODELS, "stack"]:
    p = z.loc[(z["model"] == m) & (z["seed"] == 0), "params"].iloc[0]
    if p:
        print(f"  {m:16s} {p}")
lin, nl = E.cv_picks(z)
print(
    f"  CV picks: linear {lin}, tree {nl}; lowest CV overall {cv.drop('persistence').idxmin()}"
)

print("\nC. Paired test RMSE gaps (a - b), 7-day block bootstrap 95%, seed ranges")
seen = set()
for a, b in [
    (nl, lin),
    ("forest", "linear_quad"),
    ("hgb_log", "spline_ridge_log"),
    ("forest", "spline_ridge_log"),
    ("blend", "spline_ridge_log"),
    ("stack", "spline_ridge_log"),
    ("blend", "forest"),
    ("spline_ridge_log", "linear_quad"),
    ("spline_ridge_log", "persistence"),
    ("forest", "persistence"),
]:
    if (a, b) in seen:
        continue  # the CV picks are one of the fixed pairs
    seen.add((a, b))
    print(f"  {a:16s} - {b:16s} {ci(E.gap(z, b, a, y))}")

# ---------------------------------------------------------------- learning curve
L = E.learning_curve()
L["err"] = (L["pred"] - y.loc[L["day"]].to_numpy()) ** 2
r = (
    L.groupby(["years", "n_train", "model", "seed"])["err"]
    .mean()
    .pow(0.5)
    .reset_index()
)
W = (
    r.groupby(["years", "n_train", "model"])["err"]
    .agg(rng)
    .unstack("model")[E.LC_MODELS]
)
print("\nD. Learning curve: test RMSE training on the last k Nov-Oct years")
show(W)

# ---------------------------------------------------------------- CV compare
C = E.cv_compare().set_index("model")
C["test"] = T[T["model"].isin(C.index)].groupby("model")["rmse"].mean()
print("\nE. Shuffled 5-fold vs walk-forward CV vs test (mean over seeds)")
C["test_shuffled_pick"] = [
    rng([a, b])
    for a, b in zip(C["test_shuffled_pick_min"], C["test_shuffled_pick_max"])
]
show(
    C[["shuffled_cv", "walk_forward_cv", "test", "test_shuffled_pick"]]
    .round(1)
    .sort_values("test")
)
for c in ["shuffled_cv", "walk_forward_cv"]:
    print(f"  Spearman {c} vs test: {spearmanr(C[c], C['test'])[0]:.2f}")
changed = C.index[C["shuffled_pick"] != C["walk_forward_pick"]]
print(f"  shuffled CV picks another setting for: {', '.join(changed)}")


# ---------------------------------------------------------------- rolling, designs
def gap_rows(frames, key):
    rows = []
    for k, res in frames:
        res = res.drop(columns=key)
        lin, nl = E.cv_picks(res)
        tt = E.pred(res, "persistence", 0, "test").index
        yy = y.loc[tt]
        te = {
            m: [modern.rmse(E.pred(res, m, s, "test"), yy) for s in E.SEEDS]
            for m in E.LINEAR + E.TREES
        }
        mean = {m: np.mean(v) for m, v in te.items()}
        cvs = res.groupby("model")["cv_rmse"].first().loc[E.LINEAR + E.TREES]
        rows.append(
            {
                key: k,
                "n_test": len(tt),
                "severe": int((yy >= E.ALERT).sum()),
                "persist": f"{modern.rmse(E.pred(res, 'persistence', 0, 'test'), yy):.1f}",
                "lin_pick": f"{lin} {rng(te[lin])}",
                "tree_pick": f"{nl} {rng(te[nl])}",
                "gap [95%]": ci(E.gap(res, lin, nl, y)),
                "test_best": f"{min(mean, key=mean.get)} {min(mean.values()):.1f}",
                "rho": f"{spearmanr(cvs, [mean[m] for m in cvs.index])[0]:.2f}",
            }
        )
    return pd.DataFrame(rows).set_index(key)


R = E.rolling()
print(
    "\nF. Rolling 3-year test windows (tree pick - linear pick; rho = Spearman CV vs test)"
)
show(gap_rows([(f"Nov{w}", R[R["window"] == w]) for w in E.ROLL_STARTS], "window"))
D = E.designs()
print("\nG. 1-day design variants, test Nov 2023-Oct 2026")
show(gap_rows([(k, D[D["design"] == k]) for k in E.DESIGNS], "design"))
