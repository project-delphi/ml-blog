"""How far apart the linear and the tree models really are on New Delhi PM2.5.

`delhi.run` scores the models `modern` brings from the 1976 ozone problem. This
module widens both sides on the same 30-column next-day design and asks whether
the gap survives: smoother linear models (natural-spline features with a ridge
penalty, and log-scale fits), a log-target booster, a two-model blend and a
non-negative stack, scored by error and by the Severe-day decisions the forecast
is for, over more training years, more test windows and richer designs.

Every setting, threshold and weight is chosen on walk-forward out-of-fold
predictions inside the pre-test years, never on test days. Seeded models run
five seeds; tables report the range. Results are cached under `cache/` by
`cache.frame` (tags `delhi_zoo_v2`, `delhi_designs_v2`, and `_v1` for the
learning curve, CV comparison and rolling windows); `src/check_delhi.py` prints every number.

- `zoo()`: dev (out-of-fold) and test predictions for every model, same long
  format as `modern.fit_all`; `extras()` is the subset `delhi.run` lacks.
- `scores()`: the metrics the post quotes, for one forecast.
- `learning_curve()`: test error against the number of training years.
- `cv_compare()`: shuffled 5-fold CV against walk-forward CV against test.
- `rolling()`: three-year test windows from five origins.
- `designs()`: the 1-day design with next-day weather, engineered features, BLH.
"""

import json
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from . import cache, delhi, modern, ozone

SEEDS = tuple(modern.SEEDS)
ALERT = delhi.ALERT  # India's "Severe" band, 24-hour PM2.5 of 250 µg/m³ or more
AQI_EDGES = (30, 60, 90, 120)  # India's National AQI PM2.5 bands; Severe is >= ALERT
TAU_GRID = np.arange(120, 320, 5)  # candidate alert thresholds, tuned on OOF days
MATCHED_FA = 20  # false alarms allowed in the matched-alarm comparison
TEST0 = delhi.day_of(delhi.TEST_START)
DIWALI = (
    "2015-11-11",
    "2016-10-30",
    "2017-10-19",
    "2018-11-07",
    "2019-10-27",
    "2020-11-14",
    "2021-11-04",
    "2022-10-24",
    "2023-11-12",
    "2024-10-31",
    "2025-10-20",
    "2026-11-08",
)


def _grid(**kw):
    return [dict(zip(kw, v)) for v in product(*kw.values())]


# ------------------------------------------------------------------ models
class LogTarget:
    """Fit on log PM2.5 (and log PM2.5 lags); back-transform with Duan's smearing.

    The smearing factor is the mean of exp(residual) on the training rows, so the
    back-transformed forecast estimates the mean, not the median. Log forecasts
    are clipped to the training range widened by a factor of two either side.
    """

    def __init__(self, est, log_x=True):
        self.est, self.log_x = est, log_x

    def _x(self, X):
        if not self.log_x:
            return X
        X = X.copy()
        for c in X.columns:
            if c.startswith("pm"):
                X[c] = np.log(np.clip(X[c], 1, None))
        return X

    def fit(self, X, y):
        ly = np.log(y)
        Xp = self._x(X)
        self.est.fit(Xp, ly)
        self.smear = float(np.mean(np.exp(ly - self.est.predict(Xp))))
        self.lo, self.hi = float(ly.min() - np.log(2)), float(ly.max() + np.log(2))
        return self

    def predict(self, X):
        p = np.clip(self.est.predict(self._x(X)), self.lo, self.hi)
        return np.exp(p) * self.smear


class Quad:
    """Breiman's 1970s route (`ozone.linear_route`): `modern.fit_all`'s linear_quad."""

    def fit(self, X, y):
        s1 = ozone.forward(X, y)
        E = ozone.expand(X, s1)
        s2 = ozone.forward(E, y, max_vars=20)
        self.s1, self.s2, self.m = s1, s2, LinearRegression().fit(E[s2], y)
        return self

    def predict(self, X):
        return self.m.predict(ozone.expand(X, self.s1)[self.s2])


def _spline(alpha):
    return make_pipeline(
        StandardScaler(), SplineTransformer(n_knots=5, degree=3), Ridge(alpha=alpha)
    )


def _hgb(p, s):
    return modern.boosting(**p, seed=s)


# The forests use one core each, since the folds and seeds run in parallel; a
# forest's predictions do not depend on n_jobs.
def _forest(p, s):
    return RandomForestRegressor(500, random_state=s, n_jobs=1, **p)


def _extra(p, s):
    return ExtraTreesRegressor(500, random_state=s, n_jobs=1, **p)


TREE_GRID = modern.TUNED["forest_tuned"][0]
HGB_GRID = modern.TUNED["boosting"][0]
RIDGE_GRID = _grid(alpha=list(np.logspace(-2, 3, 12)))

# name -> (grid, seeded, factory(params, seed), family). The names shared with
# `modern.fit_all` use its grids and give its predictions (`src/check_delhi.py`
# checks this on the test days).
MODELS = {
    "ols": (
        [{}],
        False,
        lambda p, s: make_pipeline(StandardScaler(), LinearRegression()),
        "linear",
    ),
    "lasso": (
        modern.TUNED["lasso"][0],
        False,
        lambda p, s: modern.lasso(**p),
        "linear",
    ),
    "ridge_log": (
        RIDGE_GRID,
        False,
        lambda p, s: LogTarget(make_pipeline(StandardScaler(), Ridge(**p))),
        "linear",
    ),
    "spline_ridge": (RIDGE_GRID, False, lambda p, s: _spline(**p), "linear"),
    "spline_ridge_log": (
        RIDGE_GRID,
        False,
        lambda p, s: LogTarget(_spline(**p)),
        "linear",
    ),
    "linear_quad": ([{}], False, lambda p, s: Quad(), "linear"),
    "forest": (
        [{"max_features": 1 / 3, "min_samples_leaf": 1}],
        True,
        _forest,
        "tree",
    ),
    "forest_tuned": (TREE_GRID, True, _forest, "tree"),
    "extra_trees": (TREE_GRID, True, _extra, "tree"),
    "boosting": (HGB_GRID, True, _hgb, "tree"),
    "hgb_log": (
        HGB_GRID,
        True,
        lambda p, s: LogTarget(_hgb(p, s), log_x=False),
        "tree",
    ),
}
LINEAR = [n for n, m in MODELS.items() if m[3] == "linear"]
TREES = [n for n, m in MODELS.items() if m[3] == "tree"]
BLEND = ("forest", "hgb_log")
STACK = [n for n in MODELS if n != "ols"]  # ols is lasso's limit; NNLS needs no twin
NEW = [
    "ols",
    "ridge_log",
    "spline_ridge",
    "spline_ridge_log",
    "hgb_log",
    "blend",
    "stack",
]


def _label(p: dict) -> str:
    return ", ".join(f"{k}={modern._fmt(v)}" for k, v in p.items())


def _fit_predict(name, p, seed, Xtr, ytr, Xte):
    return np.asarray(MODELS[name][2](p, seed).fit(Xtr, ytr).predict(Xte), float)


# ------------------------------------------------------------------ designs
def build(nextwx=False, eng=False, blh=False) -> pd.DataFrame:
    """The 1-day design with optional additions, indexed by the forecast day t.

    The base columns are `delhi.design()`'s, in its order. `nextwx` adds the eight
    ERA5 readings of the target day t + 1 (a perfect-prognosis upper bound on what a
    weather forecast could add). `eng` adds 3- and 7-day PM2.5 means (the 7-day
    over at least three valid days), day t's highest hour, the target day's season
    at two harmonics, an October-November stubble-burning flag and a Diwali window
    (target from the day before to three days after), and wind speed times the
    diurnal temperature range as a ventilation proxy. `blh` adds ERA5's
    boundary-layer height and wind times BLH; its January-June 2024 gap is filled by
    a log-linear fit on same-day weather and season, fitted on pre-test days only.
    """
    base = delhi.design()
    d = delhi.daily()
    assert (np.diff(d.index) == 1).all() and d.index[0] == 0
    pm = d["pm25"]
    wx = delhi.WEATHER
    tdate = d["date"] + pd.Timedelta(days=1)
    cols = {}
    if nextwx:
        for w in wx:
            cols[f"{w}_next"] = d[w].shift(-1)
    if eng:
        cols["pm_roll3"] = pm.rolling(3, min_periods=3).mean()
        cols["pm_roll7"] = pm.rolling(7, min_periods=3).mean()
        cols["pm_max_lag0"] = d["pm25_max"]
        g = tdate.dt.day_of_year
        for k in (1, 2):
            cols[f"tdoy_sin{k}"] = np.sin(2 * np.pi * k * g / 365.25)
            cols[f"tdoy_cos{k}"] = np.cos(2 * np.pi * k * g / 365.25)
        cols["stubble"] = tdate.dt.month.isin([10, 11]).astype(float)
        delta = np.array(
            [(tdate - x).dt.days.to_numpy() for x in pd.to_datetime(DIWALI)]
        )
        cols["diwali"] = ((delta >= -1) & (delta <= 3)).any(0).astype(float)
        cols["vent_dtr"] = d["wind_speed_mean"] * (d["temp_max"] - d["temp_min"])
        if nextwx:
            cols["vent_dtr_next"] = cols["wind_speed_mean_next"] * (
                cols["temp_max_next"] - cols["temp_min_next"]
            )
    if blh:
        b = d["blh_mean"].copy()
        doy = d["date"].dt.day_of_year
        Z = pd.concat(
            [
                d[wx + ["radiation_sum", "dewpoint_mean"]],
                np.sin(2 * np.pi * doy / 365.25).rename("s"),
                np.cos(2 * np.pi * doy / 365.25).rename("c"),
            ],
            axis=1,
        )
        ok = b.notna() & Z.notna().all(1) & (d.index < TEST0)
        fit = LinearRegression().fit(Z[ok], np.log(b[ok]))
        miss = b.isna() & Z.notna().all(1)
        b[miss] = np.exp(fit.predict(Z[miss]))
        cols["blh_lag0"] = b
        cols["vent_blh"] = b * d["wind_speed_mean"]
        if nextwx:
            cols["blh_next"] = b.shift(-1)
            cols["vent_blh_next"] = cols["blh_next"] * d["wind_speed_mean"].shift(-1)
    extra = pd.DataFrame(cols, index=d.index)
    out = base.drop(columns="target").join(extra, how="left")
    out["target"] = base["target"]
    return out.dropna()


DESIGNS = {
    "base": {},
    "nextwx": {"nextwx": True},
    "eng": {"eng": True},
    "eng+nextwx+blh": {"eng": True, "nextwx": True, "blh": True},
}


# ------------------------------------------------------------------ engine
def splits(df, edges=delhi.FOLD_EDGES, test_end=None, train_from=None):
    """Walk-forward folds from validation-block edges, refit and test positions.

    The folds are `modern.folds_from_edges` (three-day embargo), optionally cut to
    training days from `train_from`. The refit uses every day from `train_from` to
    the last validation edge; the test runs from that edge to `test_end`.
    """
    t = df.index.to_numpy()
    lo = -(10**9) if train_from is None else train_from
    hi = 10**9 if test_end is None else test_end
    folds = [(tr[t[tr] >= lo], va) for tr, va in modern.folds_from_edges(t, edges)]
    refit = np.flatnonzero((t >= lo) & (t < edges[-1]))
    test = np.flatnonzero((t >= edges[-1]) & (t < hi))
    return folds, refit, test


def evaluate(df, folds, refit, test, names, oof_seeds=SEEDS, fixed=None):
    """Choose each model's setting by walk-forward CV, then predict.

    The mean RMSE over the folds at seed 0 picks the setting (`fixed` maps a name
    to a setting to skip the search). Dev rows are out-of-fold predictions for
    the seeds in `oof_seeds` (seed 0's are the search's own); test rows come from
    a refit on `refit` for every seed. Returns rows in `modern.fit_all`'s format
    plus `settings` (JSON) and `cv_rmse` (the chosen setting's CV score).
    """
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    days = df.index.to_numpy()
    dev = np.concatenate([va for _, va in folds])
    fixed = fixed or {}
    grids = {n: [fixed[n]] if n in fixed else MODELS[n][0] for n in names}
    jobs = [
        (n, i, k)
        for n in names
        for i in range(len(grids[n]))
        for k in range(len(folds))
    ]
    run = Parallel(n_jobs=-1)
    preds = run(
        delayed(_fit_predict)(
            n, grids[n][i], 0, X.iloc[folds[k][0]], y[folds[k][0]], X.iloc[folds[k][1]]
        )
        for n, i, k in jobs
    )
    fold_rmse, fold_pred = {}, {}
    for (n, i, k), p in zip(jobs, preds):
        fold_rmse.setdefault((n, i), []).append(modern.rmse(p, y[folds[k][1]]))
        fold_pred.setdefault((n, i), []).append(p)
    best = {
        n: int(np.argmin([np.mean(fold_rmse[(n, i)]) for i in range(len(grids[n]))]))
        for n in names
    }
    jobs2 = []
    for n in names:
        seeds = SEEDS if MODELS[n][1] else (0,)
        jobs2 += [(n, s, "test", None) for s in seeds]
        jobs2 += [
            (n, s, "dev", k)
            for s in seeds
            if s in oof_seeds and s
            for k in range(len(folds))
        ]
    preds2 = run(
        delayed(_fit_predict)(
            n,
            grids[n][best[n]],
            s,
            X.iloc[refit if k is None else folds[k][0]],
            y[refit if k is None else folds[k][0]],
            X.iloc[test if k is None else folds[k][1]],
        )
        for n, s, _, k in jobs2
    )
    got = {}
    for (n, s, split, k), p in zip(jobs2, preds2):
        got.setdefault((n, s, split), []).append(p)
    out = []
    for n in names:
        p = grids[n][best[n]]
        meta = {
            "params": _label(p),
            "settings": json.dumps(p),
            "cv_rmse": float(np.mean(fold_rmse[(n, best[n])])),
        }
        for s in SEEDS if MODELS[n][1] else (0,):
            if s in oof_seeds:
                dv = fold_pred[(n, best[n])] if s == 0 else got[(n, s, "dev")]
                out.append(_rows(n, s, "dev", days[dev], np.concatenate(dv), meta))
            out.append(_rows(n, s, "test", days[test], got[(n, s, "test")][0], meta))
    out.append(
        _rows("persistence", 0, "dev", days[dev], X["pm25_lag0"].to_numpy()[dev], {})
    )
    out.append(
        _rows("persistence", 0, "test", days[test], X["pm25_lag0"].to_numpy()[test], {})
    )
    return pd.concat(out, ignore_index=True)


def _rows(model, seed, split, days, pred, meta):
    return pd.DataFrame(
        {
            "model": model,
            "seed": seed,
            "split": split,
            "day": days,
            "pred": pred,
            **meta,
        }
    )


def pred(res, model, seed, split):
    """One forecast from a long frame, indexed by day; an unseeded model's seed is 0."""
    g = res[(res["model"] == model) & (res["split"] == split)]
    if seed not in set(g["seed"]):
        seed = 0
    return g[g["seed"] == seed].set_index("day")["pred"]


def _combine(res, y, folds_dev_days):
    """The equal-weight blend of `BLEND` and the NNLS stack of `STACK`, per seed.

    Stack weights for test days are fitted on every out-of-fold day. Its own dev
    rows are walk-forward like every other model's: a validation block is
    forecast with weights fitted on the earlier blocks only, and the first block,
    which has none before it, with equal weights. Weights are non-negative with
    no intercept.
    """
    out = []
    for s in sorted(set(res.loc[res["split"] == "dev", "seed"])):
        D = {n: pred(res, n, s, "dev") for n in STACK}
        T = {n: pred(res, n, s, "test") for n in STACK}
        dd, td = D[STACK[0]].index, T[STACK[0]].index
        Dm = np.column_stack([D[n].loc[dd] for n in STACK])
        Tm = np.column_stack([T[n].loc[td] for n in STACK])
        yd = y.loc[dd].to_numpy()
        b = [STACK.index(n) for n in BLEND]
        bl_d, bl_t = Dm[:, b].mean(1), Tm[:, b].mean(1)
        out.append(_rows("blend", s, "dev", dd, bl_d, {"params": " + ".join(BLEND)}))
        out.append(_rows("blend", s, "test", td, bl_t, {"params": " + ".join(BLEND)}))
        w, _ = nnls(Dm, yd)
        sd = np.empty(len(dd))
        for k, blk in enumerate(folds_dev_days):
            m = np.isin(dd, blk)
            past = np.isin(dd, np.concatenate(folds_dev_days[:k])) if k else None
            wk = (
                nnls(Dm[past], yd[past])[0]
                if k
                else np.full(len(STACK), 1 / len(STACK))
            )
            sd[m] = Dm[m] @ wk
        lab = ", ".join(f"{n}={v:.2f}" for n, v in zip(STACK, w) if v > 0.005)
        out.append(_rows("stack", s, "dev", dd, sd, {"params": lab}))
        out.append(_rows("stack", s, "test", td, Tm @ w, {"params": lab}))
    return pd.concat(out, ignore_index=True)


def _run_zoo(design="base"):
    df = build(**DESIGNS[design])
    folds, refit, test = splits(df)
    res = evaluate(df, folds, refit, test, list(MODELS))
    blocks = [df.index.to_numpy()[va] for _, va in folds]
    res = pd.concat([res, _combine(res, df["target"], blocks)], ignore_index=True)
    for c in ("params", "settings"):
        res[c] = res[c].fillna("")
    return res


def zoo() -> pd.DataFrame:
    """Every model on the 30-column design: dev (OOF) and test rows, five seeds."""
    return cache.frame("delhi_zoo_v2", _run_zoo)


def extras() -> pd.DataFrame:
    """The models `delhi.run` lacks, in its long format."""
    z = zoo()
    return z[z["model"].isin(NEW)].reset_index(drop=True)


def settings(res, model) -> dict:
    return json.loads(res.loc[res["model"] == model, "settings"].iloc[0])


def truth() -> pd.Series:
    """Observed next-day PM2.5 by forecast day t."""
    return delhi.design()["target"]


# ------------------------------------------------------------------ metrics
def aqi_band(x):
    """India's National AQI band for a 24-hour PM2.5 mean: 0 Good ... 5 Severe."""
    x = np.asarray(x, float)
    return np.where(x >= ALERT, 5, np.digitize(x, AQI_EDGES, right=True))


def _alerts(p, y, tau):
    sev, a = y >= ALERT, p >= tau
    h, m, f = int((sev & a).sum()), int((sev & ~a).sum()), int((~sev & a).sum())
    return h, f, h / max(h + m + f, 1)


def tune_tau(p_dev, y_dev) -> float:
    """The alert threshold with the best CSI on out-of-fold days (lowest on ties)."""
    csi = [_alerts(np.asarray(p_dev), np.asarray(y_dev), t)[2] for t in TAU_GRID]
    return float(TAU_GRID[int(np.argmax(csi))])


def scores(p, y, persist, top_cut, tau=None) -> dict:
    """The post's metrics for one test forecast `p` of observed `y`.

    `top_cut` is the 90th percentile of observed test days (the same for every
    model). `tau` is the model's own alert threshold, tuned on its OOF forecasts
    (`tune_tau`); without it the tuned-CSI entries are NaN.
    """
    p, y, persist = (np.asarray(a, float) for a in (p, y, persist))
    sev = y >= ALERT
    hits, fa, csi = _alerts(p, y, ALERT)
    # Matched false alarms: alert on the MATCHED_FA highest forecasts among the
    # non-Severe days and above, so every model is held to the same false alarms.
    thr = np.sort(p[~sev])[::-1][MATCHED_FA - 1]
    top = y >= top_cut
    out = {
        "rmse": modern.rmse(p, y),
        "mae": float(np.abs(p - y).mean()),
        "log_rmse": float(
            np.sqrt(np.mean((np.log(np.clip(p, 1, None)) - np.log(y)) ** 2))
        ),
        "top10_rmse": modern.rmse(p[top], y[top]),
        "hits": hits,
        "false_alarms": fa,
        "csi": csi,
        "ap": float(average_precision_score(sev, p)),
        "hits_at_fa": int((p[sev] >= thr).sum()),
        "aqi_acc": float((aqi_band(p) == aqi_band(y)).mean()),
        "skill": float(1 - np.mean((p - y) ** 2) / np.mean((persist - y) ** 2)),
        "tau": np.nan,
        "csi_tau": np.nan,
    }
    if tau is not None:
        out["tau"] = tau
        out["csi_tau"] = _alerts(p, y, tau)[2]
    return out


def score_table(res, y=None) -> pd.DataFrame:
    """`scores` for every (model, seed) of a long frame, plus pooled OOF CV RMSE."""
    y = truth() if y is None else y
    per = pred(res, "persistence", 0, "test")
    yt = y.loc[per.index].to_numpy()
    top_cut = float(np.quantile(yt, 0.9))
    rows = []
    # A model with out-of-fold rows for seed 0 only (`delhi.run`'s quantile boosting)
    # takes seed 0's threshold and CV score for every seed.
    dev0 = {
        m: g.set_index("day")["pred"]
        for m, g in res[(res["split"] == "dev") & (res["seed"] == 0)].groupby("model")
    }
    for (m, s), g in res.groupby(["model", "seed"], sort=False):
        te = g[g["split"] == "test"].set_index("day")["pred"].loc[per.index]
        dv = g[g["split"] == "dev"].set_index("day")["pred"]
        if not len(dv) and m in dev0:
            dv = dev0[m]
        tau = tune_tau(dv, y.loc[dv.index]) if len(dv) else None
        r = scores(te, yt, per.to_numpy(), top_cut, tau)
        r["oof_rmse"] = modern.rmse(dv, y.loc[dv.index]) if len(dv) else np.nan
        rows.append({"model": m, "seed": s, **r})
    return pd.DataFrame(rows)


def gap(res, lin, nl, y=None):
    """Test RMSE of `nl` minus `lin` per seed, with its 7-day block-bootstrap interval."""
    y = truth() if y is None else y
    rows = []
    for s in SEEDS:
        a, b = pred(res, nl, s, "test"), pred(res, lin, s, "test")
        yy = y.loc[a.index].to_numpy()
        lo, hi = modern.block_bootstrap(a.to_numpy(), b.loc[a.index].to_numpy(), yy)
        d = modern.rmse(a, yy) - modern.rmse(b.loc[a.index], yy)
        rows.append({"seed": s, "diff": d, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def cv_picks(res):
    """The linear and the tree model with the lowest walk-forward CV RMSE."""
    cv = res.groupby("model")["cv_rmse"].first()
    return cv[LINEAR].idxmin(), cv[TREES].idxmin()


# ------------------------------------------------------------------ studies
LC_MODELS = ["ols", "linear_quad", "spline_ridge_log", "forest", "hgb_log"]
LC_YEARS = range(1, 9)


def _run_learning_curve():
    df, z = delhi.design(), zoo()
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    t = df.index.to_numpy()
    test = np.flatnonzero(t >= TEST0)
    jobs = []
    for k in LC_YEARS:
        start = delhi.day_of(pd.Timestamp(f"{delhi.TEST_START.year - k}-11-01"))
        tr = np.flatnonzero((t >= start) & (t < TEST0))
        for n in LC_MODELS:
            for s in SEEDS if MODELS[n][1] else (0,):
                jobs.append((k, len(tr), n, s, tr))
    preds = Parallel(n_jobs=-1)(
        delayed(_fit_predict)(n, settings(z, n), s, X.iloc[tr], y[tr], X.iloc[test])
        for _, _, n, s, tr in jobs
    )
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "years": k,
                    "n_train": nt,
                    "model": n,
                    "seed": s,
                    "day": t[test],
                    "pred": p,
                }
            )
            for (k, nt, n, s, _), p in zip(jobs, preds)
        ],
        ignore_index=True,
    )


def learning_curve() -> pd.DataFrame:
    """Test predictions from the last k November-October years only (k = 1-8).

    Settings stay at the full-data walk-forward choice; the test years are fixed.
    """
    return cache.frame("delhi_learning_curve_v1", _run_learning_curve)


def _run_cv_compare():
    df, z = delhi.design(), zoo()
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    _, refit, test = splits(df)
    rf = [
        (refit[a], refit[b])
        for a, b in KFold(5, shuffle=True, random_state=0).split(refit)
    ]
    names = list(MODELS)
    # Shuffled CV at the walk-forward settings, and the setting shuffled CV would pick.
    jobs = [
        (n, i, k) for n in names for i in range(len(MODELS[n][0])) for k in range(5)
    ]
    preds = Parallel(n_jobs=-1)(
        delayed(_fit_predict)(
            n, MODELS[n][0][i], 0, X.iloc[rf[k][0]], y[rf[k][0]], X.iloc[rf[k][1]]
        )
        for n, i, k in jobs
    )
    sc = {}
    for (n, i, k), p in zip(jobs, preds):
        sc.setdefault((n, i), []).append(modern.rmse(p, y[rf[k][1]]))
    rows, refits = [], []
    for n in names:
        g = MODELS[n][0]
        wf = g.index(settings(z, n)) if len(g) > 1 else 0
        rp = int(np.argmin([np.mean(sc[(n, i)]) for i in range(len(g))]))
        rows.append(
            {
                "model": n,
                "shuffled_cv": float(np.mean(sc[(n, wf)])),
                "walk_forward_cv": float(z.loc[z["model"] == n, "cv_rmse"].iloc[0]),
                "shuffled_pick": _label(g[rp]),
                "walk_forward_pick": _label(g[wf]),
            }
        )
        refits += [(n, s, g[rp]) for s in (SEEDS if MODELS[n][1] else (0,))]
    tp = Parallel(n_jobs=-1)(
        delayed(_fit_predict)(n, p, s, X.iloc[refit], y[refit], X.iloc[test])
        for n, s, p in refits
    )
    te = {}
    for (n, s, _), p in zip(refits, tp):
        te.setdefault(n, []).append(modern.rmse(p, y[test]))
    out = pd.DataFrame(rows)
    out["test_shuffled_pick_min"] = [min(te[n]) for n in out["model"]]
    out["test_shuffled_pick_max"] = [max(te[n]) for n in out["model"]]
    return out


def cv_compare() -> pd.DataFrame:
    """Shuffled 5-fold CV on the pre-test days against walk-forward CV, per model.

    `shuffled_cv` is at the walk-forward setting; `shuffled_pick` is the setting
    shuffled CV would choose, refit on every pre-test day and scored on test.
    """
    return cache.frame("delhi_cv_compare_v1", _run_cv_compare)


ROLL_STARTS = (2019, 2020, 2021, 2022, 2023)
ROLL_YEARS = 3


def _window(y0):
    """Validation blocks for a test window from November y0: the up to four
    November-October years before it, keeping at least two training years.
    """
    first = max(delhi.START.year + 2, y0 - 4)
    edges = tuple(
        delhi.day_of(pd.Timestamp(f"{y}-11-01")) for y in range(first, y0 + 1)
    )
    end = delhi.day_of(pd.Timestamp(f"{y0 + ROLL_YEARS}-11-01"))
    return edges, end


def _run_rolling():
    df = delhi.design()
    out = []
    for y0 in ROLL_STARTS:
        edges, end = _window(y0)
        folds, refit, test = splits(df, edges, test_end=end)
        r = evaluate(df, folds, refit, test, list(MODELS), oof_seeds=())
        out.append(r.assign(window=y0))
    return pd.concat(out, ignore_index=True).fillna({"params": "", "settings": ""})


def rolling() -> pd.DataFrame:
    """Test predictions for three-year windows from November 2019 ... 2023.

    Each window's settings come from walk-forward CV on the years before it.
    """
    return cache.frame("delhi_rolling_v1", _run_rolling)


def _run_designs():
    # Test rows only: they carry each model's CV score, which is all the post reads.
    out = []
    for name in DESIGNS:
        if name != "base":
            df = build(**DESIGNS[name])
            r = evaluate(df, *splits(df), list(MODELS), oof_seeds=())
            out.append(r[r["split"] == "test"].assign(design=name))
    return pd.concat(out, ignore_index=True).fillna({"params": "", "settings": ""})


def designs() -> pd.DataFrame:
    """Test predictions on the four 1-day designs (`DESIGNS`), settings by walk-forward CV.

    The base design is `zoo()`'s test rows; the other three are cached here.
    """
    z = zoo()
    base = z[(z["split"] == "test") & z["model"].isin(list(MODELS) + ["persistence"])]
    rest = cache.frame("delhi_designs_v2", _run_designs)
    return pd.concat([base.assign(design="base"), rest], ignore_index=True)
