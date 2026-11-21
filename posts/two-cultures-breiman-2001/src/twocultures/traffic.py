"""Hourly I-94 traffic volume, forecast a day ahead: where the trees' edge is real.

Data: UCI 492, westbound I-94 at MN DoT ATR station 301, October 2012 to September
2018 (`data/raw/metro_traffic.csv.gz`, every source row, written by
`src/fetch_traffic.py`). All cleaning happens in `clean()`, so each fix is visible:

- **"None" is a holiday name that pandas reads as missing.** The source writes the
  literal string `None` on every non-holiday row. `pd.read_csv` turns that into NaN
  by default, and then `holiday != "None"` is true on every row, so every day becomes
  a holiday and every working day disappears. `load_raw` reads with
  `keep_default_na=False`.
- **A holiday is labelled on its 00:00 row only.** The label is spread to the whole
  calendar day, and unioned with the US federal calendar, because some years are
  unlabelled (Martin Luther King Jr Day 2013, Independence Day 2014 and others).
  The calendar is known years ahead, so this uses nothing a forecaster lacks.
- **7,629 duplicate timestamps**, one row per concurrent weather label. Volume
  agrees within every duplicate set; the first row in file order is kept.
- **Sensor faults**: ten hours read 0 K (31 Jan to 2 Feb 2014), interpolated from
  neighbours; one hour reads 9,831.3 mm of rain (11 Jul 2016 17:00), set to 0.
- **The counter stops** from 8 Aug 2014 01:00 to 11 Jun 2015 20:00 (307 days), with
  shorter stops of 2 to 10 days elsewhere. Hours are not invented: a missing hour
  is simply absent, and a lag that falls in a gap is NaN (the trees take NaN
  natively; the linear models fill it, `LagFiller`).

Design: the forecast is issued at midnight starting day D, for each hour of D. It
may use calendar facts for D, the weather actually observed at the target hour
(standing in for a weather forecast: "perfect prognosis", so every model gets the
same flattering weather), and target values up to 23:00 on D-1. The lags are the
same hour 1, 2 and 7 days back, the 7-day same-hour mean, and D-1's daily total.
No same-day volume enters any feature.

Splits by time: train from 10 Oct 2012 (the source starts at 09:00 on 2 Oct, so earlier hours lack a 7-day lag), four
quarterly walk-forward validation folds in 2017 (each trained on everything before
it), test 1 Jan to 30 Sep 2018, refit once on all pre-test hours. Every setting is
chosen on the pooled validation RMSE at seed 0 and on nothing else.

- `run()`: the ladder, dev (out-of-fold, seed 0) and test rows, long format.
- `learning_curve()`: test scores training on the last k months, settings fixed.
- `probe()`: CL++, a post-hoc linear model (see its docstring).
- `scores()`, `gap()`, `breakdown()`: the numbers `src/check_traffic.py` prints.

Results are cached by `cache.frame` (tags `traffic_v1`, `traffic_lc_v1`,
`traffic_probe_v1`). Run with OMP_NUM_THREADS=1: models are single-threaded and
joblib spreads them over cores, which also keeps every number reproducible.
"""

import json
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from . import cache, modern

RAW = Path("data/raw/metro_traffic.csv.gz")
CACHE_TAG, LC_TAG, PROBE_TAG = "traffic_v2", "traffic_lc_v2", "traffic_probe_v2"
SEEDS = tuple(modern.SEEDS)
START = "2012-10-10"
FOLDS = (
    ("2017-01-01", "2017-04-01"),
    ("2017-04-01", "2017-07-01"),
    ("2017-07-01", "2017-10-01"),
    ("2017-10-01", "2018-01-01"),
)
TEST = ("2018-01-01", "2018-10-01")
LC_MONTHS = (6, 12, 24, 36, None)  # None: every pre-test hour (63 months)
BLOCK = 168  # bootstrap block: 7 days of hourly rows
CONG_Q = 0.8  # congestion hour: volume at or above the pre-test 80th percentile
TOP_Q = 0.9  # busiest 10% of test hours, by actual volume

LAGS = ["lag24", "lag48", "lag168", "sh7", "ytot"]
CAT = ["hr", "weekday", "mnth", "wx", "holiday", "workingday"]
NUM = ["tempc", "rain", "snow", "clouds"]
SPLINE = ["tempc", "clouds"]
WX = [
    "Clear",
    "Clouds",
    "Rain",
    "Drizzle",
    "Mist",
    "Haze",
    "Fog",
    "Snow",
    "Thunderstorm",
    "Smoke",
    "Squall",
]
WX_GROUP = {0: 0, 1: 0, 2: 1, 3: 1, 8: 1, 10: 1, 4: 2, 5: 2, 6: 2, 9: 2, 7: 3}
TRANSFORMS = {
    "log1p": (np.log1p, np.expm1),
    "sqrt": (np.sqrt, np.square),
    "none": (lambda x: x, lambda x: x),
}


# ------------------------------------------------------------------ data
def load_raw() -> pd.DataFrame:
    """Every source row; `keep_default_na=False` keeps the literal "None" label."""
    return pd.read_csv(RAW, keep_default_na=False)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per observed hour, indexed by local time, with the fixes above."""
    t = raw.copy()
    t["date_time"] = pd.to_datetime(t["date_time"])
    assert (t.groupby("date_time")["traffic_volume"].nunique() == 1).all()
    t = t.sort_values("date_time", kind="stable").drop_duplicates("date_time")
    t = t.set_index("date_time")
    labelled = t.index[t["holiday"] != "None"].normalize()
    federal = USFederalHolidayCalendar().holidays(t.index.min(), t.index.max())
    days = set(labelled) | set(federal)
    t["holiday"] = t.index.normalize().isin(days).astype(int)
    t.loc[t["temp"] < 200, "temp"] = np.nan  # 0 K readings
    t["temp"] = t["temp"].interpolate(limit=6)
    t.loc[t["rain_1h"] > 1000, "rain_1h"] = 0.0  # the 9831.3 mm hour
    return t


def _lags(grid: pd.Series) -> pd.DataFrame:
    """Lags known at midnight, on a complete hourly index (NaN = unobserved)."""
    out = pd.DataFrame(index=grid.index)
    out["lag24"] = grid.shift(24)
    out["lag48"] = grid.shift(48)
    out["lag168"] = grid.shift(168)
    out["sh7"] = pd.concat([grid.shift(24 * k) for k in range(1, 8)], axis=1).mean(1)
    # A day with 18-23 counted hours is scaled up to 24, so a gap does not read as
    # light traffic; fewer than 18 hours is missing.
    g = grid.groupby(grid.index.normalize())
    day = g.sum(min_count=18) * 24 / g.count()
    out["ytot"] = day.shift(1, freq="D").reindex(grid.index.normalize()).to_numpy()
    return out


def design(t: pd.DataFrame) -> pd.DataFrame:
    idx = pd.date_range(t.index.min().normalize(), t.index.max(), freq="h")
    grid = t["traffic_volume"].astype(float).reindex(idx)
    df = pd.DataFrame(index=t.index)
    df["y"] = t["traffic_volume"].astype(float)
    df["hr"] = t.index.hour
    df["weekday"] = t.index.dayofweek
    df["mnth"] = t.index.month
    df["holiday"] = t["holiday"]
    df["workingday"] = ((df["weekday"] < 5) & (df["holiday"] == 0)).astype(int)
    df["wx"] = t["weather_main"].map({c: i for i, c in enumerate(WX)}).fillna(0)
    df["wx"] = df["wx"].astype(int)
    df["tempc"] = t["temp"] - 273.15
    df["rain"] = np.log1p(t["rain_1h"])
    df["snow"] = t["snow_1h"]
    df["clouds"] = t["clouds_all"].astype(float)
    df = df.join(_lags(grid))
    return df.loc[START:]


def load() -> pd.DataFrame:
    return design(clean(load_raw()))


def _before(df, when):
    return df.loc[: pd.Timestamp(when) - pd.Timedelta(hours=1)]


def _between(df, a, b):
    return df.loc[a : pd.Timestamp(b) - pd.Timedelta(hours=1)]


def splits(df):
    """(walk-forward folds as (train, val) frames, pre-test frame, test frame)."""
    folds = [(_before(df, a), _between(df, a, b)) for a, b in FOLDS]
    return folds, _before(df, TEST[0]), _between(df, *TEST)


def cong_threshold(df) -> float:
    """Congestion cut, fixed from pre-test volumes before any model is scored."""
    return float(np.quantile(_before(df, TEST[0])["y"], CONG_Q))


# ------------------------------------------------------------------ models
class LagFiller:
    """Linear models need complete lags: fill a NaN lag with the training window's
    hour-of-week median volume, and add an indicator for a missing day lag.
    """

    def fit(self, tr):
        how = tr.index.dayofweek * 24 + tr.index.hour
        self.med = tr.groupby(how)["y"].median()
        self.ytot = tr["ytot"].median()
        return self

    def transform(self, df):
        out = df[LAGS].copy()
        how = pd.Series(df.index.dayofweek * 24 + df.index.hour, index=df.index)
        fill = how.map(self.med).fillna(self.med.median())
        miss = out[["lag24", "lag168"]].isna().any(axis=1).astype(float)
        for c in ["lag24", "lag48", "lag168", "sh7"]:
            out[c] = out[c].fillna(fill)
        out["ytot"] = out["ytot"].fillna(self.ytot)
        out["lagmiss"] = miss
        return out


class Naive:
    """Same hour last week (or yesterday), falling back to the other lag, then to
    the training hour-of-week median, when the lag hour is unobserved.
    """

    def __init__(self, col="lag168", **_):
        self.col = col

    def fit(self, tr):
        self.lf = LagFiller().fit(tr)
        return self

    def predict(self, df):
        other = "lag24" if self.col == "lag168" else "lag168"
        p = df[self.col].fillna(df[other])
        return self.lf.transform(df.assign(**{self.col: p}))[self.col].to_numpy()


def _onehot(X):
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X)


class OLSRaw:
    """Numbers in as given: hour, weekday and month as integers, the weather label
    one-hot (it is a string), raw lags.
    """

    def __init__(self, **_):
        pass

    def _X(self, df):
        num = df[[c for c in CAT if c != "wx"] + NUM].to_numpy(float)
        return np.hstack(
            [num, self.lf.transform(df).to_numpy(float), self.oh.transform(df[["wx"]])]
        )

    def fit(self, tr):
        self.lf, self.oh = LagFiller().fit(tr), _onehot(tr[["wx"]])
        self.m = LinearRegression().fit(self._X(tr), tr["y"].to_numpy())
        return self

    def predict(self, df):
        return np.clip(self.m.predict(self._X(df)), 0, None)


class CarefulLinear:
    """A statistician's data model, ridge-penalised, target log1p/sqrt/none.

    CL: one-hot hour x workingday, weekday, month, weather label, holiday; cubic
    B-splines (6 knots) on temperature and cloud cover; rain and snow linear; the
    lags on the target's scale, standardised.
    CL+ (`plus=True`, specified before any result was seen) adds hour x weather
    group, hour x season, the temperature spline x workingday, and lag24 and sh7
    slopes that vary by hour x workingday.
    """

    def __init__(self, alpha=1.0, tf="log1p", plus=False):
        self.alpha, self.tf, self.plus = alpha, tf, plus

    def _cats(self, df):
        hr = df["hr"].to_numpy()
        c = pd.DataFrame(
            {
                "hr_wd": hr * 2 + df["workingday"].to_numpy(),
                "weekday": df["weekday"].to_numpy(),
                "mnth": df["mnth"].to_numpy(),
                "wx": df["wx"].to_numpy(),
                "holiday": df["holiday"].to_numpy(),
            }
        )
        if self.plus:
            c["hr_wx"] = hr * 10 + df["wx"].map(WX_GROUP).to_numpy()
            c["hr_season"] = hr * 10 + (df["mnth"].to_numpy() % 12) // 3
        return c

    def _X(self, df, fit=False):
        g = TRANSFORMS[self.tf][0]
        cats = self._cats(df)
        lags = self.lf.transform(df)
        lin = [c for c in NUM if c not in SPLINE]
        Z = np.hstack(
            [
                g(lags[LAGS].clip(lower=0).to_numpy(float)),
                lags[["lagmiss"]].to_numpy(),
                df[lin].to_numpy(float),
            ]
        )
        hw = cats[["hr_wd"]].to_numpy()
        if fit:
            self.oh, self.sp = _onehot(cats), SplineTransformer(n_knots=6, degree=3)
            self.sp.fit(df[SPLINE])
            self.sc = StandardScaler().fit(Z)
            self.oh_hw = _onehot(hw)
        S, Zs = self.sp.transform(df[SPLINE]), self.sc.transform(Z)
        parts = [self.oh.transform(cats), S, Zs]
        if self.plus:
            wd = df["workingday"].to_numpy(float)[:, None]
            parts.append(S[:, : self.sp.n_features_out_ // len(SPLINE)] * wd)
            H = self.oh_hw.transform(hw)
            parts += [H * Zs[:, [0]], H * Zs[:, [3]]]  # lag24, sh7 slopes
        return np.hstack(parts)

    def fit(self, tr):
        self.lf = LagFiller().fit(tr)
        g = TRANSFORMS[self.tf][0]
        y = g(tr["y"].clip(lower=0).to_numpy())
        self.m = Ridge(alpha=self.alpha).fit(self._X(tr, fit=True), y)
        return self

    def predict(self, df):
        return np.clip(TRANSFORMS[self.tf][1](self.m.predict(self._X(df))), 0, None)


class CLpp(CarefulLinear):
    """CL++: POST-HOC. Designed after reading the test-set error breakdown
    (`breakdown()`: the linear models' worst hours were holidays and the days after
    weekends), so its test score is not a clean held-out comparison; alpha and the
    target transform are still chosen on the validation folds only.

    CL+ plus: hour x day-type one-hot (Mon..Sun, holiday: 8 types); lag24, lag168
    and sh7 slopes by day-type x hour-block (00-05, 06-09, 10-15, 16-23); and
    yesterday-was-non-working x hour, since lag24 then comes from another kind of day.
    """

    def __init__(self, alpha=1.0, tf="log1p"):
        super().__init__(alpha=alpha, tf=tf, plus=True)

    def _X(self, df, fit=False):
        base = super()._X(df, fit=fit)
        hr = df["hr"].to_numpy()
        dt = np.where(df["holiday"].to_numpy() == 1, 7, df["weekday"].to_numpy())
        prev = df.index.normalize() - pd.Timedelta(days=1)
        nonwork = _nonworking_days(prev.min(), df.index.max())
        prev_nw = np.asarray(prev.isin(nonwork), int)
        cats = pd.DataFrame({"hr_dt": hr * 10 + dt, "hr_prev": hr * 10 + prev_nw})
        sl = (dt * 10 + np.digitize(hr, [6, 10, 16]))[:, None]
        g = TRANSFORMS[self.tf][0]
        L = g(self.lf.transform(df)[["lag24", "lag168", "sh7"]].clip(lower=0))
        L = L.to_numpy(float)
        if fit:
            self.oh2, self.oh3 = _onehot(cats), _onehot(sl)
            self.mu, self.sd = L.mean(0), L.std(0) + 1e-9
        S, L = self.oh3.transform(sl), (L - self.mu) / self.sd
        return np.hstack(
            [base, self.oh2.transform(cats)] + [S * L[:, [j]] for j in range(3)]
        )


@lru_cache(maxsize=1)
def _holidays() -> frozenset:
    t = clean(load_raw())
    return frozenset(t.index[t["holiday"] == 1].normalize())


def _nonworking_days(a, b):
    """Weekends and holidays (labelled or federal) between two dates: calendar facts,
    so a day missing from the counter still has a day type.
    """
    days = pd.date_range(a.normalize(), b.normalize(), freq="D")
    hol = set(_holidays()) | set(USFederalHolidayCalendar().holidays(a, b))
    return days[(days.dayofweek >= 5) | days.isin(hol)]


class Tree:
    def __init__(self, kind, seed=0, tf="none", **params):
        self.kind, self.seed, self.tf, self.params = kind, seed, tf, params

    def _X(self, df):
        return df[CAT + NUM + LAGS].to_numpy(float)

    def fit(self, tr):
        if self.kind == "hgb":
            est = HistGradientBoostingRegressor(
                random_state=self.seed,
                early_stopping=False,
                max_features=0.8,
                **self.params,
            )
        else:
            cls = RandomForestRegressor if self.kind == "rf" else ExtraTreesRegressor
            est = cls(n_estimators=200, random_state=self.seed, n_jobs=1, **self.params)
        y = TRANSFORMS[self.tf][0](tr["y"].clip(lower=0).to_numpy())
        self.m = est.fit(self._X(tr), y)
        return self

    def predict(self, df):
        return np.clip(TRANSFORMS[self.tf][1](self.m.predict(self._X(df))), 0, None)


def _grid(**kw):
    return [dict(zip(kw, v)) for v in product(*kw.values())]


_FOREST = _grid(min_samples_leaf=[1, 5], max_features=[0.33, 0.6, 1.0])
_HGB = _grid(
    learning_rate=[0.05],
    max_iter=[300, 800],
    max_leaf_nodes=[15, 31, 63],
    l2_regularization=[0.0, 1.0],
)
_LIN = _grid(alpha=[0.1, 1.0, 10.0, 100.0], tf=["log1p", "sqrt", "none"])

# name -> (factory(params, seed), grid, seeded)
MODELS = {
    "seasonal_naive": (lambda p, s: Naive("lag168"), [{}], False),
    "persistence": (lambda p, s: Naive("lag24"), [{}], False),
    "ols": (lambda p, s: OLSRaw(), [{}], False),
    "cl": (lambda p, s: CarefulLinear(plus=False, **p), _LIN, False),
    "cl_plus": (lambda p, s: CarefulLinear(plus=True, **p), _LIN, False),
    "rf": (lambda p, s: Tree("rf", seed=s, **p), _FOREST, True),
    "et": (lambda p, s: Tree("et", seed=s, **p), _FOREST, True),
    "hgb": (lambda p, s: Tree("hgb", seed=s, **p), _HGB, True),
    "hgb_log": (lambda p, s: Tree("hgb", seed=s, tf="log1p", **p), _HGB, True),
}
TREES = ("rf", "et", "hgb", "hgb_log")


def _fit_predict(name, p, seed, tr, te):
    return MODELS[name][0](p, seed).fit(tr).predict(te)


def _fit_predict_clpp(p, tr, te):
    return CLpp(**p).fit(tr).predict(te)


def _label(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


def _rows(model, seed, split, keys, pred, meta):
    """Long-format rows. Forecasts are stored rounded to whole vehicles per hour
    (int16), which halves the cache; it moves no printed RMSE by more than 0.1.
    """
    pred = np.clip(np.rint(np.asarray(pred, float)), 0, np.iinfo(np.int16).max)
    return pd.DataFrame(
        {
            "model": model,
            "seed": seed,
            "split": split,
            "key": keys,
            "pred": pred.astype(np.int16),
            **meta,
        }
    )


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _search(names, folds, run):
    """Pooled walk-forward validation RMSE at seed 0 for every setting."""
    jobs = [
        (n, i, k)
        for n in names
        for i in range(len(MODELS[n][1]))
        for k in range(len(folds))
    ]
    preds = run(
        delayed(_fit_predict)(n, MODELS[n][1][i], 0, folds[k][0], folds[k][1])
        for n, i, k in jobs
    )
    yv = np.concatenate([va["y"].to_numpy() for _, va in folds])
    oof = {}
    for (n, i, _), p in zip(jobs, preds):
        oof.setdefault((n, i), []).append(p)
    oof = {key: np.concatenate(v) for key, v in oof.items()}
    best = {
        n: min(range(len(MODELS[n][1])), key=lambda i: rmse(yv, oof[(n, i)]))
        for n in names
    }
    return best, oof, yv


def _run():
    df = load()
    folds, pre, te = splits(df)
    run = Parallel(n_jobs=-1)
    names = list(MODELS)
    best, oof, yv = _search(names, folds, run)
    dev_keys = np.concatenate([va.index.to_numpy() for _, va in folds])
    jobs = [(n, s) for n in names for s in (SEEDS if MODELS[n][2] else (0,))]
    preds = run(
        delayed(_fit_predict)(n, MODELS[n][1][best[n]], s, pre, te) for n, s in jobs
    )
    test = dict(zip(jobs, preds))
    out, cv = [], {}
    for n in names:
        p = MODELS[n][1][best[n]]
        cv[n] = rmse(yv, oof[(n, best[n])])
        meta = {"params": _label(p), "settings": json.dumps(p), "cv_rmse": cv[n]}
        out.append(_rows(n, 0, "dev", dev_keys, oof[(n, best[n])], meta))
        for s in SEEDS if MODELS[n][2] else (0,):
            out.append(_rows(n, s, "test", te.index.to_numpy(), test[(n, s)], meta))
    pair = sorted(TREES, key=cv.get)[:2]
    dev_blend = 0.5 * (oof[(pair[0], best[pair[0]])] + oof[(pair[1], best[pair[1]])])
    meta = {
        "params": "+".join(pair),
        "settings": json.dumps({"pair": pair}),
        "cv_rmse": rmse(yv, dev_blend),
    }
    out.append(_rows("blend", 0, "dev", dev_keys, dev_blend, meta))
    for s in SEEDS:
        p = 0.5 * (test[(pair[0], s)] + test[(pair[1], s)])
        out.append(_rows("blend", s, "test", te.index.to_numpy(), p, meta))
    return pd.concat(out, ignore_index=True)


def run() -> pd.DataFrame:
    """Rows (model, seed, split, key, pred, params, settings, cv_rmse). `dev` rows are
    seed 0's out-of-fold predictions over the four 2017 folds; `test` rows come from
    one refit on all pre-test hours, five seeds for the seeded models. `blend` is the
    equal-weight mean of the two trees with the lowest validation RMSE.
    """
    return cache.frame(CACHE_TAG, _run)


def settings(res, model) -> dict:
    return json.loads(res.loc[res["model"] == model, "settings"].iloc[0])


def pred(res, model, seed=0, split="test") -> pd.Series:
    g = res[(res["model"] == model) & (res["split"] == split)]
    if seed not in set(g["seed"]):
        seed = 0
    return g[g["seed"] == seed].set_index("key")["pred"].astype(float)


def truth(split="test") -> pd.Series:
    df = load()
    folds, _, te = splits(df)
    if split == "test":
        return te["y"]
    return pd.concat([va["y"] for _, va in folds])


# ------------------------------------------------------------------ learning curve
def _run_learning_curve():
    res = run()
    df = load()
    _, pre, te = splits(df)
    y = te["y"].to_numpy()
    pair = settings(res, "blend")["pair"]
    names = ["cl", "cl_plus", *pair]
    jobs = []
    for k in LC_MONTHS:
        start = (
            pre.index.min()
            if k is None
            else pd.Timestamp(TEST[0]) - pd.DateOffset(months=k)
        )
        for n in names:
            for s in SEEDS if MODELS[n][2] else (0,):
                jobs.append((k, start, n, s))
    preds = Parallel(n_jobs=-1)(
        delayed(_fit_predict)(n, settings(res, n), s, pre.loc[start:], te)
        for k, start, n, s in jobs
    )
    got = {(k, n, s): p for (k, _, n, s), p in zip(jobs, preds)}
    rows = []
    top = y >= np.quantile(y, TOP_Q)
    for k, start, n, s in jobs:
        p = got[(k, n, s)]
        rows.append((k, n, s, p))
        if n == pair[1]:
            rows.append((k, "blend", s, 0.5 * (got[(k, pair[0], s)] + p)))
    out = []
    for k, n, s, p in rows:
        start = (
            pre.index.min()
            if k is None
            else pd.Timestamp(TEST[0]) - pd.DateOffset(months=k)
        )
        tr = pre.loc[start:]
        out.append(
            {
                "months": k if k is not None else -1,
                "train_from": start,
                "n_train": len(tr),
                "model": n,
                "seed": s,
                "rmse": rmse(y, p),
                "mae": float(np.mean(np.abs(y - p))),
                "rmse_top10": rmse(y[top], p[top]),
            }
        )
    return pd.DataFrame(out)


def learning_curve() -> pd.DataFrame:
    """Test scores (not predictions) training on the last k months before 2018
    (k = 6, 12, 24, 36; months = -1 is every pre-test hour), with each model's
    setting fixed at the one `run()` chose on full history. CL+'s ridge penalty was
    chosen for 63 months of data, so it overfits the shortest windows.
    """
    return cache.frame(LC_TAG, _run_learning_curve)


# ------------------------------------------------------------------ CL++ probe
def _run_probe():
    df = load()
    folds, pre, te = splits(df)
    run = Parallel(n_jobs=-1)
    preds = run(delayed(_fit_predict_clpp)(p, tr, va) for p in _LIN for tr, va in folds)
    yv = np.concatenate([va["y"].to_numpy() for _, va in folds])
    nf = len(folds)
    oof = [np.concatenate(preds[i * nf : (i + 1) * nf]) for i in range(len(_LIN))]
    i = int(np.argmin([rmse(yv, p) for p in oof]))
    p = _LIN[i]
    test = CLpp(**p).fit(pre).predict(te)
    meta = {"params": _label(p), "settings": json.dumps(p), "cv_rmse": rmse(yv, oof[i])}
    dev_keys = np.concatenate([va.index.to_numpy() for _, va in folds])
    return pd.concat(
        [
            _rows("cl_pp", 0, "dev", dev_keys, oof[i], meta),
            _rows("cl_pp", 0, "test", te.index.to_numpy(), test, meta),
        ],
        ignore_index=True,
    )


def probe() -> pd.DataFrame:
    """CL++ rows, same format as `run()`. Post-hoc: see `CLpp`."""
    return cache.frame(PROBE_TAG, _run_probe)


# ------------------------------------------------------------------ scoring
def scores(p, y, cong) -> dict:
    """Error and decision scores for one test forecast."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    top = y >= np.quantile(y, TOP_Q)
    a, f = y >= cong, p >= cong
    tp = int(np.sum(a & f))
    return {
        "rmse": rmse(y, p),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse_top10": rmse(y[top], p[top]),
        "peak_within15": float(np.mean(np.abs(p[top] - y[top]) <= 0.15 * y[top])),
        "peak_shortfall": float(np.mean(np.clip(y[top] - p[top], 0, None))),
        "cong_f1": 2 * tp / (a.sum() + f.sum()),
        "cong_recall": tp / a.sum(),
        "cong_precision": tp / max(f.sum(), 1),
    }


def gap(p, ref, y) -> dict:
    """RMSE(p) - RMSE(ref) on the test hours, with a 95% interval from circular
    moving blocks of 168 rows (seven days); `pct` scales both by RMSE(ref).
    """
    p, ref, y = (np.asarray(a, float) for a in (p, ref, y))
    lo, hi = modern.block_bootstrap(p, ref, y, block=BLOCK)
    r = rmse(y, ref)
    d = rmse(y, p) - r
    return {
        "diff": d,
        "lo": lo,
        "hi": hi,
        "pct": 100 * d / r,
        "pct_lo": 100 * lo / r,
        "pct_hi": 100 * hi / r,
    }


def day_type(index) -> np.ndarray:
    df = load().reindex(index)
    return np.where(
        df["holiday"] == 1,
        "holiday",
        np.where(df["workingday"] == 1, "working day", "weekend"),
    )
