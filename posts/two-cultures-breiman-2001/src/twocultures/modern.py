"""Models the paper does not run, on Part 1's next-day ozone design.

The design and the test months are those of `ozone`: forecasts made September-
December are scored. Model settings are chosen by walk-forward validation inside
the training period, never on the test months. Expanding-window folds start at
 successive forecast days; a three-day embargo separates each training window
from its validation block (the design uses lags of up to two days, plus the
next-day target). The mean RMSE over the folds picks the settings, and the chosen
model is then refit on every pre-test day and scored on the test months.

Chronos-2 is not trained or tuned. Its forecasts come from
`src/chronos_forecast.py`, which needs torch and runs in its own venv
(`src/requirements-chronos.txt`); this module only reads the file it writes.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import ozone

DEV_DOY = 183  # 1 July 1976 (a leap year)
SEEDS = range(5)
CACHE_TAG = (
    "ozone_modern_v3"  # the cache.frame name for run(); bump it when run() changes
)
CHRONOS_FILE = Path("cache") / "chronos2_v1.parquet"
EMBARGO = 3  # two lag days plus the next-day target, dropped at each fold edge
OZONE_FOLD_EDGES = (
    61,
    122,
    183,
    244,
)  # three validation blocks, March-August (doy 61-243)


def _grid(**kw):
    return [dict(zip(kw, v)) for v in product(*kw.values())]


def _fmt(v) -> str:
    """A setting as text; floats keep a decimal point, since sklearn reads max_features=1 as one column."""
    if isinstance(v, int):
        return str(v)
    s = f"{float(v):.3g}"
    return s if any(c in s for c in ".e") else s + ".0"


def lasso(alpha, seed=0):
    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=50_000))


def forest(max_features, min_samples_leaf, seed=0):
    return RandomForestRegressor(
        n_estimators=500,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )


def extra_trees(max_features, min_samples_leaf, seed=0):
    return ExtraTreesRegressor(
        n_estimators=500,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )


def boosting(
    learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization, max_iter, seed=0
):
    # Half the columns are tried at each split, so the seed matters.
    return HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_iter=max_iter,
        max_features=0.5,
        early_stopping=False,
        random_state=seed,
    )


class QuantileBoost:
    """Three quantile gradient boosters; the median is `pred`.

    Fits q10, q50 and q90 separately with the same settings, so the spread
    between the outer two is a predictive interval without a distributional
    assumption. `intervals` exposes the pair for the table and the figure.

    It deliberately does not implement sklearn's estimator interface
    (`get_params`/`clone`): the grid is fixed and instances are made only
    through `FACTORIES`.
    """

    QUANTILES = (0.1, 0.5, 0.9)

    def __init__(
        self, learning_rate, max_leaf_nodes, min_samples_leaf, max_iter, seed=0
    ):
        self.settings = dict(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            max_iter=max_iter,
            max_features=0.5,
            early_stopping=False,
            random_state=seed,
        )
        self.models = {}

    def fit(self, X, y):
        for q in self.QUANTILES:
            self.models[q] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, **self.settings
            ).fit(X, y)
        return self

    def predict(self, X):
        return self.models[0.5].predict(X)

    def intervals(self, X):
        return self.models[0.1].predict(X), self.models[0.9].predict(X)


# Grids are smaller than the single-dev version was: walk-forward selection
# fits each setting once per fold, so the same grid costs four times as much.
TUNED = {
    "lasso": (_grid(alpha=np.logspace(-3, 1, 20)), False),
    "forest_tuned": (
        _grid(max_features=[1 / 3, 1.0], min_samples_leaf=[1, 5, 10]),
        True,
    ),
    "boosting": (
        _grid(
            learning_rate=[0.03, 0.1],
            max_leaf_nodes=[8, 16],
            min_samples_leaf=[10, 20],
            l2_regularization=[0.0, 1.0],
            max_iter=[100, 200],
        ),
        True,
    ),
    "extra_trees": (
        _grid(max_features=[1 / 3, 1.0], min_samples_leaf=[1, 5, 10]),
        True,
    ),
    "quantile_boost": (
        _grid(
            learning_rate=[0.03, 0.1],
            max_leaf_nodes=[8, 16],
            min_samples_leaf=[10, 20],
            max_iter=[200],
        ),
        True,
    ),
}

FACTORIES = {
    "lasso": lambda p, s: lasso(**p),
    "forest_tuned": lambda p, s: forest(**p, seed=s),
    "boosting": lambda p, s: boosting(**p, seed=s),
    "extra_trees": lambda p, s: extra_trees(**p, seed=s),
    "quantile_boost": lambda p, s: QuantileBoost(**p, seed=s),
}


def folds_from_edges(index, edges, embargo=EMBARGO):
    """Expanding-window folds from validation-block boundaries.

    `edges` are values of the design index (day numbers), not row positions;
    the design has gaps, so positions are looked up through `index`. Each
    consecutive pair is a validation block. A fold trains on every row before
    the block, minus an embargo of `embargo` days at the edge.
    """
    return [
        (
            np.flatnonzero(index < v0 - embargo),
            np.flatnonzero((index >= v0) & (index < v1)),
        )
        for v0, v1 in zip(edges[:-1], edges[1:])
    ]


def masks(df: pd.DataFrame):
    """Boolean train, dev and test masks over the rows of `ozone.design()`."""
    t = df.index.to_numpy()
    return t < DEV_DOY, (t >= DEV_DOY) & (t < ozone.SPLIT_DOY), t >= ozone.SPLIT_DOY


def rmse(pred, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def run() -> pd.DataFrame:
    """Dev and test predictions on the 1976 design (see `fit_all`)."""
    df = ozone.design()
    return fit_all(
        df,
        *masks(df),
        persist="ozone_lag0",
        key="doy",
        folds=folds_from_edges(df.index.to_numpy(), OZONE_FOLD_EDGES),
    )


def fit_all(df, tr, dv, te, persist, key, folds) -> pd.DataFrame:
    """Out-of-fold and test predictions, one row per (model, seed, split, day).

    `df` is a next-day design with a `target` column. `folds` is a list of
    (training positions, validation positions) from `folds_from_edges`. Dev rows
    are out-of-fold predictions: each pre-test day is predicted by a model that
    did not see it (or the days around it). Test rows come from a refit on every
    pre-test day. `params` records the walk-forward-selected settings. Models
    with predictive intervals also carry `q10` and `q90` columns.
    """
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    fit = tr | dv
    test_idx = np.flatnonzero(te)
    dev_idx = np.concatenate([va for _, va in folds])
    # The comparison rests on this split. Fail loudly if an edited fold edge
    # reaches the test period or a fold trains on its own validation days.
    assert not np.intersect1d(dev_idx, test_idx).size, (
        "folds reach into the test period"
    )
    for tr_i, va_i in folds:
        assert not np.intersect1d(tr_i, va_i).size, "train and validation folds overlap"
    lag0 = X[persist].to_numpy()
    out = []

    def add(model, seed, split, idx, pred, params="", q10=None, q90=None):
        row = {
            "model": model,
            "seed": seed,
            "split": split,
            key: df.index.to_numpy()[idx],
            "pred": np.asarray(pred),
            "params": params,
        }
        if q10 is not None:
            row["q10"], row["q90"] = np.asarray(q10), np.asarray(q90)
        out.append(pd.DataFrame(row))

    def oof(factory, params, seed):
        """Out-of-fold predictions for one model over all walk-forward folds."""
        preds, los, his = [], [], []
        for tr_i, va_i in folds:
            m = factory(params, seed)
            m.fit(X.iloc[tr_i], y[tr_i])
            preds.append(m.predict(X.iloc[va_i]))
            if hasattr(m, "intervals"):
                lo, hi = m.intervals(X.iloc[va_i])
                los.append(lo)
                his.append(hi)
        return (
            np.concatenate(preds),
            np.concatenate(los) if los else None,
            np.concatenate(his) if his else None,
        )

    def cv_rmse(factory, params, seed):
        errs = []
        for tr_i, va_i in folds:
            m = factory(params, seed)
            m.fit(X.iloc[tr_i], y[tr_i])
            errs.append(rmse(m.predict(X.iloc[va_i]), y[va_i]))
        return float(np.mean(errs))

    add("persistence", 0, "dev", dev_idx, lag0[dev_idx])
    add("persistence", 0, "test", test_idx, lag0[test_idx])

    dev_parts = []
    for tr_i, va_i in folds:
        dev_parts.append(
            ozone.linear_route(X.iloc[tr_i], y[tr_i], X.iloc[va_i])["stage2_pred"]
        )
    add("linear_quad", 0, "dev", dev_idx, np.concatenate(dev_parts))
    add(
        "linear_quad",
        0,
        "test",
        test_idx,
        ozone.linear_route(X.iloc[fit], y[fit], X.iloc[test_idx])["stage2_pred"],
    )

    for s in SEEDS:
        if s == 0:  # only seed 0's dev rows are read; keeping all five bloats the cache
            pred, _, _ = oof(
                lambda p, seed: forest(**p, seed=seed),
                {"max_features": 1 / 3, "min_samples_leaf": 1},
                s,
            )
            add("forest", s, "dev", dev_idx, pred)
        add(
            "forest",
            s,
            "test",
            test_idx,
            forest(1 / 3, 1, seed=s).fit(X.iloc[fit], y[fit]).predict(X.iloc[test_idx]),
        )

    for name, (grid, seeded) in TUNED.items():
        factory = FACTORIES[name]
        scores = [cv_rmse(factory, p, 0) for p in grid]
        best = grid[int(np.argmin(scores))]
        label = ", ".join(f"{k}={_fmt(v)}" for k, v in best.items())
        for s in SEEDS if seeded else [0]:
            if s == 0:  # dev rows are read only for seed 0
                pred, lo, hi = oof(factory, best, s)
                add(name, s, "dev", dev_idx, pred, label, lo, hi)
            m = factory(best, s).fit(X.iloc[fit], y[fit])
            pred_t = m.predict(X.iloc[test_idx])
            lo_t, hi_t = (
                m.intervals(X.iloc[test_idx])
                if hasattr(m, "intervals")
                else (None, None)
            )
            add(name, s, "test", test_idx, pred_t, label, lo_t, hi_t)
    return pd.concat(out, ignore_index=True)


def chronos() -> pd.DataFrame:
    """Chronos-2 forecasts written by `src/chronos_forecast.py` (columns as `run`, plus q10, q90)."""
    if not CHRONOS_FILE.exists():
        raise FileNotFoundError(
            f"{CHRONOS_FILE} is missing: run src/chronos_forecast.py from the post directory "
            "with the venv in src/requirements-chronos.txt"
        )
    return pd.read_parquet(CHRONOS_FILE)


def block_bootstrap(pred, ref, y, block=7, reps=4000, seed=0):
    """95% interval for RMSE(pred) - RMSE(ref) on the test days.

    Circular moving blocks of `block` consecutive rows, so the first and last
    days are drawn as often as the middle ones.
    """
    pred, ref, y = (np.asarray(a, float) for a in (pred, ref, y))
    n = len(y)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(reps, -(-n // block)))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]
    d = np.sqrt(((pred[idx] - y[idx]) ** 2).mean(1)) - np.sqrt(
        ((ref[idx] - y[idx]) ** 2).mean(1)
    )
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
