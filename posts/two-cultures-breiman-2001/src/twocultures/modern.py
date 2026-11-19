"""Models the paper does not run, on Part 1's next-day ozone design.

The design and the test months are those of `ozone`: forecasts made September-
December are scored. For tuning, the training months are split again by the
forecast day t: January-June trains, July-August is the dev set. Each tabular
model's settings are the dev-RMSE minimiser over the grids below; the chosen
model is then refit on January-August and scored on the test months.

Chronos-2 is not trained or tuned. Its forecasts come from
`src/chronos_forecast.py`, which needs torch and runs in its own venv
(`src/requirements-chronos.txt`); this module only reads the file it writes.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import ozone

DEV_DOY = 183  # 1 July 1976 (a leap year)
SEEDS = range(5)
CACHE_TAG = (
    "ozone_modern_v2"  # the cache.frame name for run(); bump it when run() changes
)
CHRONOS_FILE = Path("cache") / "chronos2_v1.parquet"


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


TUNED = {
    "lasso": (lasso, _grid(alpha=np.logspace(-3, 1, 30)), False),
    "forest_tuned": (
        forest,
        _grid(max_features=[0.2, 1 / 3, 0.5, 1.0], min_samples_leaf=[1, 3, 5, 10]),
        True,
    ),
    "boosting": (
        boosting,
        _grid(
            learning_rate=[0.03, 0.1],
            max_leaf_nodes=[4, 8, 16],
            min_samples_leaf=[5, 10, 20],
            l2_regularization=[0.0, 1.0],
            max_iter=[100, 200, 400],
        ),
        True,
    ),
}


def masks(df: pd.DataFrame):
    """Boolean train, dev and test masks over the rows of `ozone.design()`."""
    t = df.index.to_numpy()
    return t < DEV_DOY, (t >= DEV_DOY) & (t < ozone.SPLIT_DOY), t >= ozone.SPLIT_DOY


def rmse(pred, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def run() -> pd.DataFrame:
    """Dev and test predictions, one row per (model, seed, split, day t).

    Dev predictions come from fits on January-June, test predictions from
    refits on January-August. `params` records the chosen settings.
    """
    df = ozone.design()
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    tr, dv, te = masks(df)
    fit = tr | dv
    out = []

    def add(model, seed, split, mask, pred, params=""):
        out.append(
            pd.DataFrame(
                {
                    "model": model,
                    "seed": seed,
                    "split": split,
                    "doy": df.index[mask],
                    "pred": pred,
                    "params": params,
                }
            )
        )

    lag0 = X["ozone_lag0"].to_numpy()
    add("persistence", 0, "dev", dv, lag0[dv])
    add("persistence", 0, "test", te, lag0[te])
    add(
        "linear_quad",
        0,
        "dev",
        dv,
        ozone.linear_route(X[tr], y[tr], X[dv])["stage2_pred"],
    )
    add(
        "linear_quad",
        0,
        "test",
        te,
        ozone.linear_route(X[fit], y[fit], X[te])["stage2_pred"],
    )
    for s in SEEDS:
        add("forest", s, "dev", dv, ozone.forest_route(X[tr], y[tr], X[dv], seed=s)[0])
        add(
            "forest",
            s,
            "test",
            te,
            ozone.forest_route(X[fit], y[fit], X[te], seed=s)[0],
        )

    for name, (make, grid, seeded) in TUNED.items():
        dev_rmse = [
            rmse(make(**p).fit(X[tr], y[tr]).predict(X[dv]), y[dv]) for p in grid
        ]
        best = grid[int(np.argmin(dev_rmse))]
        label = ", ".join(f"{k}={_fmt(v)}" for k, v in best.items())
        for s in SEEDS if seeded else [0]:
            add(
                name,
                s,
                "dev",
                dv,
                make(**best, seed=s).fit(X[tr], y[tr]).predict(X[dv]),
                label,
            )
            add(
                name,
                s,
                "test",
                te,
                make(**best, seed=s).fit(X[fit], y[fit]).predict(X[te]),
                label,
            )
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
