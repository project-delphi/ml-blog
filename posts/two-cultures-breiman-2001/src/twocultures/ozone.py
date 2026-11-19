"""The ozone stand-in (Breiman 2001, sec. 3.1) on the 1976 LA data.

Design: predict day t+1's maximum one-hour ozone from days t, t-1 and t-2 of
eight meteorological readings and ozone itself (27 columns), plus season (sine
and cosine of day of year) and a weekend flag: 30 candidate predictors.
The split is by the day t a forecast is made: training rows have t in January-August
(so the last target is 1 September), test rows t in September-December.

The linear route re-enacts the 1970s project as Breiman describes it: a large
linear regression with variable selection, then squares and interactions of the
retained variables with selection again. Every selection step is scored by
time-ordered cross-validation inside the training months.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from . import data

BASE = ["ozone", "vh", "wind", "humidity", "temp_sandburg", "ibh", "dpg", "ibt", "vis"]
LAGS = (0, 1, 2)
SPLIT_DOY = 245  # 1 September 1976 (a leap year)
ALERT = 20  # a relative threshold in the data's (undocumented) units


def design() -> pd.DataFrame:
    """The 30-column next-day design, indexed by day of year of day t."""
    m = data.ozone_mlbench()
    m["doy"] = m["date"].dt.day_of_year
    pairs = data.next_day(m, BASE, "ozone", lags=LAGS)
    doy = pairs.index.to_numpy()
    pairs.insert(0, "season_sin", np.sin(2 * np.pi * doy / 366))
    pairs.insert(1, "season_cos", np.cos(2 * np.pi * doy / 366))
    wd = m.set_index("doy").loc[doy, "weekday"].to_numpy()
    # mlbench codes weekday 1 = Monday ... 7 = Sunday; the target day is t + 1.
    pairs.insert(2, "weekend_next", np.isin((wd % 7) + 1, [6, 7]).astype(float))
    return pairs


def split(df: pd.DataFrame):
    tr = df.index < SPLIT_DOY
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    return X[tr], y[tr], X[~tr], y[~tr]


def _cv_mse(X: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    errs = []
    for tr, te in TimeSeriesSplit(folds).split(X):
        m = LinearRegression().fit(X[tr], y[tr])
        errs.append(np.mean((m.predict(X[te]) - y[te]) ** 2))
    return float(np.mean(errs))


def forward(X: pd.DataFrame, y: np.ndarray, max_vars: int = 15):
    """Greedy forward selection; the size is the CV-error minimiser."""
    chosen, path = [], []
    left = list(X.columns)
    while left and len(chosen) < max_vars:
        scores = {c: _cv_mse(X[chosen + [c]].to_numpy(), y) for c in left}
        best = min(scores, key=scores.get)
        chosen.append(best)
        left.remove(best)
        path.append((list(chosen), scores[best]))
    return min(path, key=lambda p: p[1])[0]


def expand(X: pd.DataFrame, cols) -> pd.DataFrame:
    """The retained columns plus their squares and pairwise products."""
    out = X[cols].copy()
    for c in cols:
        out[f"{c}^2"] = X[c] ** 2
    for a, b in combinations(cols, 2):
        out[f"{a}*{b}"] = X[a] * X[b]
    return out


def linear_route(Xtr, ytr, Xte):
    """Stage 1 and stage 2 selected linear models; returns predictions and terms."""
    s1 = forward(Xtr, ytr)
    m1 = LinearRegression().fit(Xtr[s1], ytr)
    Etr, Ete = expand(Xtr, s1), expand(Xte, s1)
    s2 = forward(Etr, ytr, max_vars=20)
    m2 = LinearRegression().fit(Etr[s2], ytr)
    return {
        "stage1_terms": s1,
        "stage1_pred": m1.predict(Xte[s1]),
        "stage2_terms": s2,
        "stage2_pred": m2.predict(Ete[s2]),
    }


def forest_route(Xtr, ytr, Xte, seed=0):
    rf = RandomForestRegressor(
        n_estimators=500, max_features=1 / 3, random_state=seed, n_jobs=-1
    ).fit(Xtr, ytr)
    return rf.predict(Xte), rf


def scores(pred, y, alert=ALERT) -> dict:
    """Error and alert-decision summaries on the test months."""
    alert_pred, alert_true = pred >= alert, y >= alert
    issued = int(alert_pred.sum())
    return {
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "r2": float(1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)),
        "alerts_issued": issued,
        "alert_days": int(alert_true.sum()),
        "hits": int((alert_pred & alert_true).sum()),
        "false_alarms": int((alert_pred & ~alert_true).sum()),
    }
