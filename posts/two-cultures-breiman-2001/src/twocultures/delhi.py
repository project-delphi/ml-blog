"""Part 1's next-day design on ten years of New Delhi PM2.5 (November 2016 to October 2026).

The target is day t+1's 24-hour mean PM2.5 at the US Embassy monitor, in µg/m³.
The candidates mirror the 1976 design: days t, t-1 and t-2 of PM2.5 and eight
ERA5 surface readings (27 columns), plus season and a weekend flag, 30 in all.

The years run November to October, so each holds one winter. Split by the
forecast day t: the test period is November 2023 to October 2026. Model settings
are chosen by expanding-window walk-forward CV on the earlier years (four
November-October validation blocks, a three-day embargo at each edge); the
chosen models are refit on every pre-test day before being scored. The models
and grids are `modern`'s, unchanged.

Chronos-2's forecasts come from `src/chronos_forecast.py delhi`, as for ozone.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import data, modern

START = pd.Timestamp("2016-11-01")
DEV_START, TEST_START = pd.Timestamp("2022-11-01"), pd.Timestamp("2023-11-01")
WEATHER = [
    "temp_max",
    "temp_min",
    "rh_mean",
    "pressure_mean",
    "precip_sum",
    "wind_speed_mean",
    "wind_u_mean",
    "wind_v_mean",
]
BASE = ["pm25"] + WEATHER
LAGS = (0, 1, 2)
ALERT = 250  # India's National AQI "Severe" band for a 24-hour PM2.5 mean (250+ µg/m³)
CACHE_TAG = "delhi_v3"  # the cache.frame name for run(); bump it when run() changes
CHRONOS_FILE = Path("cache") / "chronos2_delhi_v2.parquet"


def daily() -> pd.DataFrame:
    """One row per day, indexed by days since 1 November 2016 (`day`)."""
    pm = pd.read_csv(data.RAW / "delhi_pm25_daily.csv.gz", parse_dates=["date"])
    wx = pd.read_csv(data.RAW / "delhi_era5_daily.csv.gz", parse_dates=["date"])
    d = pm.merge(wx, on="date", how="inner", validate="one_to_one")
    d["day"] = (d["date"] - START).dt.days
    d["weekend"] = (d["date"].dt.dayofweek >= 5).astype(float)
    return d.set_index("day", drop=False).sort_index()


def design() -> pd.DataFrame:
    """The 30-column next-day design, indexed by the day t a forecast is made."""
    d = daily()
    pairs = data.next_day(d, BASE, "pm25", lags=LAGS, day="day")
    t = pairs.index.to_numpy()
    doy = d.loc[t, "date"].dt.day_of_year.to_numpy()
    pairs.insert(0, "season_sin", np.sin(2 * np.pi * doy / 365.25))
    pairs.insert(1, "season_cos", np.cos(2 * np.pi * doy / 365.25))
    pairs.insert(2, "weekend_next", d.loc[t + 1, "weekend"].to_numpy())
    return pairs


def day_of(ts: pd.Timestamp) -> int:
    return int((ts - START).days)


# Four expanding-window validation blocks: Nov 2019-Oct 2020 ... Nov 2022-Oct 2023.
FOLD_EDGES = tuple(
    day_of(pd.Timestamp(f"{y}-11-01")) for y in (2019, 2020, 2021, 2022, 2023)
)


def masks(df: pd.DataFrame):
    """Boolean train, dev and test masks over the rows of `design()`."""
    t = df.index.to_numpy()
    dev, test = day_of(DEV_START), day_of(TEST_START)
    return t < dev, (t >= dev) & (t < test), t >= test


def dates(days) -> pd.DatetimeIndex:
    return START + pd.to_timedelta(np.asarray(days), unit="D")


def run() -> pd.DataFrame:
    df = design()
    return modern.fit_all(
        df,
        *masks(df),
        persist="pm25_lag0",
        key="day",
        folds=modern.folds_from_edges(df.index.to_numpy(), FOLD_EDGES),
    )


def chronos() -> pd.DataFrame:
    """Chronos-2 forecasts written by `src/chronos_forecast.py delhi`."""
    if not CHRONOS_FILE.exists():
        raise FileNotFoundError(
            f"{CHRONOS_FILE} is missing: run `src/chronos_forecast.py delhi` from the post "
            "directory with the venv in src/requirements-chronos.txt"
        )
    return pd.read_parquet(CHRONOS_FILE)
