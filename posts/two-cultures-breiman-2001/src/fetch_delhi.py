"""Download ten years of New Delhi PM2.5 and weather, and write the daily files.

Run from the post directory with the series venv, naming a download directory
outside the repo (the hourly downloads are kept there, not committed):

    cd posts/two-cultures-breiman-2001
    ../../.venv-two-cultures/bin/python src/fetch_delhi.py ~/Downloads/delhi-raw

Sources, both open and keyless:

- PM2.5: the US Embassy monitor in New Delhi (AirNow's "US Diplomatic Post:
  New Delhi"), hourly, as archived by OpenAQ in its public S3 bucket
  (location 8118, one gzip CSV per day).
- Weather: ERA5 reanalysis at the monitor's coordinates, hourly, from the
  Open-Meteo historical archive API (`models=era5`).

Writes `data/raw/delhi_pm25_daily.csv.gz` and `data/raw/delhi_era5_daily.csv.gz`,
one row per local (India Standard Time) calendar day, 1 November 2016 to 31 October
2026 (OpenAQ's archive of the monitor starts on 9 November 2016, so the first eight
days have no readings), and prints the SHA-256 of both hourly downloads for `data/README.md`.
"""

import gzip
import hashlib
import io
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

if not Path("index.qmd").exists() or not Path("data/raw").is_dir():
    sys.exit("run from posts/two-cultures-breiman-2001 (it writes data/raw/)")

LOCATION, LAT, LON = 8118, 28.63576, 77.22445
START, END = "2016-11-01", "2026-10-31"
BUCKET = "https://openaq-data-archive.s3.amazonaws.com"
ERA5 = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5"
HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "boundary_layer_height",
]
MIN_HOURS = 18  # a day's mean needs 75% of its hours, the usual completeness rule
FAULT = 1985.0  # the monitor's error code, not a reading


def _keys(year: int) -> list[str]:
    prefix = f"records/csv.gz/locationid={LOCATION}/year={year}/"
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            return keys
        token = m.group(1)


def _get(key: str, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            r = requests.get(f"{BUCKET}/{key}", timeout=60)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2**i)


def _save(df: pd.DataFrame, path: Path) -> None:
    # Write then rename, so an interrupted run never leaves a partial file to reuse.
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(path)


def pm25_hourly(out: Path):
    # The window is in the name, so a download for another span is never reused.
    path = out / f"openaq_{LOCATION}_hourly_{START}_to_{END}.csv"
    if not path.exists():
        # A day's file runs 01:00 to the next 00:00, so END's file holds its last hour.
        # Cutting at END keeps the download, and its SHA-256, fixed after END.
        first, last = START.replace("-", ""), END.replace("-", "")
        keys = [
            k
            for y in range(int(START[:4]), int(END[:4]) + 1)
            for k in _keys(y)
            if (m := re.search(r"-(\d{8})\.csv\.gz$", k))
            and first <= m.group(1) <= last
        ]
        with ThreadPoolExecutor(6) as ex:
            parts = list(ex.map(_get, keys))
        frames = [pd.read_csv(io.BytesIO(gzip.decompress(b))) for b in parts]
        _save(pd.concat(frames, ignore_index=True), path)
    return pd.read_csv(path), path


def era5_hourly(out: Path):
    # Model and variables are in the name too, so changing either fetches afresh.
    tag = hashlib.sha256(",".join(HOURLY).encode()).hexdigest()[:8]
    path = out / f"era5_{MODEL}_{tag}_hourly_{START}_to_{END}.csv"
    if not path.exists():
        # One day either side: an accumulated value is labelled at the end of its hour,
        # so the first and last days need a neighbour's 00:00 or 23:00 label.
        day = pd.Timedelta(days=1)
        r = requests.get(
            ERA5,
            params={
                "latitude": LAT,
                "longitude": LON,
                "start_date": (pd.Timestamp(START) - day).strftime("%Y-%m-%d"),
                "end_date": (pd.Timestamp(END) + day).strftime("%Y-%m-%d"),
                "hourly": ",".join(HOURLY),
                "models": MODEL,
                "timezone": "Asia/Kolkata",
            },
            timeout=600,
        )
        r.raise_for_status()
        j = r.json()
        print(f"ERA5 grid point {j['latitude']}, {j['longitude']}")
        _save(pd.DataFrame(j["hourly"]), path)
    return pd.read_csv(path), path


def pm25_daily(h: pd.DataFrame) -> pd.DataFrame:
    h = h[h["parameter"] == "pm25"].copy()
    assert h["sensors_id"].nunique() == 1, "expected one PM2.5 sensor at the location"
    assert (h["datetime"].str[19:] == "+05:30").all(), (
        "expected India Standard Time stamps"
    )
    # OpenAQ stamps each hour at its end (a local day's file runs 01:00 to 00:00),
    # so an hour is counted on the day it started.
    t = pd.to_datetime(h["datetime"].str[:19]) - pd.Timedelta(hours=1)
    h["date"] = t.dt.normalize()
    h["hour"] = t.dt.hour
    # The feed codes a missing hour as -999 (or a small negative number) and an
    # instrument fault as 1985, often for several hours in a row.
    h = h[(h["value"] > 0) & (h["value"] != FAULT)]
    # July-November 2023 also carries half-hour stamps. Each is shifted back an hour
    # too and averaged with the hour it falls in; for stamps from 00:30 to 00:59 that
    # moves the reading to the previous day.
    h = h.groupby(["date", "hour"], as_index=False)["value"].mean()
    d = h.groupby("date")["value"].agg(pm25="mean", pm25_max="max", n_hours="size")
    d.loc[d["n_hours"] < MIN_HOURS, ["pm25", "pm25_max"]] = np.nan
    days = pd.date_range(START, END, freq="D")
    d = d.reindex(days)
    d["n_hours"] = d["n_hours"].fillna(0).astype(int)
    return d.rename_axis("date").reset_index()


def era5_daily(h: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(h["time"])
    # Instantaneous values are labelled at their hour; precipitation and radiation are
    # totals over the hour that ends at the label, so they count on the day that hour
    # started, as PM2.5 does.
    h = h.assign(
        date=t.dt.normalize(), date_acc=(t - pd.Timedelta(hours=1)).dt.normalize()
    )
    rad = np.deg2rad(h["wind_direction_10m"])
    # Wind blowing from the north-west (Punjab, Haryana) has u > 0, v < 0.
    h["wind_u"] = -h["wind_speed_10m"] * np.sin(rad)
    h["wind_v"] = -h["wind_speed_10m"] * np.cos(rad)
    g, ga = h.groupby("date"), h.groupby("date_acc")
    d = pd.DataFrame(
        {
            "temp_mean": g["temperature_2m"].mean(),
            "temp_max": g["temperature_2m"].max(),
            "temp_min": g["temperature_2m"].min(),
            "rh_mean": g["relative_humidity_2m"].mean(),
            "dewpoint_mean": g["dew_point_2m"].mean(),
            "pressure_mean": g["surface_pressure"].mean(),
            "precip_sum": ga["precipitation"].sum(),
            "wind_speed_mean": g["wind_speed_10m"].mean(),
            "wind_u_mean": g["wind_u"].mean(),
            "wind_v_mean": g["wind_v"].mean(),
            "radiation_sum": ga["shortwave_radiation"].sum() / 1000,  # kWh/m²
            "blh_mean": g["boundary_layer_height"].mean(),
            "blh_max": g["boundary_layer_height"].max(),
            "blh_min": g["boundary_layer_height"].min(),
        }
    )
    days = pd.date_range(START, END, freq="D")
    assert (g.size().reindex(days) == 24).all() and (
        ga.size().reindex(days) == 24
    ).all()
    return d.loc[days].round(3).rename_axis("date").reset_index()


def _write(df: pd.DataFrame, name: str) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(df.to_csv(index=False, date_format="%Y-%m-%d").encode())
    Path("data/raw", name).write_bytes(buf.getvalue())
    print(f"{name}: {len(df)} rows, {len(buf.getvalue()):,} bytes")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: fetch_delhi.py <download directory outside the repo>")
    out = Path(sys.argv[1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pm_h, pm_path = pm25_hourly(out)
    wx_h, wx_path = era5_hourly(out)
    pm, wx = pm25_daily(pm_h), era5_daily(wx_h)
    _write(pm, "delhi_pm25_daily.csv.gz")
    _write(wx, "delhi_era5_daily.csv.gz")
    for f in (pm_path, wx_path):
        print(f"{f.name}  sha256 {_sha(f)}")
    yr = pm.assign(y=pm["date"].dt.year).groupby("y")["pm25"]
    print(pd.DataFrame({"days": yr.count(), "mean": yr.mean().round(0)}))
