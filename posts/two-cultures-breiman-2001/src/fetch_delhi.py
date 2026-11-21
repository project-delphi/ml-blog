"""Download eleven years of New Delhi PM2.5 and weather, and write the daily files.

Run from the post directory with the series venv, naming a download directory
outside the repo (the hourly downloads are kept there, not committed):

    cd posts/two-cultures-breiman-2001
    ../../.venv-two-cultures/bin/python src/fetch_delhi.py ~/Downloads/delhi-raw

Sources, both open and keyless:

- PM2.5: the US Embassy monitor in New Delhi, hourly, from the US State Department's
  own historical files on AirNow-Tech's public S3 bucket
  (`files.airnowtech.org/airnow/EmbassyHistorical/NewDelhi/`, one CSV per year and per
  month). This replaces OpenAQ location 8118, which mirrors the same feed but carries
  the NowCast instead of the hourly value until mid-March 2017, drops, repeats and
  shifts hours in several later stretches, and passes through hours AirNow marks
  Invalid (`data/README.md` has the evidence).
- Weather: ERA5 reanalysis at the monitor's coordinates, hourly, from the Open-Meteo
  historical archive API (`models=era5`), unchanged from the OpenAQ version.

Writes `data/raw/delhi_pm25_daily.csv.gz` and `data/raw/delhi_era5_daily.csv.gz`,
one row per local (India Standard Time) calendar day, 1 November 2015 to 31 October
2026, and `airnow_sha256.txt` in the download directory, a SHA-256 for every AirNow
file. A later run in the same directory reads its file list from that manifest and
checks every file against it instead of listing the bucket again, so it rebuilds the
same two daily files offline; delete the manifest to fetch afresh.
"""

import csv
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

LAT, LON = 28.63576, 77.22445
START, END = "2015-11-01", "2026-10-31"
AIRNOW = "https://s3-us-west-1.amazonaws.com/files.airnowtech.org"
PREFIX = "airnow/EmbassyHistorical/NewDelhi/"
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
MANIFEST = "airnow_sha256.txt"  # in the download directory: the file list, pinned


def _keys() -> list[str]:
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": PREFIX}
        if token:
            params["continuation-token"] = token
        r = requests.get(AIRNOW, params=params, timeout=60)
        r.raise_for_status()
        keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            return [k for k in keys if k.endswith(".csv") and "PM2.5" in k]
        token = m.group(1)


def _get(key: str, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            r = requests.get(f"{AIRNOW}/{key}", timeout=60)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2**i)


def _read(raw: bytes, name: str) -> pd.DataFrame:
    # Four header layouts across the years; the value is `Raw Conc.` where present
    # (never the NowCast), else `Value` (Dec 2015 - Feb 2016 monthly files).
    lines = raw.decode("ascii").replace("\r\n", "\n").strip("\n").split("\n")
    header = next(csv.reader([lines[0]]))
    rows = [
        # Oct 2016's monthly file writes each row as a quoted Python list
        [c.strip() for c in ln[2:-2].split(",")]
        if ln.startswith('"[')
        else next(csv.reader([ln]))
        for ln in lines[1:]
    ]
    df = pd.DataFrame(rows, columns=header)
    assert (df["Site"] == "New Delhi").all() and (df["Duration"] == "1 Hr").all()
    x = pd.DataFrame(
        {
            # local time, 12-hour clock
            "t": pd.to_datetime(df["Date (LT)"], format="%Y-%m-%d %I:%M %p"),
            "value": pd.to_numeric(df["Raw Conc." if "Raw Conc." in df else "Value"]),
            "qc": df["QC Name"],
            "ytd": name.endswith("_YTD.csv"),
        }
    )
    # On the US spring-forward Sunday (2015-03-08 to 2018-03-11) a file has no 02:00
    # and two 03:00 rows: put the first back at 02:00.
    dup = x["t"].duplicated(keep="last")
    prev = x["t"] - pd.Timedelta(hours=1)
    fix = dup & ~prev.isin(set(x["t"]))
    assert dup.sum() == fix.sum() <= 1, name
    x.loc[fix, "t"] = prev[fix]
    return x


def pm25_hourly(out: Path):
    # One folder per year, but some files sit a folder early under the next year's
    # name (`2017/NewDelhi_PM2.5_2018_12_MTD.csv` holds December 2017), and the
    # correctly named December files stop a day short. Everything is fetched and
    # deduplicated by stamp, so neither matters.
    manifest = out / MANIFEST
    if manifest.exists():
        pinned = dict(ln.split("  ")[::-1] for ln in manifest.read_text().splitlines())
        keys = [PREFIX + k[len("airnow/") :] for k in pinned]
    else:
        keys = _keys()
    paths = [out / "airnow" / k[len(PREFIX) :] for k in keys]
    # Without a manifest every file is fetched again: the current year's YTD and
    # MTD files keep growing, so a copy on disk may be stale.
    fresh = not manifest.exists()
    todo = [(k, p) for k, p in zip(keys, paths) if fresh or not p.exists()]
    with ThreadPoolExecutor(6) as ex:
        for (k, p), b in zip(todo, ex.map(_get, [k for k, _ in todo])):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.with_suffix(".tmp").write_bytes(b)
            p.with_suffix(".tmp").rename(p)
    if manifest.exists():
        bad = [k for k, p in zip(pinned, paths) if _sha(p) != pinned[k]]
        assert not bad, f"{len(bad)} files differ from {MANIFEST}, e.g. {bad[0]}"
    h = pd.concat([_read(p.read_bytes(), p.name) for p in paths], ignore_index=True)
    return h, paths


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
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(j["hourly"]).to_csv(tmp, index=False)
        tmp.rename(path)
    return pd.read_csv(path), path


def pm25_daily(h: pd.DataFrame) -> pd.DataFrame:
    # Only hours AirNow marks Valid count. -999 (missing; it also appears marked Valid
    # in the 2015-16 monthly files), small negatives, zeros and the 1985 fault code
    # are dropped.
    h = h[(h["qc"] == "Valid") & (h["value"] > 0) & (h["value"] != FAULT)]
    # A stamp appears in a year file and a month file, with the same value.
    h = h.sort_values(["t", "ytd"], ascending=[True, False], kind="stable")
    h = h.drop_duplicates("t")
    # The stamps coincide with OpenAQ's, which mark the hour's end (a year's file runs
    # 01:00 to the next 00:00 through 2019, 02:00 to 01:00 after), so an hour is
    # counted on the day it started.
    t = h["t"] - pd.Timedelta(hours=1)
    h = h.assign(date=t.dt.normalize())
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
    pm_h, pm_paths = pm25_hourly(out)
    wx_h, wx_path = era5_hourly(out)
    pm, wx = pm25_daily(pm_h), era5_daily(wx_h)
    _write(pm, "delhi_pm25_daily.csv.gz")
    _write(wx, "delhi_era5_daily.csv.gz")
    # The AirNow files are the raw responses, so their hashes change only if the State
    # Department revises a file (the current year's files grow until it ends).
    manifest = "".join(f"{_sha(p)}  {p.relative_to(out)}\n" for p in pm_paths)
    (out / MANIFEST).write_text(manifest)
    print(
        f"{len(pm_paths)} AirNow files, {MANIFEST} sha256 "
        f"{hashlib.sha256(manifest.encode()).hexdigest()}"
    )
    print(f"{wx_path.name}  sha256 {_sha(wx_path)}")
    # Days with a mean per November-October year, labelled by the year it ends in.
    year = (pm["date"] + pd.DateOffset(months=2)).dt.year
    yr = pm.groupby(year.rename("y"))["pm25"]
    print(pd.DataFrame({"days": yr.count(), "mean": yr.mean().round(0)}))
