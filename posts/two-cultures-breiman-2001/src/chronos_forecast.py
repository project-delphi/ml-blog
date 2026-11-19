"""Write Chronos-2's next-day forecasts: ozone (1976) or New Delhi PM2.5 (2016-2026).

Run from the post directory with the Chronos venv, which the post's kernel does
not have (torch is too heavy for the series' shared venv):

    uv venv .venv-two-cultures-chronos --python 3.12
    uv pip install --python .venv-two-cultures-chronos/bin/python \
        -r posts/two-cultures-breiman-2001/src/requirements-chronos.txt
    cd posts/two-cultures-breiman-2001
    ../../.venv-two-cultures-chronos/bin/python src/chronos_forecast.py          # ozone
    ../../.venv-two-cultures-chronos/bin/python src/chronos_forecast.py delhi    # PM2.5

Zero-shot, rolling origin: the forecast made on day t sees the raw daily record
from its first day to day t (missing days left as NaN, which Chronos-2 masks) and
predicts day t + 1. The point forecast is the median of the predicted
distribution. Rows are the dev and test forecast days of `twocultures.modern`
(ozone, written to cache/chronos2_v1.parquet) or `twocultures.delhi` (PM2.5,
cache/chronos2_delhi_v2.parquet).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not Path("index.qmd").exists() or not Path("src/twocultures").is_dir():
    sys.exit(
        "run from posts/two-cultures-breiman-2001 (it reads src/ and writes cache/)"
    )
sys.path.insert(0, str(Path("src").resolve()))

from chronos import BaseChronosPipeline  # noqa: E402
from twocultures import data, delhi, modern, ozone  # noqa: E402

MODEL, REVISION = "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"
WEATHER = ["vh", "wind", "humidity", "temp_sandburg", "ibh", "dpg", "ibt", "vis"]


def daily() -> pd.DataFrame:
    m = data.ozone_mlbench()
    m["doy"] = m["date"].dt.day_of_year
    m = m.set_index("doy").sort_index()
    # mlbench codes weekday 1 = Monday ... 7 = Sunday.
    m["weekend"] = np.isin(m["weekday"], [6, 7]).astype(float)
    return m


def delhi_daily() -> pd.DataFrame:
    d = delhi.daily()
    # Every calendar day is a row, so a gap in the record is a NaN, never a skipped day.
    assert (np.diff(d.index) == 1).all()
    return d


def forecast(pipe, m, days, weather: bool, target="ozone", cov=WEATHER) -> pd.DataFrame:
    inputs = []
    for t in days:
        h = m.loc[m.index[0] : t]
        item = {"target": h[target].to_numpy(float)}
        if weather:
            item["past_covariates"] = {
                c: h[c].to_numpy(float) for c in cov + ["weekend"]
            }
            item["future_covariates"] = {"weekend": np.array([m.loc[t + 1, "weekend"]])}
        inputs.append(item)
    q, _ = pipe.predict_quantiles(
        inputs, prediction_length=1, quantile_levels=[0.1, 0.5, 0.9]
    )
    q = np.stack([x[0, 0].numpy() for x in q])
    return pd.DataFrame({"doy": days, "q10": q[:, 0], "pred": q[:, 1], "q90": q[:, 2]})


if __name__ == "__main__":
    pipe = BaseChronosPipeline.from_pretrained(
        MODEL, revision=REVISION, device_map="cpu"
    )
    if sys.argv[1:] not in ([], ["delhi"]):
        sys.exit(
            "usage: chronos_forecast.py [delhi]  (no argument runs the 1976 ozone forecasts)"
        )
    if sys.argv[1:] == ["delhi"]:
        df, m, key, path = delhi.design(), delhi_daily(), "day", delhi.CHRONOS_FILE
        _, dv, te = delhi.masks(df)
        kw = {"target": "pm25", "cov": delhi.WEATHER}
        names = [("chronos_pm25", False), ("chronos_weather", True)]
    else:
        df, m, key, path = ozone.design(), daily(), "doy", modern.CHRONOS_FILE
        _, dv, te = modern.masks(df)
        kw = {}
        names = [("chronos_ozone", False), ("chronos_weather", True)]
    out = []
    for name, weather in names:
        for split, mask in [("dev", dv), ("test", te)]:
            f = forecast(pipe, m, df.index[mask].to_numpy(), weather, **kw)
            out.append(
                f.rename(columns={"doy": key}).assign(
                    model=name, seed=0, split=split, params=f"{MODEL}@{REVISION[:7]}"
                )
            )
    res = pd.concat(out, ignore_index=True)
    path.parent.mkdir(exist_ok=True)
    res.to_parquet(path, index=False)
    y = df["target"]
    for (name, split), g in res.groupby(["model", "split"]):
        print(name, split, f"RMSE {modern.rmse(g['pred'], y.loc[g[key]]):.2f}")
