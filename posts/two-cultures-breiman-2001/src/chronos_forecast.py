"""Write Chronos-2's next-day ozone forecasts to cache/chronos2_v1.parquet.

Run from the post directory with the Chronos venv, which the post's kernel does
not have (torch is too heavy for the series' shared venv):

    uv venv .venv-two-cultures-chronos --python 3.12
    uv pip install --python .venv-two-cultures-chronos/bin/python \
        -r posts/two-cultures-breiman-2001/src/requirements-chronos.txt
    cd posts/two-cultures-breiman-2001
    ../../.venv-two-cultures-chronos/bin/python src/chronos_forecast.py

Zero-shot, rolling origin: the forecast made on day t sees the raw daily record
from 1 January to day t (missing days left as NaN, which Chronos-2 masks) and
predicts day t + 1. The point forecast is the median of the predicted
distribution. Rows are the dev and test forecast days of `twocultures.modern`.
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
from twocultures import data, modern, ozone  # noqa: E402

MODEL, REVISION = "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"
WEATHER = ["vh", "wind", "humidity", "temp_sandburg", "ibh", "dpg", "ibt", "vis"]


def daily() -> pd.DataFrame:
    m = data.ozone_mlbench()
    m["doy"] = m["date"].dt.day_of_year
    m = m.set_index("doy").sort_index()
    # mlbench codes weekday 1 = Monday ... 7 = Sunday.
    m["weekend"] = np.isin(m["weekday"], [6, 7]).astype(float)
    return m


def forecast(pipe, m, days, weather: bool) -> pd.DataFrame:
    inputs = []
    for t in days:
        h = m.loc[1:t]
        item = {"target": h["ozone"].to_numpy(float)}
        if weather:
            item["past_covariates"] = {
                c: h[c].to_numpy(float) for c in WEATHER + ["weekend"]
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
    df = ozone.design()
    _, dv, te = modern.masks(df)
    m = daily()
    out = []
    for name, weather in [("chronos_ozone", False), ("chronos_weather", True)]:
        for split, mask in [("dev", dv), ("test", te)]:
            f = forecast(pipe, m, df.index[mask].to_numpy(), weather)
            out.append(
                f.assign(
                    model=name, seed=0, split=split, params=f"{MODEL}@{REVISION[:7]}"
                )
            )
    res = pd.concat(out, ignore_index=True)
    modern.CHRONOS_FILE.parent.mkdir(exist_ok=True)
    res.to_parquet(modern.CHRONOS_FILE, index=False)
    y = df["target"]
    for (name, split), g in res.groupby(["model", "split"]):
        print(name, split, f"RMSE {modern.rmse(g['pred'], y.loc[g['doy']]):.2f}")
