"""Write widget-data/data.js for Part 1's goodness-of-fit widget.

Run from the post directory after `src/warm_cache.py`:
    ../../.venv-two-cultures/bin/python src/export_widget_data.py
It reads cache/gof_v1.parquet, so the widget shows the same replicates as the
post's figure.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("src").resolve()))

from twocultures import gof  # noqa: E402


def main() -> None:
    df = pd.read_parquet("cache/gof_v1.parquet")
    df["ratio"] = df["mse_rf"] / df["mse_ols"]
    Z = gof.predictors()
    out = {"etas": list(gof.ETAS), "reps": int(df["rep"].nunique()), "shapes": {}}
    for shape in gof.SHAPES:
        rows = []
        for eta in gof.ETAS:
            d = df[(df["shape"] == shape) & (df["eta"] == eta)]
            fit, res = gof.residual_sample(Z, gof.COLS, eta, shape=shape)
            rows.append(
                {
                    "eta": eta,
                    "reset": float((d["p_reset"] < 0.05).mean()),
                    "quad": float((d["p_quad"] < 0.05).mean()),
                    "rainbow": float((d["p_rainbow"] < 0.05).mean()),
                    "ratio_med": float(d["ratio"].median()),
                    "ratio_lo": float(d["ratio"].quantile(0.25)),
                    "ratio_hi": float(d["ratio"].quantile(0.75)),
                    "fit": np.round(np.asarray(fit), 2).tolist(),
                    "res": np.round(np.asarray(res), 2).tolist(),
                }
            )
        out["shapes"][shape] = rows
    Path("widget-data").mkdir(exist_ok=True)
    Path("widget-data/data.js").write_text(
        "window.WK_DATA = Object.assign(window.WK_DATA || {}, "
        + json.dumps({"gof": out}, separators=(",", ":"))
        + ");\n"
    )


if __name__ == "__main__":
    main()
