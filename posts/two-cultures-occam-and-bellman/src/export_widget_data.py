"""Write widget-data/data.js for Part 3's double-descent widget.

Run from the post directory after src/warm_cache.py (it reads
cache/randfeat_v1.parquet and cache/randfeat_ridge_v1.parquet).
"""

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    out = {"n_train": 1000, "series": {}}
    for key, f in [("minnorm", "randfeat_v1"), ("ridge", "randfeat_ridge_v1")]:
        df = pd.read_parquet(f"cache/{f}.parquet")
        g = (
            df.groupby("width")
            .agg(
                train=("train_error", "mean"),
                test=("test_error", "mean"),
                test_lo=("test_error", "min"),
                test_hi=("test_error", "max"),
                norm=("norm", "mean"),
            )
            .reset_index()
        )
        out["series"][key] = {c: [round(float(v), 4) for v in g[c]] for c in g.columns}
        out["seeds"] = int(df["seed"].nunique())
    Path("widget-data").mkdir(exist_ok=True)
    Path("widget-data/data.js").write_text(
        "window.WK_DATA = Object.assign(window.WK_DATA || {}, "
        + json.dumps({"doubledescent": out}, separators=(",", ":"))
        + ");\n"
    )


if __name__ == "__main__":
    main()
