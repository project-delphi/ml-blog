"""Write widget-data/data.js for Part 4's importance widget.

Run from the post directory after the post's cells have filled cache/
(`importance_reps_v2.parquet`):
    ../../.venv-two-cultures/bin/python src/export_widget_data.py
"""

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    df = pd.read_parquet("cache/importance_reps_v2.parquet")
    df = df[df["j"] >= 0]
    names = df[df["boot"] == 0].sort_values("j")["var"].tolist()
    out = {"names": names, "boots": int(df["boot"].max()), "perm": [], "mdi": []}
    for b in range(out["boots"] + 1):
        d = df[df["boot"] == b].sort_values("j")
        out["perm"].append([round(float(v), 2) for v in d["perm"]])
        out["mdi"].append([round(float(v), 4) for v in d["mdi"]])
    Path("widget-data").mkdir(exist_ok=True)
    Path("widget-data/data.js").write_text(
        "window.WK_DATA = Object.assign(window.WK_DATA || {}, "
        + json.dumps({"importance": out}, separators=(",", ":"))
        + ");\n"
    )


if __name__ == "__main__":
    main()
