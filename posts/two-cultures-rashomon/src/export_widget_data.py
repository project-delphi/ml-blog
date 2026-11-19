"""Write widget-data/data.js for Part 2's subset explorer.

Run from the post directory:
    ../../.venv-two-cultures/bin/python src/export_widget_data.py
It recomputes the 142,506 ozone subset fits (under a second) from the same code
the post runs, so the widget and the post's figures show the same numbers.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("../two-cultures-breiman-2001/src").resolve()))

from twocultures import rashomon_exp  # noqa: E402

MAX_RATIO = 1.10  # keep every subset within 10% of the best training RSS


def main() -> None:
    names, fits = rashomon_exp.ozone_subsets()
    ratio = fits["rss"] / fits["rss"].min()
    keep = np.where(ratio <= MAX_RATIO)[0]
    keep = keep[np.argsort(ratio[keep])]
    edges = np.linspace(1.0, 2.0, 101)
    hist, _ = np.histogram(np.minimum(ratio, 2.0 - 1e-9), bins=edges)
    out = {
        "names": names,
        "n_subsets": int(len(ratio)),
        "best_test_mse": float(fits["test_mse"].min()),
        "hist_edges": edges.round(3).tolist(),
        "hist": hist.tolist(),
        "subsets": [
            {
                "v": fits["idx"][i].tolist(),
                "r": round(float(ratio[i]), 5),
                "t": round(float(fits["test_mse"][i]), 2),
                "b": np.round(fits["coef"][i], 3).tolist(),
            }
            for i in keep
        ],
    }
    Path("widget-data").mkdir(exist_ok=True)
    Path("widget-data/data.js").write_text(
        "window.WK_DATA = Object.assign(window.WK_DATA || {}, "
        + json.dumps({"subsets": out}, separators=(",", ":"))
        + ");\n"
    )


if __name__ == "__main__":
    main()
