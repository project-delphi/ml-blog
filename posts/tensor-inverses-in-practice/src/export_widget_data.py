r"""Write the JSON the browser widget reads.

Every state the widget can show is precomputed here. The browser only looks
values up and draws them -- there is no linear algebra in the page. Because
this imports the same `tinv` module the post's cells import, the widget's
numbers and the post's numbers are the same computation.

Usage:
    .venv-tensor-factorizations/bin/python \\
        posts/tensor-inverses-in-practice/src/export_widget_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tinv as T  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "widget-data"
BUDGET_BYTES: Final[int] = 1_024_000

LEVELS: Final = [round(0.1 * i, 1) for i in range(11)]
SOLVERS: Final = ["inv", "pinv"]
BENT_SLICE: Final = 2


def strip(block: np.ndarray) -> dict:
    """One strip of frontal slices, scaled to [-1, 1] with its scale kept aside.

    Entries of the inverse run to 1e11 at the top of the conditioning sweep, so
    shipping raw values would cost far more characters than shipping a pattern
    plus one number. The widget prints the scale under each strip.
    """
    scale = float(np.abs(block).max())
    unit = block / scale if scale else block
    return {
        "scale": scale,
        "slices": [
            [round(float(v), 4) for v in unit[:, :, k].ravel()]
            for k in range(block.shape[2])
        ],
    }


def main() -> int:
    payload: dict = {
        "n": T.TOY_N,
        "slices": T.TOY_SLICES,
        "levels": LEVELS,
        "solvers": SOLVERS,
        "bent_slice": BENT_SLICE,
        "seed": T.SEED,
        "rcond": 1e-8,
        "by_level": [],
        # Indexed by solver, then by position in LEVELS. Not keyed by the
        # float itself: "0.0" from Python and 0 from JavaScript do not match.
        "by_state": {solver: [] for solver in SOLVERS},
    }

    for level in LEVELS:
        state = T.widget_state(level, "inv", BENT_SLICE)
        payload["by_level"].append(
            {
                "level": level,
                "spatial": strip(state["spatial"]),
                "fourier": strip(state["fourier"]),
                "conds": [float(f"{c:.6g}") for c in state["conds"]],
            },
        )

    for solver in SOLVERS:
        for level in LEVELS:
            state = T.widget_state(level, solver, BENT_SLICE)
            payload["by_state"][solver].append(
                {
                    "finverse": strip(state["finverse"]),
                    "inverse": strip(state["inverse"]),
                    "product": strip(state["product"]),
                    "residual": float(f"{state['residual']:.6g}"),
                    "max_abs": float(f"{state['max_abs']:.6g}"),
                    "solve_error": float(f"{state['solve_error']:.6g}"),
                },
            )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tinv.json"
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    size = path.stat().st_size
    print(f"wrote {path} ({size / 1024:.0f} kB)")
    if size > BUDGET_BYTES:
        raise ValueError(f"widget data is {size / 1024:.0f} kB, over budget")

    # Spot checks, so the numbers in the prose can be eyeballed against these.
    for level in (0.0, 0.5, 1.0):
        for solver in SOLVERS:
            i = LEVELS.index(level)
            s = payload["by_state"][solver][i]
            kappa = payload["by_level"][i]["conds"][BENT_SLICE]
            print(
                f"  level {level:.1f} {solver:>4}  kappa {kappa:9.2e}"
                f"  residual {s['residual']:9.2e}"
                f"  max|X| {s['max_abs']:9.2e}"
                f"  solve {s['solve_error']:9.2e}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
