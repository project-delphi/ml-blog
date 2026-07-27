"""Write the small JSON files the Observable JS widgets read.

Design rule: ship *data*, never replicates.  The browser does its own resampling,
so nothing here scales with ``B``.  The whole directory stays well under 1 MB.

Outputs, all committed to the repository:

``returns.json``  the n = 59 head-to-head window plus the long and short S&P windows
``toy.json``      the n = 5 toy sample, the exact 126-pattern Efron enumeration,
                  and the exact laws of mean / median / max / min / mode
``mnist.json``    run-length-encoded per-example correctness, per-class counts,
                  and the confusion matrix
``ladder.json``   the three denominators and sqrt(n/(n+1)) over a range of n
"""

from __future__ import annotations

import itertools
import json
from math import comb, factorial
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import data as dt

_HERE: Final[Path] = Path(__file__).resolve().parent.parent
OUT: Final[Path] = _HERE / "widget-data"

# The toy sample used throughout the post's exact-enumeration sections.
TOY: Final[list[float]] = [1.0, 2.0, 3.0, 4.0, 10.0]
# The three-point sample used by the simplex widgets.
TRIPLE: Final[list[float]] = [1.0, 3.0, 10.0]


def _round(values: NDArray[np.float64], places: int = 4) -> list[float]:
    """Round an array to keep the JSON small."""
    return [round(float(v), places) for v in values]


def export_returns(returns: dt.Returns) -> dict[str, object]:
    """Both series over the head-to-head window, plus the two S&P windows."""
    lo, hi = pd.Period(dt.HEAD_START), pd.Period(dt.HEAD_END)
    sp = returns.sp500[(returns.sp500.index >= lo) & (returns.sp500.index <= hi)]
    ib = returns.ibm[(returns.ibm.index >= lo) & (returns.ibm.index <= hi)]
    long_series = returns.sp500[returns.sp500.index >= pd.Period(dt.LONG_START)]
    short_series = returns.sp500.iloc[-dt.SHORT_N :]

    def pack(series: pd.Series) -> dict[str, object]:
        return {
            "dates": [str(p) for p in series.index],
            "returns": _round(series.to_numpy(np.float64)),
        }

    return {
        "sp500": pack(sp),
        "ibm": pack(ib),
        "sp500_long": pack(long_series),
        "sp500_short": pack(short_series),
        "retrieved": returns.retrieved,
        "labels": returns.window_labels(),
    }


def _efron_patterns(n: int) -> tuple[list[tuple[int, ...]], NDArray[np.float64]]:
    """Every multiplicity pattern for a size-``n`` resample, with its probability.

    Stars and bars gives ``comb(2n - 1, n - 1)`` patterns; each has multinomial
    probability ``n! / (prod c_i!) * n**(-n)``.
    """
    patterns = [p for p in itertools.product(range(n + 1), repeat=n) if sum(p) == n]
    probs = np.array(
        [
            factorial(n) / np.prod([float(factorial(c)) for c in p]) / n**n
            for p in patterns
        ]
    )
    return patterns, probs


def export_toy() -> dict[str, object]:
    """The n = 5 toy sample with exact laws for both methods."""
    x = np.array(TOY)
    n = x.size
    patterns, probs = _efron_patterns(n)
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError(f"Efron pattern probabilities sum to {probs.sum()}, not 1")

    weights = np.array(patterns, dtype=np.float64) / n

    # Efron exact laws, by enumeration.
    means = weights @ x
    cumulative = np.cumsum(weights, axis=1)
    med_idx = (cumulative < 0.5).sum(axis=1)
    max_idx = n - 1 - np.argmax(weights[:, ::-1] > 0.0, axis=1)
    min_idx = np.argmax(weights > 0.0, axis=1)

    def law(index: NDArray[np.int64]) -> list[float]:
        out = np.zeros(n)
        np.add.at(out, index, probs)
        return _round(out, 6)

    # Bayesian exact laws.
    bayes_median = [comb(n - 1, j) * 2.0 ** (-(n - 1)) for j in range(n)]
    bayes_max = [0.0] * (n - 1) + [1.0]
    bayes_min = [1.0] + [0.0] * (n - 1)
    bayes_mode = [1.0 / n] * n

    return {
        "x": TOY,
        "triple": TRIPLE,
        "n": n,
        "n_patterns": len(patterns),
        "patterns": [list(p) for p in patterns],
        "pattern_probs": _round(probs, 8),
        "efron": {
            "mean_values": _round(means, 6),
            "median": law(med_idx),
            "max": law(max_idx),
            "min": law(min_idx),
        },
        "bayes": {
            "median": _round(np.array(bayes_median), 6),
            "max": bayes_max,
            "min": bayes_min,
            "mode": _round(np.array(bayes_mode), 6),
        },
        "ladder": {
            "s2": round(float(np.mean((x - x.mean()) ** 2)), 6),
            "bayes": round(float(np.mean((x - x.mean()) ** 2) / (n + 1)), 6),
            "efron": round(float(np.mean((x - x.mean()) ** 2) / n), 6),
            "classical": round(float(np.mean((x - x.mean()) ** 2) / (n - 1)), 6),
        },
    }


def export_mnist() -> dict[str, object]:
    """Per-class counts and the confusion matrix.

    The per-example correctness vector is deliberately *not* shipped: every
    metric the widgets compute is a function of the per-class counts, so the
    10,000 individual outcomes would be bytes nothing reads.
    """
    import mnist_experiment as mx

    cache = mx.train_and_cache()
    confusion = cache.confusion()
    per_class = {
        str(c): {
            "support": int((cache.y_true == c).sum()),
            "correct": int(((cache.y_true == c) & cache.correct).sum()),
            "predicted": int((cache.y_pred == c).sum()),
        }
        for c in range(10)
    }
    return {
        "n": cache.n,
        "k": cache.k,
        "accuracy": round(cache.accuracy, 6),
        "backend": cache.backend,
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def export_ladder(max_n: int = 500) -> dict[str, object]:
    """The three denominators and the sd ratio over a range of ``n``."""
    ns = np.arange(2, max_n + 1)
    return {
        "n": ns.tolist(),
        "ratio": _round(np.sqrt(ns / (ns + 1.0)), 6),
        "marked": {"short": 12, "head": 59, "long": 437},
    }


def main() -> None:
    """Write every widget data file and report the total size."""
    OUT.mkdir(parents=True, exist_ok=True)
    returns = dt.load_returns()
    payloads = {
        "returns.json": export_returns(returns),
        "toy.json": export_toy(),
        "mnist.json": export_mnist(),
        "ladder.json": export_ladder(),
    }
    total = 0
    for name, payload in payloads.items():
        path = OUT / name
        path.write_text(json.dumps(payload, separators=(",", ":")))
        size = path.stat().st_size
        total += size
        print(f"  {name:16s} {size / 1024:8.1f} KB")
    print(f"  {'total':16s} {total / 1024:8.1f} KB  (budget 1024 KB)")
    if total > 1_024_000:
        raise ValueError(f"widget data is {total / 1024:.0f} KB, over the 1 MB budget")


if __name__ == "__main__":
    main()
