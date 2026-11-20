"""Read cached results for the posts: one tidy frame per experiment, and summaries.

The posts never run an experiment; they call these functions on ``cache/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / "cache"
RUN_COLS = [
    "cell_id",
    "experiment",
    "generator",
    "gen_params",
    "transforms",
    "D",
    "intrinsic_dim",
    "noise_sigma",
    "n",
    "data_seed",
    "method",
    "method_params",
    "config",
    "algo_seed",
    "seconds",
    "status",
]


def runs(exp: str) -> pd.DataFrame:
    """runs.parquet of one experiment."""
    return pd.read_parquet(CACHE / exp / "runs.parquet")


def load(exp: str, ok_only: bool = True) -> pd.DataFrame:
    """Metric rows joined to their run, one row per (cell, metric, k, reference)."""
    r = runs(exp)
    if ok_only:
        r = r[r.status == "ok"]
    m = pd.read_parquet(CACHE / exp / "metrics.parquet")
    cols = [c for c in RUN_COLS if c in r.columns]
    df = m.merge(r[cols], on="cell_id", how="inner")
    df["gp"] = df.gen_params.map(json.loads)
    return df


def pick(
    df: pd.DataFrame, metric: str, k: float | None = None, reference: str | None = None
) -> pd.DataFrame:
    """Rows of one metric (and k, reference when given)."""
    out = df[df.metric == metric]
    if k is not None:
        out = out[out.k == k]
    if reference is not None:
        out = out[out.reference == reference]
    return out


def summary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Median, min, max, mean and count of ``value`` over the remaining columns."""
    g = df.groupby(by, dropna=False)["value"]
    return g.agg(
        median="median", lo="min", hi="max", mean="mean", n="count"
    ).reset_index()


def paired_diff(
    df: pd.DataFrame, a: str, b: str, by: list[str], unit: str = "data_seed"
) -> pd.DataFrame:
    """Per-unit difference config a minus config b (paired on the same dataset draw)."""
    keys = by + [unit]
    wide = df[df.config.isin([a, b])].pivot_table(
        index=keys, columns="config", values="value", aggfunc="mean"
    )
    wide = wide.dropna()
    wide["diff"] = wide[a] - wide[b]
    return wide.reset_index()


def embeddings(exp: str) -> dict[str, np.ndarray]:
    """The kept embeddings of an experiment, keyed by cell_id, rescaled to float32."""
    path = CACHE / exp / "embeddings.npz"
    if not path.exists():
        return {}
    r = runs(exp).set_index("cell_id")
    out = {}
    with np.load(path) as z:
        for key in z.files:
            Y = z[key].astype(np.float32)
            if key in r.index and "info" in r.columns:
                info = json.loads(r.at[key, "info"] or "{}")
                Y = Y / float(info.get("embedding_scale", 1.0))  # stored as Y * scale
            out[key] = Y
    return out


def fmt(x: float, d: int = 2) -> str:
    """A number for prose, with a true minus sign."""
    s = f"{x:.{d}f}"
    if float(s) == 0:  # no "-0.00"
        s = s.lstrip("-")
    return s.replace("-", "−")


def rng(lo: float, hi: float, d: int = 2) -> str:
    """'lo–hi' for prose; a single number when they round equal."""
    a, b = fmt(lo, d), fmt(hi, d)
    if a == b:
        return a
    return f"{a} to {b}" if "−" in a + b else f"{a}–{b}"
