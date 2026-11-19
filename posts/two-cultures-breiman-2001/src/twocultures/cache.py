"""A small disk cache for the expensive experiments.

A cached result is a file under the calling post's `cache/` directory. The file
name carries a version tag chosen by the caller, so changing an experiment means
changing its tag, which leaves the stale file unread. Delete `cache/` to force a
full recomputation.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

CACHE = Path("cache")


def frame(name: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return the cached DataFrame `name`, computing and storing it if absent."""
    path = CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    CACHE.mkdir(exist_ok=True)
    df = fn()
    df.to_parquet(path, index=False)
    return df


def obj(name: str, fn: Callable[[], dict]) -> dict:
    """Return the cached JSON-serialisable dict `name`, computing it if absent."""
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    CACHE.mkdir(exist_ok=True)
    out = fn()
    path.write_text(json.dumps(out, indent=1))
    return out
