"""Fill this post's cache/ ahead of a render (run from the post directory).

The Chronos-2 forecasts (cache/chronos2_v1.parquet) are not filled here: they
need torch, which this venv lacks. See src/chronos_forecast.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from twocultures import cache, gof, modern  # noqa: E402

if __name__ == "__main__":
    df = cache.frame("gof_v1", lambda: gof.run(reps=200))
    print(df.groupby(["shape", "eta"])[["p_reset", "mse_rf", "mse_ols"]].median())
    mod = cache.frame(modern.CACHE_TAG, modern.run)
    print(mod.groupby(["model", "split"]).size())
