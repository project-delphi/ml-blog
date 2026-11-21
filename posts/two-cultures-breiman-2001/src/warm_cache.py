"""Fill this post's cache/ ahead of a render (run from the post directory).

The Chronos-2 forecasts (cache/chronos2_v3.parquet, cache/chronos2_delhi_v4.parquet)
are not filled here: they need torch, which this venv lacks. See src/chronos_forecast.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from twocultures import cache, delhi, gof, modern, traffic  # noqa: E402
from twocultures import delhi_eval as E  # noqa: E402

if __name__ == "__main__":
    df = cache.frame("gof_v1", lambda: gof.run(reps=200))
    print(df.groupby(["shape", "eta"])[["p_reset", "mse_rf", "mse_ols"]].median())
    mod = cache.frame(modern.CACHE_TAG, modern.run)
    print(mod.groupby(["model", "split"]).size())
    dmod = cache.frame(delhi.CACHE_TAG, delhi.run)
    print(dmod.groupby(["model", "split"]).size())
    # The Delhi and I-94 comparisons: about 45 minutes from cold on ten cores.
    for fn in (
        E.zoo,
        E.learning_curve,
        E.cv_compare,
        E.rolling,
        E.designs,
        traffic.run,
        traffic.learning_curve,
        traffic.probe,
    ):
        print(fn.__module__, fn.__name__, len(fn()))
