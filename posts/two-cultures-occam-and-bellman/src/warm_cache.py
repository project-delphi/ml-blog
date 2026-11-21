"""Fill this post's cache/ ahead of a render (run from the post directory).

The post's cells call the same functions through `twocultures.cache`, so a render
with a warm cache reads these files and a render without one recomputes them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../two-cultures-breiman-2001/src").resolve()))

import pandas as pd  # noqa: E402
from twocultures import bellman, cache, occam  # noqa: E402

if __name__ == "__main__":
    which = sys.argv[1:] or ["small", "large", "svm", "rf", "credit"]
    if "large" in which:
        print(cache.frame("table2_large_v1", occam.large).to_string())
    if "small" in which:
        df = cache.frame("table2_small_v1", lambda: occam.small(reps=100))
        print(df.groupby("dataset")[["tree", "forest"]].mean())
    if "svm" in which:
        print(cache.frame("svm_v1", bellman.svm_degrees).to_string())
    if "rf" in which:
        a = cache.frame("randfeat_v1", lambda: bellman.random_features())
        b = cache.frame(
            "randfeat_ridge_v1", lambda: bellman.random_features(ridge=1e-1)
        )
        print(
            pd.concat([a, b])
            .groupby(["ridge", "width"])["test_error"]
            .mean()
            .to_string()
        )
    if "credit" in which:

        def run():
            return pd.concat(
                [bellman.credit_models(seed=s)[0].assign(seed=s) for s in (0, 1, 2)]
            )

        print(cache.frame("credit_v1", run).to_string())
