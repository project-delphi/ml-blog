"""Fill this post's cache/ ahead of a render (run from the post directory)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from twocultures import cache, gof  # noqa: E402

if __name__ == "__main__":
    df = cache.frame("gof_v1", lambda: gof.run(reps=200))
    print(df.groupby(["shape", "eta"])[["p_reset", "mse_rf", "mse_ols"]].median())
