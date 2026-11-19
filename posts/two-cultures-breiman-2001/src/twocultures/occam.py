"""Breiman's Table 2: a pruned single tree against a random forest."""

import time

import pandas as pd

from . import data, models

FOREST_TREES = 500


def small(reps: int = 100, seed: int = 0) -> pd.DataFrame:
    """Repeated 10% holdouts on the five small sets; one row per (set, rep)."""
    frames = []
    for name, (X, y) in data.table2_small().items():
        t = time.time()
        res = models.repeated_holdout(
            {
                "tree": lambda r: models.tree(seed=r),
                "forest": lambda r: models.forest(seed=r, n_estimators=FOREST_TREES),
            },
            X,
            y,
            reps=reps,
            seed=seed,
        )
        df = pd.DataFrame(res)
        df["dataset"] = name
        df["seconds"] = time.time() - t
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def large(seed: int = 0) -> pd.DataFrame:
    """Fixed train/test sets; one row per set."""
    rows = []
    for name, (Xtr, ytr, Xte, yte) in data.table2_large().items():
        t = time.time()
        e_tree = models.test_error(models.tree(seed=seed), Xtr, ytr, Xte, yte)
        e_forest = models.test_error(
            models.forest(seed=seed, n_estimators=FOREST_TREES, n_jobs=-1),
            Xtr,
            ytr,
            Xte,
            yte,
        )
        rows.append(
            {
                "dataset": name,
                "tree": e_tree,
                "forest": e_forest,
                "n_test": len(yte),
                "seconds": time.time() - t,
            }
        )
    return pd.DataFrame(rows)
