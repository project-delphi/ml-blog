"""Does default UMAP fail when classes hold fewer points than n_neighbors?

Subsamples scikit-learn's digits to m images per class and scores UMAP maps at
n_neighbors 5 and 15 against the original pixels, on k-means ARI and on 5-fold
3-NN class accuracy. The main results vote with 10 neighbours; at 8 per class a
training fold holds only about 6 points per class, which would cap a 10-NN vote,
so this test uses 3. Writes cache/class_size.parquet.

    ../../.venv-umap-tsne/bin/python src/class_size.py
"""

from __future__ import annotations

import itertools
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

CACHE = Path(__file__).resolve().parent.parent / "cache"
PER_CLASS = (8, 10, 12, 15, 20, 40, 80, 160)
N_NEIGHBORS = (5, 15)
SEEDS = (0, 1, 2, 3, 4)


def one(m, seed, nn):
    import umap

    d = load_digits()
    rng = np.random.default_rng(seed)
    idx = np.concatenate(
        [rng.choice(np.flatnonzero(d.target == c), m, replace=False) for c in range(10)]
    )
    X, y = d.data[idx].astype(np.float32), d.target[idx]
    if nn == "original":
        Z = X
    else:
        Z = umap.UMAP(
            n_neighbors=nn, random_state=seed, n_jobs=1, compatibility_layout=True
        ).fit_transform(X)
    km = KMeans(10, n_init=10, random_state=seed).fit_predict(Z)
    return dict(
        per_class=m,
        seed=seed,
        n_neighbors=str(nn),
        ari=adjusted_rand_score(y, km),
        knn_accuracy=cross_val_score(
            KNeighborsClassifier(3),
            Z,
            y,
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        ).mean(),
    )


def main():
    grid = itertools.product(PER_CLASS, SEEDS, (*N_NEIGHBORS, "original"))
    rows = Parallel(n_jobs=9)(delayed(one)(*g) for g in grid)
    df = pd.DataFrame(rows)
    CACHE.mkdir(exist_ok=True)
    df.to_parquet(CACHE / "class_size.parquet", index=False)
    print(
        df.groupby(["per_class", "n_neighbors"])[["ari", "knn_accuracy"]]
        .mean()
        .unstack()
        .round(3)
    )


if __name__ == "__main__":
    main()
