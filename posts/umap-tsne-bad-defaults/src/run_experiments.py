"""Score default UMAP against a grid of its own settings, PCA and default t-SNE.

Writes ``cache/scores.parquet``: one row per dataset x seed x method x setting, plus
the same scores computed in the original space. Run from the post folder:

    ../../.venv-umap-tsne/bin/python src/run_experiments.py

MNIST, Fashion-MNIST and the Olivetti faces are downloaded once by scikit-learn
into ~/scikit_learn_data; the post itself only reads the cache.
"""

from __future__ import annotations

import functools
import itertools
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.datasets import (
    fetch_olivetti_faces,
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_wine,
    make_blobs,
)
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
CACHE = HERE / "cache"
PBMC = HERE.parent / "when-umap-lies" / "cache" / "elephant" / "pbmc_reference.npz"

SEEDS = (0, 1, 2)
K = 10
N_TRIPLETS = 20_000
N_NEIGHBORS = (5, 10, 15, 30, 50, 100)
MIN_DIST = (0.0, 0.1, 0.25, 0.5, 0.99)
SUBSAMPLE = 10_000


# --------------------------------------------------------------------------- data


@functools.cache
def _openml_full(name: str):
    d = fetch_openml(name, version=1, as_frame=False, parser="liac-arff")
    return d.data.astype(np.uint8), d.target.astype(int)


def _openml(name: str, seed: int):
    X, y = _openml_full(name)
    idx = np.random.default_rng(seed).choice(len(X), SUBSAMPLE, replace=False)
    return X[idx].astype(np.float32) / 255.0, y[idx]


def load(name: str, seed: int):
    """Return (X, y). Seed changes the draw only for subsampled or synthetic sets."""
    if name == "PBMC 3k":
        d = np.load(PBMC, allow_pickle=True)
        return d["X"].astype(np.float32), d["labels"].astype(int)
    if name == "digits":
        d = load_digits()
        return d.data.astype(np.float32), d.target
    if name == "MNIST":
        return _openml("mnist_784", seed)
    if name == "Fashion-MNIST":
        return _openml("Fashion-MNIST", seed)
    if name == "Olivetti faces":
        d = fetch_olivetti_faces()
        return d.data.astype(np.float32), d.target
    if name == "wine":
        d = load_wine()
        return StandardScaler().fit_transform(d.data).astype(np.float32), d.target
    if name == "breast cancer":
        d = load_breast_cancer()
        return StandardScaler().fit_transform(d.data).astype(np.float32), d.target
    if name == "noisy clusters":
        X, y = make_blobs(
            n_samples=3000,
            centers=5,
            n_features=5,
            cluster_std=1.0,
            random_state=1234 + seed,
        )
        noise = np.random.default_rng(seed).normal(0, 3, size=(3000, 45))
        return np.hstack([X, noise]).astype(np.float32), y
    if name == "nested clusters":
        rs = np.random.RandomState(99 + seed)
        big = rs.randn(3, 20) * 5.0
        sub = rs.randn(3, 3, 20) * 1.5
        X = np.zeros((2700, 20), dtype=np.float32)
        y = np.repeat(np.arange(9), 300)
        for c in range(9):
            X[y == c] = big[c // 3] + sub[c // 3, c % 3] + rs.normal(0, 0.5, (300, 20))
        return X, y
    raise KeyError(name)


DATASETS = (
    "PBMC 3k",
    "digits",
    "MNIST",
    "Fashion-MNIST",
    "Olivetti faces",
    "wine",
    "breast cancer",
    "noisy clusters",
    "nested clusters",
)


# ------------------------------------------------------------------------ scores


class Reference:
    """Everything about the original space that scoring a map needs."""

    def __init__(self, X, y, seed):
        self.y = y
        self.k_classes = len(np.unique(y))
        self.knn = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X)[1][:, 1:]
        rng = np.random.default_rng(seed)
        self.trip = rng.integers(0, len(X), size=(N_TRIPLETS, 3))
        self.trip_sign = np.sign(self._d(X, 0, 1) - self._d(X, 0, 2))
        cls = np.unique(y)
        self.centroid_d = pdist(np.array([X[y == c].mean(0) for c in cls]))

    def _d(self, Z, a, b):
        return np.linalg.norm(Z[self.trip[:, a]] - Z[self.trip[:, b]], axis=1)

    def score(self, Z, seed):
        y = self.y
        nz = NearestNeighbors(n_neighbors=K + 1).fit(Z).kneighbors(Z)[1][:, 1:]
        recall = np.mean([len(set(a) & set(b)) for a, b in zip(self.knn, nz)]) / K
        folds = StratifiedKFold(5, shuffle=True, random_state=seed)
        acc = cross_val_score(KNeighborsClassifier(K), Z, y, cv=folds).mean()
        km = KMeans(self.k_classes, n_init=10, random_state=seed).fit_predict(Z)
        trip = np.mean(np.sign(self._d(Z, 0, 1) - self._d(Z, 0, 2)) == self.trip_sign)
        cls = np.unique(y)
        cd = pdist(np.array([Z[y == c].mean(0) for c in cls]))
        return {
            "neighbor recall": recall,
            "kNN accuracy": acc,
            "k-means ARI": adjusted_rand_score(y, km),
            "triplet accuracy": trip,
            "centroid rho": spearmanr(self.centroid_d, cd).statistic,
        }


# ----------------------------------------------------------------------- methods


def fit(X, method, seed, **kw):
    if method == "PCA":
        return PCA(2, random_state=seed).fit_transform(X)
    if method == "t-SNE":
        from openTSNE import TSNE

        return np.asarray(TSNE(n_components=2, random_state=seed, n_jobs=1).fit(X))
    import umap

    # umap-learn 0.6 warns that its default pipeline will switch to
    # compatibility_layout=False (recursive initialisation, Adam). Pin both
    # explicitly: "UMAP" is today's default, "UMAP 0.6 layout" the announced one.
    compat = method == "UMAP"
    return umap.UMAP(
        n_components=2, random_state=seed, n_jobs=1, compatibility_layout=compat, **kw
    ).fit_transform(X)


def jobs():
    for name, seed in itertools.product(DATASETS, SEEDS):
        yield name, seed, "PCA", None, None
        yield name, seed, "t-SNE", None, None
        yield name, seed, "UMAP 0.6 layout", 15, 0.1
        for nn, md in itertools.product(N_NEIGHBORS, MIN_DIST):
            yield name, seed, "UMAP", nn, md


@functools.cache
def prepared(name, seed):
    X, y = load(name, seed)
    return X, y, Reference(X, y, seed)


def run(name, seed, method, nn, md):
    X, y, ref = prepared(name, seed)
    kw = {} if nn is None else {"n_neighbors": nn, "min_dist": md}
    t0 = time.perf_counter()
    Z = fit(X, method, seed, **kw)
    sec = time.perf_counter() - t0
    row = dict(
        dataset=name, seed=seed, method=method, n_neighbors=nn, min_dist=md, seconds=sec
    )
    row.update(ref.score(Z, seed))
    return row


def ceiling(name, seed):
    """The same scores computed in the original space, where possible."""
    X, y = load(name, seed)
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = cross_val_score(KNeighborsClassifier(K), X, y, cv=folds).mean()
    km = KMeans(len(np.unique(y)), n_init=10, random_state=seed).fit_predict(X)
    return dict(
        dataset=name,
        seed=seed,
        method="original space",
        **{"kNN accuracy": acc, "k-means ARI": adjusted_rand_score(y, km)},
    )


def main():
    # Workers must import this file as a module: lru_cache'd functions defined in
    # __main__ do not pickle, and a by-value copy would lose the per-process cache.
    os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    sys.path.insert(0, os.environ["PYTHONPATH"])
    import run_experiments as me

    CACHE.mkdir(exist_ok=True)
    only = set(sys.argv[1:])
    todo = [j for j in jobs() if not only or j[0] in only]
    # Group by dataset+seed so each worker builds a Reference once per block.
    todo.sort(key=lambda j: (j[0], j[1]))
    out = Parallel(n_jobs=9, verbose=5, batch_size=8)(delayed(me.run)(*j) for j in todo)
    rows = list(out)
    names = sorted({j[0] for j in todo})
    ceil = Parallel(n_jobs=9)(delayed(me.ceiling)(n, s) for n in names for s in SEEDS)
    df = pd.concat([pd.DataFrame(rows), pd.DataFrame(ceil)], ignore_index=True)
    suffix = "" if not only else "_" + "_".join(sorted(only)).replace(" ", "-")
    df.to_parquet(CACHE / f"scores{suffix}.parquet", index=False)
    print(df.groupby(["dataset", "method"]).size())


if __name__ == "__main__":
    main()
