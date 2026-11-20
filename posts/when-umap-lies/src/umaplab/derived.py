"""Analyses computed from the datasets of an experiment, not from its embeddings.

``graph_fidelity`` asks how faithful the neighbour graph handed to a method is, before
any embedding is drawn: kNN preservation against the truth of the kNN graph built in
(a) the raw input, and (b) the input reduced to its first ``pca_dims`` principal
components, the standard single-cell preprocessing step.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .benchmark import build_dataset
from .metrics import knn_preservation, local_radius
from .report import CACHE, runs


def snn_neighbours(X: np.ndarray, K: int, k: int = 10) -> np.ndarray:
    """K neighbours of each point re-ranked by shared neighbours within its K-NN list.

    A same-information baseline for a map built from a K-neighbour graph: no layout,
    only pooling. Candidates are each point's K nearest neighbours; their score is the
    number of K-NN lists they share with the point (ties broken by original distance
    rank); the top k are returned. Self excluded throughout.
    """
    import scipy.sparse as sp

    from .metrics import knn_indices

    n = X.shape[0]
    idx = knn_indices(X, K)  # (n, K), nearest first
    rows = np.repeat(np.arange(n), K + 1)
    cols = np.concatenate([np.arange(n)[:, None], idx], axis=1).ravel()
    A = sp.csr_matrix(
        (np.ones_like(rows, dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    S = (A @ A.T).tocsr()
    out = np.empty((n, k), dtype=int)
    for i in range(n):
        cand = idx[i]
        shared = np.asarray(S[i, cand].todense()).ravel()
        order = np.lexsort((np.arange(K), -shared))  # most shared first, then nearer
        out[i] = cand[order[:k]]
    return out


def graph_fidelity(
    exp: str, ks=(10, 30), pca_dims=(2, 10, 50), snn=(15, 90)
) -> pd.DataFrame:
    """One row per (dataset draw, space, k) with Q_NX(k) against the truth.

    Spaces: the raw input, its first ``pca_dims`` principal components, and the
    shared-neighbour re-ranking of the input's ``snn`` neighbour lists (k = 10 only).
    """
    from .benchmark import Cell
    from .metrics import knn_indices

    r = runs(exp)
    r = r[(r.method == "input")] if "input" in set(r.method) else r
    keys = r.drop_duplicates(
        ["generator", "gen_params", "transforms", "n", "data_seed"]
    )
    rows = []
    for _, row in keys.iterrows():
        cell = Cell(
            experiment=exp,
            generator=row.generator,
            gen_params=json.loads(row.gen_params),
            transforms=tuple(json.loads(row.transforms)),
            n=int(row.n),
            data_seed=int(row.data_seed),
            method="pca",
            method_params={},
            algo_seed=0,
        )
        ds = build_dataset(cell)
        base = dict(
            generator=row.generator,
            gen_params=row.gen_params,
            transforms=row.transforms,
            D=int(ds.X.shape[1]),
            noise_sigma=row.noise_sigma,
            data_seed=int(row.data_seed),
        )
        spaces = {"input": ds.X}
        for p in pca_dims:
            if p < ds.X.shape[1]:
                spaces[f"pca{p}"] = PCA(p, random_state=0).fit_transform(ds.X)
        for name, Z in spaces.items():
            for k, q in knn_preservation(ds.latent, Z, list(ks)).items():
                rows.append({**base, "space": name, "k": k, "value": q})
        if snn:
            truth10 = knn_indices(ds.latent, 10)
            for K in snn:
                nb = snn_neighbours(ds.X, K, 10)
                q = float(
                    np.mean([len(np.intersect1d(a, b)) for a, b in zip(nb, truth10)])
                    / 10
                )
                rows.append({**base, "space": f"snn{K}", "k": 10, "value": q})
        rk = float(np.median(local_radius(ds.clean, 10)) ** 2)
        rows.append({**base, "space": "clean_r10_sq", "k": 10, "value": rk})
    return pd.DataFrame(rows)


def valley_ratio(Z: np.ndarray, labels: np.ndarray, grid: int = 400) -> float:
    """Depth of the density dip between two classes, along the axis joining them.

    Project Z onto the unit vector between the two class centroids, standardise, and
    estimate the 1-D density with a Gaussian KDE (Scott's bandwidth). Find the highest
    density on each side of the centroids' midpoint (the two modes); the ratio is the
    lowest density between those two modes over the lower of them. A unimodal
    projection has both "modes" at its one peak, so it scores 1; an empty gap between
    the classes scores 0.
    """
    from scipy.stats import gaussian_kde

    c0, c1 = Z[labels == 0].mean(0), Z[labels == 1].mean(0)
    u = (c1 - c0) / np.linalg.norm(c1 - c0)
    p = (Z - (c0 + c1) / 2) @ u
    p = p / p.std()
    xs = np.linspace(p.min(), p.max(), grid)
    f = gaussian_kde(p)(xs)
    left = np.flatnonzero(xs <= 0)
    right = np.flatnonzero(xs >= 0)
    i = left[np.argmax(f[left])]
    j = right[np.argmax(f[right])]
    if j - i < 2:
        return 1.0
    return float(f[i : j + 1].min() / min(f[i], f[j]))


def _gap_job(args):
    from sklearn.cluster import HDBSCAN

    from .embeddings import embed, set_thread_env
    from .generators import overlapping

    set_thread_env(1)
    delta, seed, method, params = args
    ds = overlapping(2000, seed, dim=10, delta=delta)
    out = []
    mcs = max(10, ds.X.shape[0] // 100)
    if method == "input":
        Z = ds.X
    else:
        Z = embed(method, ds.X, seed, params).Y.astype(float)
    from .benchmark import config_label

    cfg = "input" if method == "input" else config_label(method, params)
    base = dict(delta=delta, data_seed=seed, config=cfg)
    out.append({**base, "metric": "valley_ratio", "value": valley_ratio(Z, ds.labels)})
    # HDBSCAN twice: scikit-learn's default (allow_single_cluster=False, which can never
    # answer "one cluster") and allow_single_cluster=True. Prefix "hdbscan" / "hdbscan1".
    for prefix, single in (("hdbscan", False), ("hdbscan1", True)):
        lab = HDBSCAN(min_cluster_size=mcs, allow_single_cluster=single).fit_predict(Z)
        sizes = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.array([0])
        n = len(lab)
        rows = {
            "clusters": float(lab.max() + 1),
            "noise": float(np.mean(lab < 0)),
            "largest": float(sizes.max() / n),
            # points in clusters other than the largest, as a share of all points
            "fragments": float((sizes.sum() - sizes.max()) / n),
        }
        for k, v in rows.items():
            out.append({**base, "metric": f"{prefix}_{k}", "value": v})
    return out


def overlap_gaps(n_workers: int = 3) -> pd.DataFrame:
    """Valley ratio and HDBSCAN cluster counts, in the data and on five maps, 5 draws."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from .experiments import OVERLAP_DELTA, SEEDS5

    methods = [("input", {}), ("pca", {}), ("umap", {}), ("tsne", {})]
    methods += [("pacmap", {}), ("trimap", {}), ("umap", {"n_neighbors": 5})]
    methods += [("tsne", {"perplexity": 5})]
    jobs = [(d, s, m, p) for d in OVERLAP_DELTA for s in SEEDS5 for m, p in methods]
    ctx = get_context("spawn")
    with ProcessPoolExecutor(n_workers, mp_context=ctx) as pool:
        rows = [r for out in pool.map(_gap_job, jobs) for r in out]
    return pd.DataFrame(rows)


def seed_pairs(exp: str = "stability") -> pd.DataFrame:
    """Compare every pair of method seeds on the same data draw, after alignment.

    For each (generator, config, data_seed), the ten pairs among five method seeds:
    Procrustes disparity (after translation, scale, rotation and reflection), the
    share of each point's 10 / 30 map neighbours shared by the two maps, and the
    Spearman correlation of sampled map distances. Rotation and reflection cannot
    change the last two, so they measure change that alignment cannot remove.
    """
    from itertools import combinations

    from .report import embeddings
    from .stability import seed_pair_rows

    r = runs(exp)
    E = embeddings(exp)
    r = r[r.cell_id.isin(E.keys())]
    rows = []
    for (gen, gp, cfg, ds), sub in r.groupby(
        ["generator", "gen_params", "config", "data_seed"]
    ):
        by_seed = dict(zip(sub.algo_seed, sub.cell_id))
        for a, b in combinations(sorted(by_seed), 2):
            for row in seed_pair_rows(E[by_seed[a]], E[by_seed[b]], ks=(10, 30)):
                rows.append(
                    dict(
                        generator=gen,
                        gen_params=gp,
                        config=cfg,
                        data_seed=int(ds),
                        seed_a=int(a),
                        seed_b=int(b),
                        **row,
                    )
                )
    return pd.DataFrame(rows)


def run_all() -> None:
    """Write cache/<exp>/graph_fidelity.parquet for the noise experiments."""
    for exp in ("noise_dims", "noise_sigma", "isometry"):
        if (CACHE / exp / "runs.parquet").exists():
            graph_fidelity(exp).to_parquet(
                CACHE / exp / "graph_fidelity.parquet", index=False
            )
    overlap_gaps().to_parquet(CACHE / "overlap" / "gaps.parquet", index=False)
