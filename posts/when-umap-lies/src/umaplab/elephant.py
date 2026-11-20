"""An elephant-shaped embedding in the spirit of Chari & Pachter's Picasso, scored both ways.

Picasso (Chari & Pachter 2023) trains an autoencoder whose 2-D latent space is pulled
onto an elephant outline: each cell is matched to a distinct point of the shape by
linear sum assignment, and a linear, bias-free decoder must still reconstruct the data.
The loss is f * shape cost + (1 - f) * reconstruction cost.

We do not retrain Picasso. We build the cheapest embedding with the same two
ingredients. A linear decoder from 2-D is best served by the PCA plane, so the
reconstruction-optimal layout is 2-D PCA. The shape term is an optimal assignment of
those PCA positions to points sampled inside an elephant silhouette. Interpolating

    Y_f = (1 - f) * Y_pca + f * Y_elephant

sweeps from pure PCA (f = 0) to a pure elephant (f = 1) that keeps PCA's coarse
arrangement of the cells. A second elephant assigns cells to shape points at random
(``random``), the control that keeps no structure at all.

Every embedding is scored with both papers' metrics:

- Chari & Pachter: Pearson correlation, embedding against reference space, of the
  distances between cell-type centroids (``inter_corr``) and of the mean within-type
  pairwise distance per type (``intra_corr``).
- Lause, Berens & Kobak: leave-one-out 10-NN majority-vote accuracy (``knn_acc``),
  10-NN recall against the reference space (``knn_recall``), silhouette on the labels
  (``silhouette``), and the best AMI over HDBSCAN runs with min_cluster_size =
  min_samples in {5, 10, 15, 20, 30, 40, 50, 75, 100}, noise points assigned to the
  nearest cluster (``max_ami``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .generators import elephant_target
from .metrics import knn_indices, knn_preservation

F_GRID = tuple(np.round(np.linspace(0.0, 1.0, 11), 2))
HDBSCAN_SIZES = (5, 10, 15, 20, 30, 40, 50, 75, 100)


def _standardise(Y: np.ndarray) -> np.ndarray:
    """Centre and scale to unit total variance (so shapes are comparable in size)."""
    Y = Y - Y.mean(0)
    return Y / np.sqrt((Y**2).sum(1).mean())


def elephant_layouts(X: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """2-D PCA, the PCA-to-elephant optimal assignment, and the random assignment."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    pca = _standardise(PCA(2, random_state=seed).fit_transform(X))
    shape = _standardise(elephant_target(n, seed).X)
    # The elephant stays upright; the assignment minimises the total squared distance
    # each cell moves from its PCA position to its elephant point.
    _, perm = linear_sum_assignment(cdist(pca, shape, "sqeuclidean"))
    matched = shape[perm]
    random = shape[rng.permutation(n)]
    return {"pca": pca, "elephant": matched, "random": random}


def _centroid_dists(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cents = np.stack([Z[labels == c].mean(0) for c in np.unique(labels)])
    return pdist(cents)


def _intra(Z: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) > 300:  # mean pairwise distance on a fixed-size subsample
            idx = rng.choice(idx, 300, replace=False)
        out.append(pdist(Z[idx]).mean() if len(idx) > 1 else 0.0)
    return np.array(out)


def _loo_knn_accuracy(Z: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    idx = knn_indices(Z, k)
    votes = labels[idx]
    pred = np.array(
        [np.bincount(v, minlength=labels.max() + 1).argmax() for v in votes]
    )
    return float((pred == labels).mean())


def _max_ami(Z: np.ndarray, labels: np.ndarray) -> float:
    best = -1.0
    for m in HDBSCAN_SIZES:
        lab = HDBSCAN(min_cluster_size=m, min_samples=m).fit_predict(Z)
        if (lab >= 0).sum() == 0:
            continue
        noise = lab < 0
        if noise.any():  # noise to the nearest clustered point's cluster
            nn = NearestNeighbors(n_neighbors=1).fit(Z[~noise])
            lab = lab.copy()
            lab[noise] = lab[~noise][nn.kneighbors(Z[noise])[1][:, 0]]
        best = max(best, adjusted_mutual_info_score(labels, lab))
    return float(best)


def score(
    ref: np.ndarray, Y: np.ndarray, labels: np.ndarray, seed: int = 0
) -> dict[str, float]:
    """Both papers' metrics for embedding Y of reference space ref."""
    rng = np.random.default_rng(seed)
    return {
        "inter_corr": float(
            np.corrcoef(_centroid_dists(ref, labels), _centroid_dists(Y, labels))[0, 1]
        ),
        "intra_corr": float(
            np.corrcoef(
                _intra(ref, labels, np.random.default_rng(seed)), _intra(Y, labels, rng)
            )[0, 1]
        ),
        "knn_acc": _loo_knn_accuracy(Y, labels),
        "knn_recall": knn_preservation(ref, Y, [10])[10],
        "silhouette": float(silhouette_score(Y, labels)),
        "max_ami": _max_ami(Y, labels),
    }


def run(
    name: str,
    X: np.ndarray,
    labels: np.ndarray,
    others: dict[str, np.ndarray],
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Score the f-sweep, the random elephant, and any other embeddings given.

    ``others`` maps a label (e.g. "umap·s0") to an embedding of X. Returns long-format
    rows {dataset, embedding, f, metric, value} and the layouts to keep.
    """
    lay = elephant_layouts(X, seed)
    rows, keep = [], {}
    for f in F_GRID:
        Y = (1 - f) * lay["pca"] + f * lay["elephant"]
        keep[f"elephant·f{f:.1f}"] = Y
        for m, v in score(X, Y, labels, seed).items():
            rows.append(
                dict(dataset=name, embedding="elephant", f=f, metric=m, value=v)
            )
    keep["random_elephant"] = lay["random"]
    for m, v in score(X, lay["random"], labels, seed).items():
        rows.append(dict(dataset=name, embedding="random", f=1.0, metric=m, value=v))
    for label, Y in others.items():
        keep[label] = _standardise(Y)
        for m, v in score(X, Y, labels, seed).items():
            rows.append(
                dict(dataset=name, embedding=label, f=np.nan, metric=m, value=v)
            )
    return pd.DataFrame(rows), keep
