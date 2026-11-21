"""Local, global, density, cluster and graph metrics for a 2-D embedding.

Conventions shared by every function here:

- **Neighbours are exact Euclidean kNN with the point itself excluded by index**, not by
  distance (sklearn ``NearestNeighbors.kneighbors()`` with no query). An exact duplicate
  of point i is therefore a legitimate neighbour of i at distance 0, and i never appears
  in its own list even when more than k points coincide with it. Ties at the k-th
  distance are broken by the search backend: deterministic, but arbitrary.
- **Ranks** (trustworthiness, continuity) are 1-based and self excluded. A point tied in
  distance with others takes the smallest rank of its tie group, so a tie at the
  neighbourhood boundary is never counted as an intrusion; on tie-free data this is
  exactly the Venna & Kaski rank and matches ``sklearn.manifold.trustworthiness``.
  Ranks are computed in row chunks, so memory stays bounded (tens of MB per chunk)
  however large n is; no n x n matrix is ever held.
- Every function takes float data and works in float64 internally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import pearsonr, spearmanr

if TYPE_CHECKING:  # generators.py imports nothing from here; keep it that way
    from .generators import Dataset

GRAPH_K = 15  # k of the kNN graphs behind graph_jaccard and n_components
DENSITY_K = 15  # k of the local radius behind density_spearman / true_density_spearman
SILHOUETTE_MAX_N = 3000  # silhouette is O(n^2); larger inputs are subsampled
_CHUNK_BYTES = 32 * 2**20  # per-array budget for one row chunk of distances
_BRUTE_DIM = 20  # above this ambient dim, brute-force kNN and Gram-trick distances
# Rank route switch: sort whole rows once the share of distances that would be
# bucketed, summed over the index sets, exceeds this (measured crossover ~0.6-0.8).
_SORT_FRACTION = 0.6

__all__ = [
    "knn_indices",
    "knn_preservation",
    "rnx",
    "trustworthiness",
    "continuity",
    "sample_pairs",
    "distance_correlation",
    "pair_distances",
    "normalized_stress",
    "local_radius",
    "density_preservation",
    "true_density_correlation",
    "knn_graph",
    "graph_edge_jaccard",
    "n_components",
    "cluster_scores",
    "evaluate",
    "input_fidelity",
]


# --------------------------------------------------------------------------- helpers


def _points(Z: Any, name: str = "Z") -> np.ndarray:
    """Return Z as a finite (n, d) float64 array; a 1-D input becomes (n, 1)."""
    A = np.asarray(Z, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2 or A.shape[0] < 2 or A.shape[1] < 1:
        raise ValueError(f"{name} must be an (n, d) array with n >= 2, got {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError(f"{name} contains NaN or inf")
    return A


def _pair(ref: Any, emb: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate a (reference, embedding) pair with the same number of points."""
    R, E = _points(ref, "ref"), _points(emb, "emb")
    if R.shape[0] != E.shape[0]:
        raise ValueError(f"ref has {R.shape[0]} points but emb has {E.shape[0]}")
    return R, E


def _check_k(k: int, n: int, upper: int, what: str) -> int:
    """Validate 1 <= k < upper for an integer k."""
    if isinstance(k, bool) or int(k) != k:
        raise ValueError(f"k must be an integer, got {k!r}")
    k = int(k)
    if not 1 <= k < upper:
        raise ValueError(f"{what} needs 1 <= k < {upper} for n = {n}, got k = {k}")
    return k


def _knn(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact kNN distances and indices of every point, self excluded, sorted by distance."""
    from sklearn.neighbors import NearestNeighbors

    algorithm = "brute" if Z.shape[1] > _BRUTE_DIM else "auto"
    nn = NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(Z)
    dist, idx = nn.kneighbors()
    return dist, idx


def _row_sq_dists(Z: np.ndarray, sq_norms: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Squared distances from rows lo:hi of Z (centred) to every row, self set to inf."""
    if Z.shape[1] > _BRUTE_DIM:  # Gram trick, in place: one (c, n) array
        D = Z[lo:hi] @ Z.T
        D *= -2.0
        D += sq_norms[lo:hi, None]
        D += sq_norms[None, :]
        np.maximum(D, 0.0, out=D)
    else:  # coordinate-wise differences: exact ties stay exact ties
        D = np.zeros((hi - lo, Z.shape[0]))
        diff = np.empty_like(D)
        for c in range(Z.shape[1]):
            np.subtract(Z[lo:hi, c, None], Z[None, :, c], out=diff)
            np.square(diff, out=diff)
            D += diff
    D[np.arange(hi - lo), np.arange(lo, hi)] = np.inf
    return D


def _ranks_of(Z: np.ndarray, index_sets: Sequence[np.ndarray]) -> list[np.ndarray]:
    """For each (n, K) index array, the 1-based rank of each listed j among i's neighbours in Z.

    rank(i, j) = 1 + #{l != i : d(i, l) < d(i, j)}, so a tie group shares its smallest
    rank. Rows are processed in chunks, and each chunk takes whichever of two exact
    counting routes is cheaper. When the listed points are mostly near i (a good
    embedding), only the few distances below the largest threshold are bucketed
    against the sorted thresholds. When they are spread out (a poor embedding), the
    chunk's rows are sorted once and every threshold is located by binary search.
    """
    n = Z.shape[0]
    # Centring curbs cancellation in the Gram trick; the exact path keeps raw values.
    Zc = Z - Z.mean(axis=0) if Z.shape[1] > _BRUTE_DIM else Z
    sq_norms = np.einsum("ij,ij->i", Zc, Zc)
    out = [np.empty(s.shape, dtype=np.int32) for s in index_sets]
    chunk = max(1, min(n, _CHUNK_BYTES // (8 * n)))
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        c = hi - lo
        D = _row_sq_dists(Zc, sq_norms, lo, hi)
        rows = np.arange(c)[:, None]
        thrs = [
            D[rows, s[lo:hi]] for s in index_sets
        ]  # (c, K) distance to each listed j
        below = sum(float((D < t.max(axis=1, keepdims=True)).mean()) for t in thrs)
        if below > _SORT_FRACTION:
            S = np.sort(D, axis=1)
            for thr, res in zip(thrs, out, strict=True):
                for r in range(c):
                    res[lo + r] = np.searchsorted(S[r], thr[r], side="left") + 1
            del S
        else:
            for thr, res in zip(thrs, out, strict=True):
                K = thr.shape[1]
                order = np.argsort(thr, axis=1, kind="stable")
                ts = np.take_along_axis(thr, order, axis=1)
                less = np.empty((c, K), dtype=np.int64)
                for r in range(c):
                    # bucket(v) = #thresholds <= v; v < ts[m] iff bucket(v) <= m, and a
                    # value >= max(ts) is below no threshold, so it can be dropped.
                    v = D[r][D[r] < ts[r, -1]]
                    bucket = np.searchsorted(ts[r], v, side="right")
                    less[r] = np.cumsum(np.bincount(bucket, minlength=K + 1))[:K]
                np.put_along_axis(res[lo:hi], order, less + 1, axis=1)
        del D  # free this chunk before the next one is allocated
    return out


def _trust_from_ranks(ranks: np.ndarray, k: int) -> float:
    """Venna-Kaski statistic from the reference ranks of each point's first k embedding neighbours."""
    n = ranks.shape[0]
    penalty = np.maximum(ranks[:, :k].astype(np.int64) - k, 0).sum()
    return float(1.0 - 2.0 * penalty / (n * k * (2.0 * n - 3.0 * k - 1.0)))


def _overlap(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise size of the intersection of two (n, k) index arrays with distinct rows."""
    s = np.sort(np.concatenate([A, B], axis=1), axis=1)
    return (s[:, 1:] == s[:, :-1]).sum(axis=1)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation, NaN (without a warning) if either side is constant."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, NaN (without a warning) if either side is constant."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def _row(metric: str, k: float | None, reference: str, value: float) -> dict:
    """One long-format metric row."""
    return {
        "metric": metric,
        "k": float("nan") if k is None else float(k),
        "reference": reference,
        "value": float(value),
    }


# ---------------------------------------------------------------- neighbourhood metrics


def knn_indices(Z: np.ndarray, k: int) -> np.ndarray:
    """Exact Euclidean k nearest neighbours of every point, self excluded, nearest first."""
    Z = _points(Z)
    k = _check_k(k, Z.shape[0], Z.shape[0], "knn_indices")
    return _knn(Z, k)[1]


def knn_preservation(
    ref: np.ndarray, emb: np.ndarray, ks: Sequence[int]
) -> dict[int, float]:
    """Q_NX(k): mean_i |N_k^ref(i) ∩ N_k^emb(i)| / k. Equals kNN precision = recall."""
    R, E = _pair(ref, emb)
    n = R.shape[0]
    ks = [_check_k(k, n, n, "knn_preservation") for k in ks]
    if not ks:
        return {}
    K = max(ks)
    iR, iE = _knn(R, K)[1], _knn(E, K)[1]
    return {k: float(_overlap(iR[:, :k], iE[:, :k]).mean() / k) for k in ks}


def rnx(ref: np.ndarray, emb: np.ndarray, ks: Sequence[int]) -> dict[int, float]:
    """Chance-adjusted R_NX(k) = ((n-1) Q_NX(k) - k) / (n - 1 - k) (Lee & Verleysen).

    Needs k < n - 1: at k = n - 1 every neighbourhood is everything and R_NX is 0/0.
    """
    n = _points(ref, "ref").shape[0]
    ks = [_check_k(k, n, n - 1, "rnx") for k in ks]
    q = knn_preservation(ref, emb, ks)
    return {k: ((n - 1) * q[k] - k) / (n - 1 - k) for k in ks}


def trustworthiness(ref: np.ndarray, emb: np.ndarray, k: int) -> float:
    """Venna & Kaski trustworthiness T(k): penalises embedding neighbours that are far in ref.

    T(k) = 1 - 2 / (n k (2n - 3k - 1)) * sum_i sum_{j in U_k(i)} (r(i, j) - k), with
    U_k(i) the embedding k-NN of i outside its ref k-NN and r the 1-based ref rank.
    The normaliser is valid for k < n / 2 only, so larger k raises.
    """
    R, E = _pair(ref, emb)
    n = R.shape[0]
    k = _check_k(k, n, (n + 1) // 2, "trustworthiness (k < n/2)")
    (ranks,) = _ranks_of(R, [_knn(E, k)[1]])
    return _trust_from_ranks(ranks, k)


def continuity(ref: np.ndarray, emb: np.ndarray, k: int) -> float:
    """Continuity C(k): penalises ref neighbours pushed apart in emb; = trustworthiness(emb, ref, k)."""
    return trustworthiness(emb, ref, k)


# -------------------------------------------------------------------- global metrics


def sample_pairs(n: int, m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw m distinct unordered pairs i<j, reproducibly; m is capped at n(n-1)/2.

    Below the cap the pairs are a uniform random m-subset of all pairs, sorted by
    (i, j). At or above the cap every pair is returned, in the same order.
    """
    n, m = int(n), int(m)
    if n < 2:
        raise ValueError(f"need n >= 2 points to form a pair, got n = {n}")
    if m < 1:
        raise ValueError(f"need m >= 1 pairs, got m = {m}")
    total = n * (n - 1) // 2
    if m >= total:
        i, j = np.triu_indices(n, k=1)
        return i.astype(np.int64), j.astype(np.int64)
    t = np.sort(np.random.default_rng(seed).choice(total, size=m, replace=False))
    # Unrank t in the row-major upper triangle: row i starts at off(i) = i(2n-i-1)/2.
    off = lambda r: r * (2 * n - r - 1) // 2  # noqa: E731
    i = np.floor(((2 * n - 1) - np.sqrt((2.0 * n - 1) ** 2 - 8.0 * t)) / 2).astype(
        np.int64
    )
    i = np.clip(i, 0, n - 2)
    for _ in range(2):  # float rounding can leave i off by one either way
        i = np.where(off(i) > t, i - 1, i)
        i = np.where(off(i + 1) <= t, i + 1, i)
    j = t - off(i) + i + 1
    return i, j


def pair_distances(Z: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Euclidean distance between Z[i] and Z[j] for each index pair, in bounded-memory chunks."""
    Z = _points(Z)
    i, j = np.asarray(i, dtype=np.intp), np.asarray(j, dtype=np.intp)
    if i.shape != j.shape or i.ndim != 1:
        raise ValueError("i and j must be 1-D index arrays of the same length")
    out = np.empty(i.shape[0])
    step = max(1, (4 * 2**20) // Z.shape[1])
    for lo in range(0, i.shape[0], step):
        diff = Z[i[lo : lo + step]] - Z[j[lo : lo + step]]
        out[lo : lo + step] = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    return out


def _dist_pair(ref_d: Any, emb_d: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate two equal-length 1-D distance vectors."""
    d, e = np.asarray(ref_d, dtype=np.float64), np.asarray(emb_d, dtype=np.float64)
    if d.ndim != 1 or d.shape != e.shape or d.shape[0] < 2:
        raise ValueError("ref_d and emb_d must be 1-D arrays of the same length >= 2")
    if not (np.isfinite(d).all() and np.isfinite(e).all()):
        raise ValueError("distances contain NaN or inf")
    return d, e


def distance_correlation(ref_d: np.ndarray, emb_d: np.ndarray) -> dict[str, float]:
    """{'spearman': ..., 'pearson': ...} over the sampled pair distances."""
    d, e = _dist_pair(ref_d, emb_d)
    return {"spearman": _spearman(d, e), "pearson": _pearson(d, e)}


def normalized_stress(ref_d: np.ndarray, emb_d: np.ndarray) -> float:
    """sqrt(sum (d - a e)^2 / sum d^2) with the optimal scale a = <d,e>/<e,e>.

    Invariant to rescaling emb_d; 0 for an exactly scaled copy; 1 if emb_d is all zero
    (the best a then leaves the whole of d unexplained). NaN if ref_d is all zero.
    """
    d, e = _dist_pair(ref_d, emb_d)
    dd = float(d @ d)
    if dd == 0:
        return float("nan")
    ee = float(e @ e)
    a = float(d @ e) / ee if ee > 0 else 0.0
    resid = d - a * e
    return float(np.sqrt(max(float(resid @ resid), 0.0) / dd))


# ------------------------------------------------------------------- density metrics


def local_radius(Z: np.ndarray, k: int) -> np.ndarray:
    """Distance from each point to its k-th nearest neighbour (self excluded)."""
    Z = _points(Z)
    k = _check_k(k, Z.shape[0], Z.shape[0], "local_radius")
    return _knn(Z, k)[0][:, k - 1]


def density_preservation(ref: np.ndarray, emb: np.ndarray, k: int = DENSITY_K) -> float:
    """Spearman correlation of log local radius in ref vs emb (densMAP-style).

    Spearman is rank-based and log is monotone, so this is computed on the radii
    themselves, which keeps a zero radius (duplicate points) finite.
    """
    R, E = _pair(ref, emb)
    k = _check_k(k, R.shape[0], R.shape[0], "density_preservation")
    return _spearman(_knn(R, k)[0][:, k - 1], _knn(E, k)[0][:, k - 1])


def true_density_correlation(
    density: np.ndarray, emb: np.ndarray, k: int = DENSITY_K
) -> float:
    """Spearman of the known sampling density vs 1/r_k^2 in the 2-D embedding.

    1/r^2 is a decreasing function of r, so this equals Spearman(density, -r_k), which
    is how it is computed (a zero radius would make 1/r^2 infinite).
    """
    E = _points(emb, "emb")
    rho = np.asarray(density, dtype=np.float64)
    if rho.shape != (E.shape[0],) or not np.isfinite(rho).all():
        raise ValueError("density must be a finite (n,) array matching emb")
    k = _check_k(k, E.shape[0], E.shape[0], "true_density_correlation")
    return _spearman(rho, -_knn(E, k)[0][:, k - 1])


# --------------------------------------------------------------------- graph metrics


def _graph_from_idx(idx: np.ndarray) -> sp.csr_matrix:
    """Symmetrised (union) binary adjacency from an (n, k) neighbour array."""
    n, k = idx.shape
    rows = np.repeat(np.arange(n), k)
    A = sp.csr_matrix(
        (np.ones(n * k, dtype=np.int8), (rows, idx.ravel())), shape=(n, n)
    )
    G = ((A + A.T) > 0).astype(np.int8)
    G.setdiag(0)
    G.eliminate_zeros()
    return G.tocsr()


def knn_graph(Z: np.ndarray, k: int) -> sp.csr_matrix:
    """Symmetrised (union) kNN adjacency: i~j if either is among the other's k nearest."""
    return _graph_from_idx(knn_indices(Z, k))


def graph_edge_jaccard(G1: sp.spmatrix, G2: sp.spmatrix) -> float:
    """Jaccard index |E1 ∩ E2| / |E1 ∪ E2| of the undirected edge sets of two graphs."""
    if G1.shape != G2.shape:
        raise ValueError(f"graphs have different shapes {G1.shape} and {G2.shape}")
    A = (sp.csr_matrix(G1) != 0).astype(np.int8)
    B = (sp.csr_matrix(G2) != 0).astype(np.int8)
    inter = A.multiply(B).nnz
    union = A.nnz + B.nnz - inter
    return float(inter / union) if union else 1.0


def n_components(G: sp.spmatrix) -> int:
    """Number of connected components of an undirected graph."""
    return int(connected_components(sp.csr_matrix(G), directed=False)[0])


# ------------------------------------------------------------------- cluster metrics


def cluster_scores(
    Z: np.ndarray, labels: np.ndarray, seed: int, n_clusters: int | None = None
) -> dict:
    """Cluster recovery in the space Z (embedding or original) against true labels.

    Keys: ``silhouette`` (true labels; on a seeded random subsample of
    ``SILHOUETTE_MAX_N`` points when n is larger, since it is O(n^2)),
    ``knn_accuracy`` (mean accuracy of a 15-NN classifier over 5-fold
    StratifiedKFold(shuffle=True, random_state=seed)), ``kmeans_ari``/``kmeans_ami``
    (KMeans with k = n_clusters, default the number of distinct labels, n_init=10,
    random_state=seed), ``hdbscan_ari``/``hdbscan_ami`` (sklearn HDBSCAN with
    min_cluster_size = max(10, n // 100); every noise point is its own singleton
    cluster) and ``hdbscan_n_clusters`` (clusters found, noise excluded). A score
    that is undefined for the input (one class, too few points) is NaN.
    """
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        silhouette_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier

    Z = _points(Z)
    y = np.asarray(labels)
    n = Z.shape[0]
    if y.shape != (n,):
        raise ValueError(f"labels must have shape ({n},), got {y.shape}")
    classes, y_int = np.unique(y, return_inverse=True)
    n_true = len(classes) if n_clusters is None else int(n_clusters)
    if not 1 <= n_true <= n:
        raise ValueError(f"n_clusters must be in [1, {n}], got {n_true}")
    rng = np.random.default_rng(seed)
    nan = float("nan")
    out: dict[str, float] = {}

    sub = (
        np.arange(n)
        if n <= SILHOUETTE_MAX_N
        else rng.choice(n, SILHOUETTE_MAX_N, replace=False)
    )
    n_lab = len(np.unique(y_int[sub]))
    out["silhouette"] = (
        float(silhouette_score(Z[sub], y_int[sub])) if 2 <= n_lab < len(sub) else nan
    )

    if len(classes) >= 2 and np.bincount(y_int).max() >= 5:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        accs = []
        for tr, te in folds.split(Z, y_int):
            clf = KNeighborsClassifier(n_neighbors=min(15, len(tr))).fit(
                Z[tr], y_int[tr]
            )
            accs.append(float((clf.predict(Z[te]) == y_int[te]).mean()))
        out["knn_accuracy"] = float(np.mean(accs))
    else:
        out["knn_accuracy"] = nan

    km = KMeans(n_clusters=n_true, n_init=10, random_state=seed).fit_predict(Z)
    out["kmeans_ari"] = float(adjusted_rand_score(y_int, km))
    out["kmeans_ami"] = float(adjusted_mutual_info_score(y_int, km))

    mcs = max(10, n // 100)
    if n > mcs:
        hd = HDBSCAN(min_cluster_size=mcs, copy=True).fit_predict(Z)
        noise = hd < 0
        out["hdbscan_n_clusters"] = float(len(np.unique(hd[~noise])))
        hd = hd.copy()
        hd[noise] = hd.max(initial=-1) + 1 + np.arange(noise.sum())
        out["hdbscan_ari"] = float(adjusted_rand_score(y_int, hd))
        out["hdbscan_ami"] = float(adjusted_mutual_info_score(y_int, hd))
    else:
        out["hdbscan_ari"] = out["hdbscan_ami"] = out["hdbscan_n_clusters"] = nan
    return out


# ---------------------------------------------------------------------- evaluation


def _neighbourhood_rows(
    refs: Mapping[str, np.ndarray], E: np.ndarray, ks: Sequence[int]
) -> tuple[
    list[dict], dict[str, tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]
]:
    """Q_NX, R_NX, T, C, density and graph rows of E against each reference cloud.

    Each space's kNN is computed once, at the largest k needed, and sliced for the
    smaller ones; E's ranks for every reference come from a single pass over E.
    Returns (rows, {name: (dist, idx)} for each reference, (dist, idx) for E).
    """
    n = E.shape[0]
    k_nn = sorted({int(k) for k in ks if 1 <= int(k) < n - 1})
    k_tc = [k for k in k_nn if k < n / 2]
    k_aux = GRAPH_K if GRAPH_K < n else None
    K = max([*k_nn, k_aux or 0])
    if K == 0:
        return [], {}, (np.empty((n, 0)), np.empty((n, 0), dtype=np.intp))

    # A reference given twice as the same array object (latent is X) is computed once.
    first = {
        name: next(m for m, Q in refs.items() if Q is R) for name, R in refs.items()
    }
    unique = [name for name in refs if first[name] == name]
    nn_E = _knn(E, K)
    nn_R = {name: _knn(refs[name], K) for name in unique}
    rank_R: dict[str, np.ndarray] = {}
    rank_E: dict[str, np.ndarray] = {}
    if k_tc:
        Kt = max(k_tc)
        rank_E = dict(
            zip(unique, _ranks_of(E, [nn_R[m][1][:, :Kt] for m in unique]), strict=True)
        )
        rank_R = {m: _ranks_of(refs[m], [nn_E[1][:, :Kt]])[0] for m in unique}
    nn_R = {name: nn_R[first[name]] for name in refs}

    rows: list[dict] = []
    for name in refs:
        iR, iE = nn_R[name][1], nn_E[1]
        for k in k_nn:
            q = float(_overlap(iR[:, :k], iE[:, :k]).mean() / k)
            rows.append(_row("knn_preservation", k, name, q))
            rows.append(_row("rnx", k, name, ((n - 1) * q - k) / (n - 1 - k)))
        for k in k_tc:
            t = _trust_from_ranks(rank_R[first[name]], k)
            c = _trust_from_ranks(rank_E[first[name]], k)
            rows.append(_row("trustworthiness", k, name, t))
            rows.append(_row("continuity", k, name, c))
        if k_aux:
            rows.append(
                _row(
                    "density_spearman",
                    DENSITY_K,
                    name,
                    _spearman(
                        nn_R[name][0][:, DENSITY_K - 1], nn_E[0][:, DENSITY_K - 1]
                    ),
                )
            )
            G_R = _graph_from_idx(iR[:, :GRAPH_K])
            G_E = _graph_from_idx(iE[:, :GRAPH_K])
            rows.append(
                _row("graph_jaccard", GRAPH_K, name, graph_edge_jaccard(G_R, G_E))
            )
    return rows, nn_R, nn_E


def _reference_clouds(ds: Dataset, refs: Sequence[str]) -> dict[str, np.ndarray]:
    """Map reference names to point clouds: truth -> ds.latent, input -> ds.X."""
    unknown = set(refs) - {"truth", "input"}
    if unknown:
        raise ValueError(
            f"unknown reference(s) {sorted(unknown)}; use 'truth' and/or 'input'"
        )
    source = {"truth": ds.latent, "input": ds.X}
    clouds: dict[str, np.ndarray] = {}
    for name in dict.fromkeys(refs):
        clouds[name] = _points(
            source[name], f"ds.{'latent' if name == 'truth' else 'X'}"
        )
    if (
        "truth" in clouds
        and "input" in clouds
        and np.array_equal(clouds["truth"], clouds["input"])
    ):
        clouds["input"] = clouds["truth"]  # latent == X: share the kNN and ranks
    return clouds


def evaluate(
    ds: Dataset,
    emb: np.ndarray,
    *,
    ks: Sequence[int] = (5, 10, 30, 100),
    n_pairs: int = 50_000,
    seed: int = 0,
    refs: Sequence[str] = ("truth", "input"),
) -> list[dict]:
    """Every applicable metric, as long-format rows {metric, k, reference, value}.

    reference="truth" compares against ds.latent (and ds.geodesic for
    geodesic_spearman), reference="input" against ds.X, and reference="emb" marks a
    metric of the embedding alone (cluster scores, n_components of its k=15 graph).
    true_density_spearman and geodesic_spearman are reported with reference="truth",
    and only when "truth" is in ``refs``.

    Which k values are kept: knn_preservation and rnx need k < n - 1, and
    trustworthiness and continuity need k < n / 2 (the Venna-Kaski normaliser);
    other k values are skipped for those metrics. density_*, graph_jaccard and
    n_components use k = 15 and are skipped when n <= 15. Cluster metrics need
    ds.labels; true_density_spearman needs ds.density; geodesic_spearman needs
    ds.geodesic to return distances.
    """
    E = _points(emb, "emb")
    n = E.shape[0]
    if _points(ds.X, "ds.X").shape[0] != n:
        raise ValueError(f"emb has {n} rows but ds.X has {np.shape(ds.X)[0]}")
    clouds = _reference_clouds(ds, refs)

    rows, _, nn_E = _neighbourhood_rows(clouds, E, ks)

    i, j = sample_pairs(n, n_pairs, seed)
    e_d = pair_distances(E, i, j)
    for name, R in clouds.items():
        r_d = pair_distances(R, i, j)
        corr = distance_correlation(r_d, e_d)
        rows.append(_row("dist_spearman", None, name, corr["spearman"]))
        rows.append(_row("dist_pearson", None, name, corr["pearson"]))
        rows.append(_row("stress", None, name, normalized_stress(r_d, e_d)))
        if name == "truth":
            geo = ds.geodesic(i, j)
            if geo is not None:
                geo = np.asarray(geo, dtype=np.float64)
                rows.append(_row("geodesic_spearman", None, name, _spearman(geo, e_d)))

    if GRAPH_K < n:
        r_E = nn_E[0][:, DENSITY_K - 1]
        if ds.density is not None and "truth" in clouds:
            rho = np.asarray(ds.density, dtype=np.float64)
            if rho.shape != (n,):
                raise ValueError(f"ds.density must have shape ({n},), got {rho.shape}")
            rows.append(
                _row("true_density_spearman", DENSITY_K, "truth", _spearman(rho, -r_E))
            )
        G_E = _graph_from_idx(nn_E[1][:, :GRAPH_K])
        rows.append(_row("n_components", GRAPH_K, "emb", n_components(G_E)))

    if ds.labels is not None:
        for metric, value in cluster_scores(E, ds.labels, seed).items():
            rows.append(_row(metric, None, "emb", value))
    return rows


def input_fidelity(ds: Dataset, ks: Sequence[int] = (5, 10, 30, 100)) -> list[dict]:
    """The local metrics of the input itself: ds.X scored as an "embedding" of ds.latent.

    This is the quality of the neighbour graph a method is handed, before any
    embedding. Rows: knn_preservation, rnx, trustworthiness, continuity (same k rules
    as ``evaluate``), density_spearman and graph_jaccard (k = 15), all with
    reference="truth"; plus n_components of the k = 15 graph of ds.latent
    (reference="truth") and of ds.X (reference="input").
    """
    X = _points(ds.X, "ds.X")
    L = _points(ds.latent, "ds.latent")
    if L.shape[0] != X.shape[0]:
        raise ValueError("ds.latent and ds.X have different numbers of points")
    rows, nn_R, nn_X = _neighbourhood_rows({"truth": L}, X, ks)
    if GRAPH_K < X.shape[0]:
        G_L = _graph_from_idx(nn_R["truth"][1][:, :GRAPH_K])
        G_X = _graph_from_idx(nn_X[1][:, :GRAPH_K])
        rows.append(_row("n_components", GRAPH_K, "truth", n_components(G_L)))
        rows.append(_row("n_components", GRAPH_K, "input", n_components(G_X)))
    return rows
