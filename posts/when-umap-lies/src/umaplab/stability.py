"""Procrustes alignment and seed-to-seed comparisons of embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .metrics import _knn, _overlap, _spearman, pair_distances, sample_pairs

__all__ = ["procrustes_align", "seed_pair_rows"]


def procrustes_align(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """Align B to A by translation, uniform scale, rotation and reflection; return the
    aligned B and the Procrustes disparity (scipy.spatial.procrustes semantics).

    The disparity is scipy's: both clouds are centred and scaled to unit Frobenius
    norm, B is rotated/reflected and scaled to best fit A, and the disparity is the
    sum of squared differences that remain (in [0, 1]; symmetric in A and B). Unlike
    scipy, the returned B is mapped back into A's own frame and scale (A's centroid and
    norm), so it can be drawn over A directly.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or A.shape != B.shape:
        raise ValueError(
            f"A and B must be 2-D arrays of equal shape, got {A.shape}, {B.shape}"
        )
    if A.shape[0] < 2:
        raise ValueError("Procrustes alignment needs at least 2 points")
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise ValueError("A or B contains NaN or inf")
    mu_A, mu_B = A.mean(axis=0), B.mean(axis=0)
    a, b = A - mu_A, B - mu_B
    norm_A, norm_B = np.linalg.norm(a), np.linalg.norm(b)
    if norm_A == 0 or norm_B == 0:
        raise ValueError("Input matrices must contain >1 unique points")
    a /= norm_A
    b /= norm_B
    # min over orthogonal R and scale s of ||a - s b R||: R = U V^T, s = sum(S) for b^T a = U S V^T
    U, S, Vt = np.linalg.svd(b.T @ a)
    fitted = (b @ (U @ Vt)) * S.sum()
    disparity = float(np.sum(np.square(a - fitted)))
    return fitted * norm_A + mu_A, disparity


def seed_pair_rows(
    ref_emb: np.ndarray,
    other_emb: np.ndarray,
    ks: Sequence[int] = (10, 30),
    *,
    n_pairs: int = 50_000,
    seed: int = 0,
) -> list[dict]:
    """Rows {metric: 'procrustes_disparity'|'seed_knn_overlap'|'seed_dist_spearman', k, value}.

    Rows also carry reference="emb" so they share the metrics.parquet schema.
    seed_knn_overlap is Q_NX(k) between the two embeddings (k < n - 1 kept);
    seed_dist_spearman is over ``n_pairs`` pairs drawn by ``sample_pairs(n, n_pairs, seed)``.
    """
    A = np.asarray(ref_emb, dtype=np.float64)
    B = np.asarray(other_emb, dtype=np.float64)
    _, disparity = procrustes_align(A, B)  # validates shapes and finiteness
    n = A.shape[0]

    def row(metric: str, k: float | None, value: float) -> dict:
        k = float("nan") if k is None else float(k)
        return {"metric": metric, "k": k, "reference": "emb", "value": float(value)}

    rows = [row("procrustes_disparity", None, disparity)]
    k_ok = sorted({int(k) for k in ks if 1 <= int(k) < n - 1})
    if k_ok:
        K = max(k_ok)
        iA, iB = _knn(A, K)[1], _knn(B, K)[1]
        for k in k_ok:
            rows.append(
                row("seed_knn_overlap", k, _overlap(iA[:, :k], iB[:, :k]).mean() / k)
            )
    i, j = sample_pairs(n, n_pairs, seed)
    rows.append(
        row(
            "seed_dist_spearman",
            None,
            _spearman(pair_distances(A, i, j), pair_distances(B, i, j)),
        )
    )
    return rows
