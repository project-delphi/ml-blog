"""Tests for umaplab.stability."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial import procrustes as scipy_procrustes
from umaplab import stability as S


def _similar_copy(A, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(A.shape[1], A.shape[1])))
    Q[:, 0] *= -1  # include a reflection
    return 0.37 * A @ Q + np.array([4.0, -9.0])


def test_similar_copy_aligns_exactly():
    A = np.random.default_rng(0).normal(size=(300, 2))
    B = _similar_copy(A)
    aligned, disparity = S.procrustes_align(A, B)
    assert disparity < 1e-10
    np.testing.assert_allclose(aligned, A, atol=1e-10)  # back in A's frame and scale
    rows = S.seed_pair_rows(A, B)
    got = {
        (r["metric"], None if math.isnan(r["k"]) else r["k"]): r["value"] for r in rows
    }
    assert got[("procrustes_disparity", None)] < 1e-10
    assert got[("seed_knn_overlap", 10.0)] == 1.0
    assert got[("seed_knn_overlap", 30.0)] == 1.0
    assert got[("seed_dist_spearman", None)] == pytest.approx(1.0)


def test_matches_scipy_procrustes():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(200, 2))
    B = A @ np.array([[0.0, 1.0], [1.0, 0.0]]) * 5 + rng.normal(size=(200, 2))
    aligned, disparity = S.procrustes_align(A, B)
    m1, m2, d_scipy = scipy_procrustes(A, B)
    assert disparity == pytest.approx(d_scipy, rel=1e-10)
    assert 0 < disparity < 1
    # scipy returns standardised matrices; undo A's standardisation to compare
    a0 = A - A.mean(0)
    np.testing.assert_allclose(aligned, m2 * np.linalg.norm(a0) + A.mean(0), atol=1e-10)
    # symmetric, as in scipy
    assert S.procrustes_align(B, A)[1] == pytest.approx(disparity, rel=1e-10)


def test_procrustes_validation():
    A = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValueError):
        S.procrustes_align(A, A[:-1])
    with pytest.raises(ValueError):
        S.procrustes_align(A, np.ones_like(A))  # one unique point
    with pytest.raises(ValueError):
        S.procrustes_align(A.ravel(), A.ravel())
    bad = A.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        S.procrustes_align(A, bad)


def test_seed_pair_rows_schema_and_k_rules():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(40, 2)), rng.normal(size=(40, 2))
    rows = S.seed_pair_rows(A, B, ks=(5, 39, 100))
    assert all(set(r) == {"metric", "k", "reference", "value"} for r in rows)
    assert {r["metric"] for r in rows} == {
        "procrustes_disparity",
        "seed_knn_overlap",
        "seed_dist_spearman",
    }
    assert [r["k"] for r in rows if r["metric"] == "seed_knn_overlap"] == [5.0]
    d = next(r["value"] for r in rows if r["metric"] == "procrustes_disparity")
    assert 0.5 < d <= 1.0  # unrelated clouds
