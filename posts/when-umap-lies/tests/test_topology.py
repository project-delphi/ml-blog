"""Known-shape tests for umaplab.topology (ripser)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from umaplab import topology as T


def _noisy_circle(n=600, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    return np.c_[np.cos(a), np.sin(a)] + noise * rng.normal(size=(n, 2))


def _two_blobs(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack(
        [rng.normal(size=(n // 2, 2)), rng.normal(size=(n // 2, 2)) + [12.0, 0.0]]
    )


def _filled_disk(n=800, seed=0):
    rng = np.random.default_rng(seed)
    r, a = np.sqrt(rng.uniform(size=n)), rng.uniform(0, 2 * np.pi, n)
    return np.c_[r * np.cos(a), r * np.sin(a)]


def test_subsample_maxmin_reproducible_distinct_and_spread():
    Z = _filled_disk(2000)
    idx = T.subsample(Z, 100, seed=3)
    assert len(idx) == 100 and len(set(idx.tolist())) == 100
    np.testing.assert_array_equal(idx, T.subsample(Z, 100, seed=3))
    assert idx[0] != T.subsample(Z, 100, seed=4)[0]  # seeded start
    rand = T.subsample(Z, 100, seed=3, method="random")
    assert len(set(rand.tolist())) == 100
    # covering radius: maxmin covers the cloud far better than a random draw
    cover = lambda s: cdist(Z, Z[s]).min(axis=1).max()  # noqa: E731
    assert cover(idx) < 0.6 * cover(rand)
    # farthest-point property: each new point maximises distance to those before it
    d = cdist(Z, Z[idx[:10]])
    assert np.isclose(d[:, :9].min(axis=1).max(), d[idx[9], :9].min())


def test_subsample_edge_cases():
    Z = np.random.default_rng(0).normal(size=(20, 3))
    np.testing.assert_array_equal(T.subsample(Z, 20, 0), np.arange(20))
    np.testing.assert_array_equal(T.subsample(Z, 50, 0), np.arange(20))
    with pytest.raises(ValueError):
        T.subsample(Z, 5, 0, method="kmeans")
    with pytest.raises(ValueError):
        T.subsample(Z, 0, 0)


def test_persistence_is_scale_free():
    Z = _noisy_circle(300)
    a = T.persistence(Z, m=200)
    b = T.persistence(1000.0 * Z + 7.0, m=200)
    for da, db in zip(a, b, strict=True):
        np.testing.assert_allclose(da, db, rtol=1e-9, atol=1e-12)
    finite = np.concatenate([d[np.isfinite(d[:, 1])] for d in a])
    assert finite.max() <= 1.0 + 1e-12  # diameter units
    assert np.isinf(a[0][:, 1]).sum() == 1


def test_noisy_circle_has_one_dominant_loop():
    dgms = T.persistence(_noisy_circle(), maxdim=1, m=300, seed=0)
    top1, top2, _ = T.top_persistence(dgms, 1)
    assert top1 / top2 > 3
    assert T.betti_estimate(dgms, T.DEFAULT_THRESHOLD) == (1, 1)


def test_two_blobs_have_two_components():
    dgms = T.persistence(_two_blobs(), maxdim=1, m=300, seed=0)
    assert T.betti_estimate(dgms, T.DEFAULT_THRESHOLD) == (2, 0)


def test_filled_disk_has_no_dominant_loop():
    dgms = T.persistence(_filled_disk(), maxdim=1, m=300, seed=0)
    top1, top2, _ = T.top_persistence(dgms, 1)
    assert T.betti_estimate(dgms, T.DEFAULT_THRESHOLD) == (1, 0)
    assert top1 < T.DEFAULT_THRESHOLD
    assert top1 < 0.2 * T.top_persistence(T.persistence(_noisy_circle(), m=300), 1)[0]


def test_betti_estimate_and_top_persistence_on_hand_diagrams():
    dgms = [
        np.array([[0.0, 0.05], [0.0, 0.3], [0.0, np.inf]]),
        np.array([[0.2, 0.25], [0.1, 0.9]]),
    ]
    assert T.betti_estimate(dgms, 0.1) == (2, 1)
    assert T.betti_estimate(dgms, 0.0) == (3, 2)
    assert T.top_persistence(dgms, 0, r=3) == pytest.approx(
        [0.3, 0.05, 0.0]
    )  # inf excluded
    assert T.top_persistence(dgms, 1, r=1) == pytest.approx([0.8])
    with pytest.raises(ValueError):
        T.betti_estimate(dgms, -0.1)
    with pytest.raises(ValueError):
        T.top_persistence(dgms, 2)
    with pytest.raises(ValueError):
        T.top_persistence(dgms, 1, r=0)


def test_topology_rows():
    X = np.c_[_noisy_circle(400), 0.01 * np.random.default_rng(0).normal(size=(400, 8))]
    ds = SimpleNamespace(X=X)
    rows = T.topology_rows(ds, _two_blobs(400), seed=0, m=200)
    assert all(set(r) == {"metric", "k", "reference", "value"} for r in rows)
    assert all(math.isnan(r["k"]) for r in rows)
    got = {(r["metric"], r["reference"]): r["value"] for r in rows}
    assert set(got) == {
        (m, ref)
        for m in ("h1_top1", "h1_top2", "h1_ratio", "betti0", "betti1")
        for ref in ("input", "emb")
    }
    assert got[("betti1", "input")] == 1 and got[("h1_ratio", "input")] > 3
    assert got[("betti0", "emb")] == 2 and got[("betti1", "emb")] == 0
