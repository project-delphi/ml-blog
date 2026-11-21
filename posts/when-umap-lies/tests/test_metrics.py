"""Known-answer tests for umaplab.metrics (no dependency on generators.py)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.manifold import trustworthiness as sk_trustworthiness
from umaplab import metrics as M

METRIC_NAMES = {
    "knn_preservation",
    "rnx",
    "trustworthiness",
    "continuity",
    "dist_spearman",
    "dist_pearson",
    "geodesic_spearman",
    "stress",
    "density_spearman",
    "true_density_spearman",
    "silhouette",
    "knn_accuracy",
    "kmeans_ari",
    "kmeans_ami",
    "hdbscan_ari",
    "hdbscan_ami",
    "hdbscan_n_clusters",
    "graph_jaccard",
    "n_components",
}


@dataclass
class FakeDataset:
    """Stand-in for generators.Dataset with only the attributes metrics reads."""

    X: np.ndarray
    latent: np.ndarray
    labels: np.ndarray | None = None
    density: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    geo: Any = None  # callable (i, j) -> distances, or None

    def geodesic(self, i, j):
        return None if self.geo is None else self.geo(i, j)


def _rotation_scale_reflection(Z, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(Z.shape[1], Z.shape[1])))
    Q[:, 0] *= -1  # make sure a reflection is in there
    return 3.7 * Z @ Q + np.array([5.0, -2.0])


def _rows_by(rows, reference):
    """{(metric, k): value} for one reference; a NaN k becomes None."""
    return {
        (r["metric"], None if math.isnan(r["k"]) else r["k"]): r["value"]
        for r in rows
        if r["reference"] == reference
    }


def _direct_trustworthiness(ref, emb, k):
    """Venna & Kaski by the book: full rank matrices and explicit U_k sets."""
    n = len(ref)
    dr, de = cdist(ref, ref), cdist(emb, emb)
    np.fill_diagonal(dr, np.inf)
    np.fill_diagonal(de, np.inf)
    order_r = np.argsort(dr, axis=1, kind="stable")
    rank_r = np.empty_like(order_r)
    rank_r[np.arange(n)[:, None], order_r] = np.arange(1, n + 1)
    nn_r = order_r[:, :k]
    nn_e = np.argsort(de, axis=1, kind="stable")[:, :k]
    total = 0
    for i in range(n):
        for j in set(nn_e[i]) - set(nn_r[i]):
            total += rank_r[i, j] - k
    return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * total


# ------------------------------------------------------------------ trustworthiness

# Six points on a line, every distance distinct so no ranks tie.
SIX_REF = np.array([0.0, 1.0, 3.0, 7.0, 12.0, 20.0])[:, None]
# Embedding: point i sits at the original position of point PERM[i].
SIX_EMB = SIX_REF[[0, 4, 2, 1, 3, 5]]


def test_trustworthiness_hand_computed_six_points():
    # k = 2, n = 6: normaliser 2 / (n k (2n - 3k - 1)) = 2 / (6*2*5) = 1/30.
    # Original 2-NN: 0:{1,2} 1:{0,2} 2:{1,0} 3:{2,4} 4:{3,5} 5:{4,3}
    # Embedding positions 0,12,3,1,7,20 -> 2-NN: 0:{3,2} 1:{4,5} 2:{3,0} 3:{0,2}
    #   4:{2,1} 5:{1,4}
    # Intruders j and their original ranks r(i,j), penalty r - k:
    #   i=0: 3 (r=3) -> 1;  i=1: 4 (r=4) -> 2, 5 (r=5) -> 3;  i=2: 3 (r=3) -> 1
    #   i=3: 0 (r=4) -> 2;  i=4: 2 (r=3) -> 1, 1 (r=4) -> 2;  i=5: 1 (r=4) -> 2
    # Sum = 14, so T(2) = 1 - 14/30 = 8/15.
    t = M.trustworthiness(SIX_REF, SIX_EMB, 2)
    assert t == pytest.approx(8 / 15, abs=1e-12)
    assert sk_trustworthiness(SIX_REF, SIX_EMB, n_neighbors=2) == pytest.approx(8 / 15)
    assert _direct_trustworthiness(SIX_REF, SIX_EMB, 2) == pytest.approx(8 / 15)
    # Continuity by the same hand count with roles swapped: missing neighbours
    #   i=0: 1 (emb r=4) -> 2; i=1: 0 (r=5) -> 3, 2 (r=3) -> 1; i=2: 1 (r=4) -> 2
    #   i=3: 4 (r=3) -> 1;     i=4: 3 (r=3) -> 1, 5 (r=5) -> 3; i=5: 3 (r=4) -> 2
    # Sum = 15, so C(2) = 1 - 15/30 = 1/2.
    assert M.continuity(SIX_REF, SIX_EMB, 2) == pytest.approx(0.5, abs=1e-12)
    assert sk_trustworthiness(SIX_EMB, SIX_REF, n_neighbors=2) == pytest.approx(0.5)


@pytest.mark.parametrize("D", [3, 40])  # both distance code paths (direct, Gram)
def test_trustworthiness_matches_direct_and_sklearn(D):
    rng = np.random.default_rng(D)
    ref = rng.normal(size=(300, D))
    emb = ref[:, :2] + 0.7 * rng.normal(size=(300, 2))
    for k in (1, 3, 10, 50, 149):
        ours = M.trustworthiness(ref, emb, k)
        assert ours == pytest.approx(_direct_trustworthiness(ref, emb, k), abs=1e-12)
        assert ours == pytest.approx(
            sk_trustworthiness(ref, emb, n_neighbors=k), abs=1e-12
        )


def test_trustworthiness_chunking_does_not_change_result(monkeypatch):
    rng = np.random.default_rng(5)
    ref, emb = rng.normal(size=(257, 30)), rng.normal(size=(257, 2))
    full = (M.trustworthiness(ref, emb, 20), M.continuity(ref, emb, 20))
    monkeypatch.setattr(M, "_CHUNK_BYTES", 8 * 257 * 7)  # 7-row chunks, ragged tail
    assert (M.trustworthiness(ref, emb, 20), M.continuity(ref, emb, 20)) == full


@pytest.mark.parametrize("D", [2, 30])
def test_both_rank_routes_agree(monkeypatch, D):
    rng = np.random.default_rng(D)
    Z = rng.integers(0, 5, size=(300, D)).astype(float)  # integer grid: exact ties
    good = M.knn_indices(Z + 0.01 * rng.normal(size=Z.shape), 25)
    bad = rng.permuted(np.tile(np.arange(300), (300, 1)), axis=1)[:, :25]
    bad = np.where(bad == np.arange(300)[:, None], (bad + 1) % 300, bad)
    results = []
    for route in (-1.0, 1e9):  # force sort, force bucket
        monkeypatch.setattr(M, "_SORT_FRACTION", route)
        results.append(M._ranks_of(Z, [good, bad]))
    for a, b in zip(*results, strict=True):
        np.testing.assert_array_equal(a, b)
    # and both equal the definition: 1 + #{l != i : d(i, l) < d(i, j)}
    d = cdist(Z, Z, "sqeuclidean")
    np.fill_diagonal(d, np.inf)
    for idx, ranks in zip((good, bad), results[0], strict=True):
        thr = np.take_along_axis(d, idx, axis=1)
        expect = (d[:, None, :] < thr[:, :, None]).sum(-1) + 1
        if D <= 20:  # exact arithmetic path: ties are exact
            np.testing.assert_array_equal(ranks, expect)
        else:  # Gram path: round-off may split a tie, never moves a rank beyond it
            lo = (d[:, None, :] < thr[:, :, None] - 1e-9).sum(-1) + 1
            hi = (d[:, None, :] <= thr[:, :, None] + 1e-9).sum(-1)
            assert ((ranks >= lo) & (ranks <= hi)).all()


def test_trustworthiness_memory_is_chunked():
    import tracemalloc

    rng = np.random.default_rng(0)
    n = 4000
    ref, emb = rng.normal(size=(n, 5)), rng.normal(size=(n, 2))
    tracemalloc.start()
    M.trustworthiness(ref, emb, 10)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Peak is a couple of row-chunk buffers, not the 256 MB of a dense float64
    # distance matrix plus its argsort (sklearn's route).
    assert peak < 3 * M._CHUNK_BYTES + 10 * 2**20
    assert peak < 0.3 * 2 * 8 * n * n  # ~67 MB measured at n = 4000


def test_continuity_is_swapped_trustworthiness():
    rng = np.random.default_rng(2)
    ref, emb = rng.normal(size=(120, 6)), rng.normal(size=(120, 2))
    for k in (5, 20):
        assert M.continuity(ref, emb, k) == M.trustworthiness(emb, ref, k)


def test_trustworthiness_validation():
    rng = np.random.default_rng(0)
    ref, emb = rng.normal(size=(10, 3)), rng.normal(size=(10, 2))
    assert 0 <= M.trustworthiness(ref, emb, 4) <= 1  # largest k < n/2
    for bad in (0, 5, 9, 2.5, True):
        with pytest.raises(ValueError):
            M.trustworthiness(ref, emb, bad)
    with pytest.raises(ValueError):
        M.trustworthiness(ref, emb[:9], 2)
    bad_ref = ref.copy()
    bad_ref[0, 0] = np.nan
    with pytest.raises(ValueError):
        M.trustworthiness(bad_ref, emb, 2)


def test_identity_with_duplicates_is_fully_trustworthy():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 2))
    Z = np.repeat(base, 3, axis=0)  # every point has two exact duplicates
    assert M.trustworthiness(Z, Z, 5) == 1.0
    assert M.continuity(Z, Z, 5) == 1.0


# ---------------------------------------------------------------------------- kNN


def test_knn_indices_exclude_self_and_are_exact():
    rng = np.random.default_rng(0)
    for D in (2, 30):
        Z = rng.normal(size=(200, D))
        idx = M.knn_indices(Z, 7)
        assert idx.shape == (200, 7)
        assert not (idx == np.arange(200)[:, None]).any()
        d = cdist(Z, Z)
        np.fill_diagonal(d, np.inf)
        np.testing.assert_array_equal(
            np.sort(idx, axis=1), np.sort(np.argsort(d, 1)[:, :7], 1)
        )


def test_knn_indices_with_heavy_duplicates():
    # 12 copies of one point: more coincident points than k, so the self index must be
    # dropped by index, not by distance.
    Z = np.vstack([np.zeros((12, 2)), np.arange(1.0, 9.0)[:, None] * [1.0, 0.0]])
    idx = M.knn_indices(Z, 5)
    assert not (idx == np.arange(len(Z))[:, None]).any()
    assert all(len(set(row)) == 5 for row in idx)
    assert (
        idx[:12] < 12
    ).all()  # a duplicate's neighbours are its copies, at distance 0


def test_knn_validation():
    Z = np.random.default_rng(0).normal(size=(10, 2))
    for bad in (0, 10, 11):
        with pytest.raises(ValueError):
            M.knn_indices(Z, bad)
    with pytest.raises(ValueError):
        M.knn_indices(np.zeros((1, 2)), 1)
    with pytest.raises(ValueError):
        M.knn_preservation(Z, Z[:9], [3])
    with pytest.raises(ValueError):
        M.rnx(Z, Z, [9])  # k = n - 1: R_NX is 0/0


# ---------------------------------------------------------- identity and permutation


def test_identity_and_similarity_transform_are_perfect():
    rng = np.random.default_rng(0)
    latent = rng.uniform(size=(400, 2))
    ds = FakeDataset(X=latent, latent=latent)
    for emb in (latent, _rotation_scale_reflection(latent)):
        rows = _rows_by(M.evaluate(ds, emb, ks=(5, 10, 30), n_pairs=5000), "truth")
        for k in (5.0, 10.0, 30.0):
            assert rows[("knn_preservation", k)] == pytest.approx(1.0)
            assert rows[("rnx", k)] == pytest.approx(1.0)
            assert rows[("trustworthiness", k)] == pytest.approx(1.0)
            assert rows[("continuity", k)] == pytest.approx(1.0)
        assert rows[("dist_spearman", None)] == pytest.approx(1.0)
        assert rows[("dist_pearson", None)] == pytest.approx(1.0)
        assert rows[("stress", None)] == pytest.approx(0.0, abs=1e-12)
        assert rows[("density_spearman", 15.0)] == pytest.approx(1.0)
        assert rows[("graph_jaccard", 15.0)] == pytest.approx(1.0)


def test_random_permutation_is_chance_level():
    rng = np.random.default_rng(3)
    n, ks = 2000, [5, 10, 30]
    ref = rng.normal(size=(n, 2))
    emb = ref[rng.permutation(n)]
    q, r = M.knn_preservation(ref, emb, ks), M.rnx(ref, emb, ks)
    for k in ks:
        p = k / (n - 1)
        # overlap per point is hypergeometric(n-1, k, k); SE of the mean of overlap/k
        var = k * p * (1 - p) * (n - 1 - k) / (n - 2) / k**2
        se = math.sqrt(var / n)
        assert abs(q[k] - p) < 3 * se
        assert abs(r[k]) < 3 * se * (n - 1) / (n - 1 - k)
    assert abs(M.density_preservation(ref, emb)) < 0.1


def test_knn_preservation_and_rnx_formula():
    rng = np.random.default_rng(1)
    ref, emb = rng.normal(size=(150, 4)), rng.normal(size=(150, 2))
    q, r = M.knn_preservation(ref, emb, [10]), M.rnx(ref, emb, [10])
    iR, iE = M.knn_indices(ref, 10), M.knn_indices(emb, 10)
    direct = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(iR, iE, strict=True)])
    assert q[10] == pytest.approx(direct)
    assert r[10] == pytest.approx((149 * direct - 10) / (149 - 10))


# ---------------------------------------------------------------- pairs and stress


def test_sample_pairs_distinct_ordered_deterministic():
    for n, m in ((50, 300), (10_000, 50_000), (2001, 1999)):
        i, j = M.sample_pairs(n, m, seed=7)
        assert len(i) == m and (i < j).all() and (i >= 0).all() and (j < n).all()
        assert len(np.unique(i * n + j)) == m
        i2, j2 = M.sample_pairs(n, m, seed=7)
        np.testing.assert_array_equal(i, i2)
        np.testing.assert_array_equal(j, j2)
        i3, _ = M.sample_pairs(n, m, seed=8)
        assert not np.array_equal(i, i3)


def test_sample_pairs_caps_at_all_pairs():
    for m in (45, 46, 10**6):
        i, j = M.sample_pairs(10, m, seed=0)
        assert len(i) == 45
        assert set(zip(i.tolist(), j.tolist(), strict=True)) == {
            (a, b) for a in range(10) for b in range(a + 1, 10)
        }
    i, j = M.sample_pairs(10, 44, seed=0)  # just below the cap: still distinct
    assert len(set(zip(i.tolist(), j.tolist(), strict=True))) == 44


def test_sample_pairs_is_uniform_over_pairs():
    # every pair index equally likely: each of the 6 pairs of n=4 drawn ~ m/6 times
    counts = np.zeros(6)
    for s in range(600):
        i, j = M.sample_pairs(4, 2, seed=s)
        for a, b in zip(i, j, strict=True):
            counts[
                {(0, 1): 0, (0, 2): 1, (0, 3): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5}[a, b]
            ] += 1
    assert np.all(np.abs(counts - 200) < 4 * math.sqrt(200))


def test_sample_pairs_validation():
    with pytest.raises(ValueError):
        M.sample_pairs(1, 5, 0)
    with pytest.raises(ValueError):
        M.sample_pairs(10, 0, 0)


def test_pair_distances_and_correlation():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(30, 5))
    i, j = M.sample_pairs(30, 100, 0)
    np.testing.assert_allclose(
        M.pair_distances(Z, i, j), np.linalg.norm(Z[i] - Z[j], axis=1)
    )
    with pytest.raises(ValueError):
        M.pair_distances(Z, i, j[:-1])
    d = M.pair_distances(Z, i, j)
    c = M.distance_correlation(d, 2 * d + 1)
    assert c["spearman"] == pytest.approx(1.0) and c["pearson"] == pytest.approx(1.0)
    assert math.isnan(M.distance_correlation(d, np.ones_like(d))["spearman"])
    with pytest.raises(ValueError):
        M.distance_correlation(d, d[:-1])


def test_normalized_stress():
    rng = np.random.default_rng(0)
    d = rng.uniform(1, 2, size=500)
    e = d + 0.3 * rng.normal(size=500)
    s = M.normalized_stress(d, e)
    assert 0 < s < 1
    assert M.normalized_stress(d, 17.0 * e) == pytest.approx(s)
    assert M.normalized_stress(d, 0.01 * d) == pytest.approx(0.0, abs=1e-12)
    a = (d @ e) / (e @ e)
    assert s == pytest.approx(np.sqrt(np.sum((d - a * e) ** 2) / np.sum(d**2)))
    assert M.normalized_stress(d, np.zeros_like(d)) == 1.0
    with pytest.raises(ValueError):
        M.normalized_stress(d, e[:-1])


# --------------------------------------------------------------------------- density


def test_density_preservation_identity_and_local_radius():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 2)) * np.linspace(0.1, 3, 300)[:, None]
    assert M.density_preservation(Z, Z) == pytest.approx(1.0)
    assert M.density_preservation(Z, _rotation_scale_reflection(Z)) == pytest.approx(
        1.0
    )
    d = cdist(Z, Z)
    np.fill_diagonal(d, np.inf)
    np.testing.assert_allclose(M.local_radius(Z, 15), np.sort(d, axis=1)[:, 14])
    with pytest.raises(ValueError):
        M.local_radius(Z, 300)


def test_true_density_correlation():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(500, 2))
    r = M.local_radius(emb, 15)
    assert M.true_density_correlation(1 / r**2, emb) == pytest.approx(1.0)
    assert M.true_density_correlation(r, emb) == pytest.approx(-1.0)
    # Gaussian sample: true density exp(-|x|^2/2) tracks the kNN density estimate
    rho = np.exp(-0.5 * (emb**2).sum(1))
    assert M.true_density_correlation(rho, emb) > 0.8
    with pytest.raises(ValueError):
        M.true_density_correlation(rho[:-1], emb)


# ------------------------------------------------------------------- graph, clusters


def _two_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    Z = np.vstack(
        [rng.normal(size=(n // 2, 2)), rng.normal(size=(n // 2, 2)) + [50.0, 0.0]]
    )
    return Z, np.repeat([0, 1], n // 2)


def test_graph_metrics():
    Z, _ = _two_blobs()
    G = M.knn_graph(Z, 5)
    assert (G != G.T).nnz == 0 and G.diagonal().sum() == 0
    idx = M.knn_indices(Z, 5)
    assert all(G[i, j] for i in range(len(Z)) for j in idx[i])
    assert M.graph_edge_jaccard(G, G) == 1.0
    assert M.n_components(G) == 2
    G2 = M.knn_graph(np.random.default_rng(1).normal(size=Z.shape), 5)
    assert 0 <= M.graph_edge_jaccard(G, G2) < 0.1
    with pytest.raises(ValueError):
        M.graph_edge_jaccard(G, G2[:-1, :-1])


def test_cluster_scores_two_far_blobs():
    Z, y = _two_blobs(400)
    s = M.cluster_scores(Z, y, seed=0)
    assert s["kmeans_ari"] == 1.0 and s["kmeans_ami"] == pytest.approx(1.0)
    assert s["knn_accuracy"] == 1.0
    assert s["hdbscan_ari"] == pytest.approx(1.0, abs=0.02)
    assert s["hdbscan_n_clusters"] == 2
    assert s["silhouette"] > 0.9
    assert s == M.cluster_scores(Z, y, seed=0)


def test_cluster_scores_noise_points_are_singletons():
    Z, y = _two_blobs(400)
    Z = np.vstack([Z, [[25.0, 300.0], [25.0, -300.0]]])  # two isolated points -> noise
    y = np.append(y, [0, 1])
    s = M.cluster_scores(Z, y, seed=0)
    assert s["hdbscan_n_clusters"] == 2
    assert 0.95 < s["hdbscan_ari"] < 1.0  # singletons, not a merged "noise" cluster


def test_cluster_scores_subsamples_silhouette(monkeypatch):
    monkeypatch.setattr(M, "SILHOUETTE_MAX_N", 100)
    Z, y = _two_blobs(400)
    a, b = M.cluster_scores(Z, y, seed=0), M.cluster_scores(Z, y, seed=1)
    assert a["silhouette"] > 0.9 and a["silhouette"] != b["silhouette"]


def test_cluster_scores_validation():
    Z, y = _two_blobs(100)
    with pytest.raises(ValueError):
        M.cluster_scores(Z, y[:-1], seed=0)
    with pytest.raises(ValueError):
        M.cluster_scores(Z, y, seed=0, n_clusters=0)
    one = M.cluster_scores(Z, np.zeros(100, int), seed=0)
    assert math.isnan(one["silhouette"]) and math.isnan(one["knn_accuracy"])


# --------------------------------------------------------------------------- evaluate


def _fake(n=300, labels=True, density=True, geo=True, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.uniform(size=(n, 2))
    X = np.hstack([latent, 0.05 * rng.normal(size=(n, 8))])
    return FakeDataset(
        X=X,
        latent=latent,
        labels=(latent[:, 0] > 0.5).astype(int) if labels else None,
        density=np.ones(n) + latent[:, 0] if density else None,
        geo=(lambda i, j: np.linalg.norm(latent[i] - latent[j], axis=1))
        if geo
        else None,
    )


def test_evaluate_row_schema_and_names():
    ds = _fake()
    emb = ds.latent + 0.05 * np.random.default_rng(1).normal(size=ds.latent.shape)
    rows = M.evaluate(ds, emb, ks=(5, 10, 30, 100), n_pairs=2000)
    assert all(set(r) == {"metric", "k", "reference", "value"} for r in rows)
    assert all(
        isinstance(r["k"], float) and isinstance(r["value"], float) for r in rows
    )
    assert {r["metric"] for r in rows} == METRIC_NAMES
    assert {r["reference"] for r in rows} == {"truth", "input", "emb"}
    # each (metric, k, reference) appears once
    keys = [
        (r["metric"], "nan" if math.isnan(r["k"]) else r["k"], r["reference"])
        for r in rows
    ]
    assert len(keys) == len(set(keys))
    by = _rows_by(rows, "truth")
    assert by[("trustworthiness", 30.0)] == pytest.approx(
        M.trustworthiness(ds.latent, emb, 30)
    )
    assert by[("continuity", 100.0)] == pytest.approx(M.continuity(ds.latent, emb, 100))
    assert by[("knn_preservation", 10.0)] == pytest.approx(
        M.knn_preservation(ds.latent, emb, [10])[10]
    )
    byi = _rows_by(rows, "input")
    assert byi[("trustworthiness", 5.0)] == pytest.approx(
        M.trustworthiness(ds.X, emb, 5)
    )
    geo = [r for r in rows if r["metric"] == "geodesic_spearman"]
    assert len(geo) == 1 and geo[0]["reference"] == "truth" and geo[0]["value"] > 0.95


def test_evaluate_skips_k_and_optional_metrics():
    ds = _fake(n=60, labels=False, density=False, geo=False)
    rows = M.evaluate(ds, ds.latent, ks=(5, 10, 30, 58, 59, 100), n_pairs=10**6)
    names = {r["metric"] for r in rows}
    assert names.isdisjoint(
        {"geodesic_spearman", "true_density_spearman", "silhouette", "kmeans_ari"}
    )

    def ks_of(m):
        return {r["k"] for r in rows if r["metric"] == m and r["reference"] == "truth"}

    assert ks_of("knn_preservation") == {5.0, 10.0, 30.0, 58.0}  # k < n - 1
    assert ks_of("rnx") == {5.0, 10.0, 30.0, 58.0}
    assert ks_of("trustworthiness") == {5.0, 10.0}  # k < n / 2
    assert ks_of("continuity") == {5.0, 10.0}
    # a tiny dataset: no k=15 metrics at all, still the global ones
    small = _fake(n=12)
    rows = M.evaluate(small, small.latent, ks=(5,), n_pairs=10)
    names = {r["metric"] for r in rows}
    assert "graph_jaccard" not in names and "n_components" not in names
    assert {"dist_spearman", "stress", "trustworthiness"} <= names


def test_evaluate_refs_and_validation():
    ds = _fake(n=100)
    rows = M.evaluate(ds, ds.latent, ks=(5,), n_pairs=500, refs=("input",))
    assert {r["reference"] for r in rows} == {"input", "emb"}
    with pytest.raises(ValueError):
        M.evaluate(ds, ds.latent, refs=("latent",))
    with pytest.raises(ValueError):
        M.evaluate(ds, ds.latent[:-1])


def test_evaluate_latent_is_input_gives_equal_rows():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(150, 3))
    ds = FakeDataset(X=Z, latent=Z)
    rows = M.evaluate(ds, Z[:, :2], ks=(5, 20), n_pairs=1000)
    t, i = _rows_by(rows, "truth"), _rows_by(rows, "input")
    assert t.keys() == i.keys() and all(t[key] == i[key] for key in t)


def test_input_fidelity():
    ds = _fake(n=200)
    rows = M.input_fidelity(ds, ks=(5, 10))
    assert all(set(r) == {"metric", "k", "reference", "value"} for r in rows)
    names = {r["metric"] for r in rows}
    assert names == {
        "knn_preservation",
        "rnx",
        "trustworthiness",
        "continuity",
        "density_spearman",
        "graph_jaccard",
        "n_components",
    }
    by = _rows_by(rows, "truth")
    assert by[("trustworthiness", 10.0)] == pytest.approx(
        M.trustworthiness(ds.latent, ds.X, 10)
    )
    assert by[("n_components", 15.0)] == 1.0
    assert _rows_by(rows, "input")[("n_components", 15.0)] == 1.0
