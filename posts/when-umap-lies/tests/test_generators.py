"""Tests for umaplab.generators: shapes, ground truth, and the ambient transforms."""

import numpy as np
import pytest
from matplotlib.path import Path as MplPath
from scipy.linalg import null_space
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from umaplab import generators as g

SMALL = {
    "gaussian_clusters": {},
    "swiss_roll": {"noise": 0.1},
    "circle": {"noise": 0.05, "dim": 4},
    "sphere": {"noise": 0.05},
    "branching": {"levels": 2},
    "variable_density": {},
    "overlapping": {},
    "latent_factor": {"D": 12, "noise": 0.1},
    "uniform_ball": {},
    "elephant_target": {},
}


def _all(n=300, seed=0):
    return {name: g.GENERATORS[name](n, seed, **kw) for name, kw in SMALL.items()}


# --------------------------------------------------------------------------- basics


@pytest.mark.parametrize("name", list(SMALL))
def test_shapes_dtypes_and_meta(name):
    n = 300
    ds = g.GENERATORS[name](n, 0, **SMALL[name])
    assert ds.name == name
    assert ds.X.shape[0] == n and ds.X.dtype == np.float64 and ds.X.ndim == 2
    assert ds.latent.shape[0] == n and ds.latent.dtype == np.float64
    assert ds.clean.shape == ds.X.shape and ds.clean.dtype == np.float64
    assert np.all(np.isfinite(ds.X))
    if ds.labels is not None:
        assert ds.labels.shape == (n,) and np.issubdtype(ds.labels.dtype, np.integer)
    if ds.density is not None:
        assert ds.density.shape == (n,) and np.all(ds.density > 0)
    m = ds.meta
    for key in (
        "generator",
        "params",
        "seed",
        "n",
        "intrinsic_dim",
        "ambient_dim",
        "betti",
        "transforms",
    ):
        assert key in m
    assert m["generator"] == name and m["seed"] == 0 and m["n"] == n
    assert m["ambient_dim"] == ds.X.shape[1]
    assert m["transforms"] == []
    assert m["betti"] is None or isinstance(m["betti"], tuple)
    for k, v in m["params"].items():
        assert m[k] == v


@pytest.mark.parametrize("name", list(SMALL))
def test_determinism_and_seed_sensitivity(name):
    a = g.GENERATORS[name](200, 3, **SMALL[name])
    b = g.GENERATORS[name](200, 3, **SMALL[name])
    c = g.GENERATORS[name](200, 4, **SMALL[name])
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.latent, b.latent)
    assert not np.allclose(a.X, c.X)


def test_clean_is_x_without_noise():
    ds = g.swiss_roll(100, 0)
    assert ds.clean is ds.X
    noisy = g.swiss_roll(100, 0, noise=0.5)
    assert not np.allclose(noisy.clean, noisy.X)
    np.testing.assert_array_equal(noisy.clean, ds.clean)


def test_geodesic_unknown_returns_none():
    for name in ("gaussian_clusters", "overlapping", "elephant_target"):
        ds = g.GENERATORS[name](50, 0)
        assert ds.geodesic(np.arange(5), np.arange(5, 10)) is None


@pytest.mark.parametrize(
    "name", ["swiss_roll", "circle", "sphere", "branching", "variable_density"]
)
def test_geodesic_vectorised_symmetric_zero_diagonal(name):
    ds = g.GENERATORS[name](200, 0)
    i = np.arange(200)
    np.testing.assert_allclose(ds.geodesic(i, i), 0.0, atol=1e-12)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 200, (2, 500))
    np.testing.assert_allclose(ds.geodesic(a, b), ds.geodesic(b, a))
    # broadcasting: a 2-D grid of pairs
    grid = ds.geodesic(i[:10, None], i[None, :10])
    assert grid.shape == (10, 10)


# --------------------------------------------------------------------------- swiss roll


def test_swiss_roll_arc_length_inverse():
    s = np.linspace(0, g._roll_length(1.5), 1001)
    t = g._arc_inverse(s)
    np.testing.assert_allclose(g._arc(t) - g._arc(g._T0), s, atol=1e-9)
    assert t[0] == pytest.approx(1.5 * np.pi) and t[-1] == pytest.approx(4.5 * np.pi)


def test_swiss_roll_latent_matches_euclidean_for_neighbours():
    ds = g.swiss_roll(2000, 0)
    d, j = cKDTree(ds.X).query(ds.X, k=2)
    i = np.arange(len(ds.X))
    lat = ds.geodesic(i, j[:, 1])
    rel = np.abs(lat - d[:, 1]) / d[:, 1]
    assert rel.max() < 0.02


def test_swiss_roll_geodesic_dominates_euclidean():
    ds = g.swiss_roll(1000, 1)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 1000, (2, 20000))
    geo = ds.geodesic(a, b)
    euc = np.linalg.norm(ds.clean[a] - ds.clean[b], axis=1)
    assert np.all(geo >= euc - 1e-9)


def test_swiss_roll_uniform_on_surface():
    ds = g.swiss_roll(20000, 0)
    length = g._roll_length(1.5)
    counts, _ = np.histogram(ds.extra["s"], bins=10, range=(0, length))
    expected = len(ds.X) / 10
    assert np.all(np.abs(counts - expected) < 5 * np.sqrt(expected))
    # the s-range covers sklearn's t-range at 1.5 turns
    assert ds.extra["t"].min() >= 1.5 * np.pi - 1e-9
    assert ds.extra["t"].max() <= 4.5 * np.pi + 1e-9


# --------------------------------------------------------------------------- sphere, circle


def test_sphere_unit_norm_and_geodesic_range():
    ds = g.sphere(500, 0)
    np.testing.assert_allclose(np.linalg.norm(ds.X, axis=1), 1.0, atol=1e-12)
    i, j = np.triu_indices(500, 1)
    geo = ds.geodesic(i, j)
    assert geo.min() >= 0 and geo.max() <= np.pi
    euc = np.linalg.norm(ds.X[i] - ds.X[j], axis=1)
    np.testing.assert_allclose(euc, 2 * np.sin(geo / 2), atol=1e-9)
    assert ds.meta["betti"] == (1, 0, 1)
    assert g.sphere(10, 0, m=3).meta["betti"] == (1, 0, 0, 1)


def test_circle_geodesic_at_most_pi_times_radius():
    ds = g.circle(500, 0, radius=2.0)
    i, j = np.triu_indices(500, 1)
    geo = ds.geodesic(i, j)
    assert geo.max() <= 2.0 * np.pi + 1e-12
    euc = np.linalg.norm(ds.X[i] - ds.X[j], axis=1)
    np.testing.assert_allclose(euc, 2 * 2.0 * np.sin(geo / (2 * 2.0)), atol=1e-9)
    unit = g.circle(300, 1)
    assert unit.geodesic(*np.triu_indices(300, 1)).max() <= np.pi


# --------------------------------------------------------------------------- branching


@pytest.mark.parametrize("levels", [1, 2])
def test_branching_pseudotime_monotone_along_each_branch(levels):
    ds = g.branching(1500, 0, levels=levels)
    nodes = ds.extra["nodes"]
    ends = ds.extra["seg_nodes"]
    for b in np.unique(ds.labels):
        m = ds.labels == b
        start = nodes[ends[b, 0]]
        along = np.linalg.norm(ds.clean[m] - start, axis=1)
        order = np.argsort(along)
        assert np.all(np.diff(ds.extra["pseudotime"][m][order]) >= -1e-12)
    # children start where their parent ends
    parent = ds.extra["seg_parent"]
    for k in range(1, len(ends)):
        assert ends[k, 0] == ends[parent[k], 1]
    n_seg = 1 + 3 + (6 if levels == 2 else 0)
    assert len(ends) == n_seg


def test_branching_geodesic_is_a_metric():
    ds = g.branching(800, 0, levels=2)
    rng = np.random.default_rng(1)
    a, b, c = rng.integers(0, 800, (3, 20000))
    dab, dbc, dac = ds.geodesic(a, b), ds.geodesic(b, c), ds.geodesic(a, c)
    np.testing.assert_allclose(dab, ds.geodesic(b, a))
    assert np.all(dac <= dab + dbc + 1e-9)
    assert np.all(dab >= 0)
    # geodesic between noiseless positions is never shorter than the chord
    euc = np.linalg.norm(ds.clean[a] - ds.clean[b], axis=1)
    assert np.all(dab >= euc - 1e-9)


def test_branching_geodesic_from_root_is_pseudotime():
    ds = g.branching(500, 2, levels=2)
    root = int(np.argmin(ds.extra["pseudotime"]))
    i = np.arange(500)
    pt = ds.extra["pseudotime"]
    # d(x, root_sample) = pt(x) - pt(root) whenever root_sample sits on the trunk
    assert ds.labels[root] == 0
    np.testing.assert_allclose(ds.geodesic(i, root), pt - pt[root], atol=1e-12)


def test_branching_points_uniform_by_length():
    ds = g.branching(6000, 0)
    counts = np.bincount(ds.labels, minlength=4)
    assert np.all(np.abs(counts - 1500) < 5 * np.sqrt(1500))


# --------------------------------------------------------------------------- variable density


@pytest.mark.parametrize("shape", ["strip", "roll"])
def test_variable_density_histogram_matches_density(shape):
    n = 20000
    ds = g.variable_density(n, 0, shape=shape)
    s = ds.extra["s"]
    length = g._roll_length(1.5)
    edges = np.linspace(0, length, 11)
    counts, _ = np.histogram(s, bins=edges)
    # expected counts from the stored density: integrate pdf_s = density * height
    fine = np.linspace(0, length, 20001)
    c = np.log(10.0)
    pdf = c * np.exp(c * fine / length) / (length * 9.0)
    cdf = np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(fine))])
    expected = n * np.diff(np.interp(edges, fine, cdf))
    assert np.all(np.abs(counts - expected) < 4 * np.sqrt(expected))
    # the stored per-point density is that same pdf / height
    np.testing.assert_allclose(
        ds.density * 21.0, c * np.exp(c * s / length) / (length * 9.0)
    )
    assert ds.density.max() / ds.density.min() == pytest.approx(10.0, rel=0.02)


def test_variable_density_roll_is_isometric_flattening():
    strip = g.variable_density(1500, 0, shape="strip")
    roll = g.variable_density(1500, 0, shape="roll")
    np.testing.assert_array_equal(strip.latent, roll.latent)
    np.testing.assert_allclose(strip.X[:, :2], strip.latent)
    d, j = cKDTree(roll.X).query(roll.X, k=2)
    lat = roll.geodesic(np.arange(1500), j[:, 1])
    assert np.max(np.abs(lat - d[:, 1]) / d[:, 1]) < 0.02


def test_variable_density_ratio_one_is_uniform():
    ds = g.variable_density(500, 0, density_ratio=1.0)
    assert np.ptp(ds.density) == 0


# --------------------------------------------------------------------------- overlapping


def test_overlapping_posterior_in_unit_interval_and_calibrated():
    ds = g.overlapping(40000, 0, dim=3, delta=1.5, weights=(0.3, 0.7))
    post = ds.extra["posterior"]
    assert post.min() >= 0 and post.max() <= 1
    bins = np.clip((post * 10).astype(int), 0, 9)
    for b in range(10):
        m = bins == b
        if m.sum() < 200:
            continue
        freq = ds.labels[m].mean()
        p = post[m].mean()
        se = np.sqrt(p * (1 - p) / m.sum())
        assert abs(freq - p) < 4 * se + 1e-3
    assert ds.labels.mean() == pytest.approx(0.7, abs=0.01)


def test_overlapping_posterior_matches_density_ratio():
    ds = g.overlapping(200, 0, delta=2.0)
    mu = ds.extra["means"]
    l0 = -0.5 * ((ds.X - mu[0]) ** 2).sum(1)
    l1 = -0.5 * ((ds.X - mu[1]) ** 2).sum(1)
    np.testing.assert_allclose(ds.extra["posterior"], 1 / (1 + np.exp(l0 - l1)))


# --------------------------------------------------------------------------- clusters


def test_gaussian_clusters_min_centre_distance_and_sizes():
    ds = g.gaussian_clusters(2000, 0, separation=4.0, within_sd=0.5, imbalance=5.0)
    c = ds.extra["centres"]
    assert pdist(c).min() == pytest.approx(4.0 * 0.5, rel=1e-12)
    sizes = np.bincount(ds.labels)
    assert sizes.sum() == 2000
    np.testing.assert_array_equal(sizes, ds.extra["sizes"])
    assert sizes.max() / sizes.min() == pytest.approx(5.0, rel=0.05)
    assert np.all(np.diff(sizes) <= 0)


def test_gaussian_clusters_balanced_by_default():
    sizes = np.bincount(g.gaussian_clusters(1000, 0).labels)
    np.testing.assert_array_equal(sizes, 100)


def test_gaussian_clusters_anisotropy_and_within_sd():
    ds = g.gaussian_clusters(
        6000, 0, n_clusters=2, dim=4, anisotropy=9.0, within_sd=2.0
    )
    for k in range(2):
        Xk = ds.X[ds.labels == k]
        ev = np.linalg.eigvalsh(np.cov(Xk.T))
        sd = np.sqrt(ev)
        assert sd.max() / sd.min() == pytest.approx(9.0, rel=0.15)
        # geometric mean of the axis sds is within_sd
        assert np.exp(np.log(sd).mean()) == pytest.approx(2.0, rel=0.05)
        np.testing.assert_allclose(Xk.mean(0), ds.extra["centres"][k], atol=0.3)


def test_gaussian_clusters_density_is_mixture_pdf():
    from scipy.stats import multivariate_normal

    ds = g.gaussian_clusters(300, 0, n_clusters=3, dim=3, anisotropy=4.0)
    sds, R = ds.extra["axis_sds"], ds.extra["rotations"]
    w = ds.extra["sizes"] / 300
    pdf = sum(
        w[k]
        * multivariate_normal(
            ds.extra["centres"][k], R[k] @ np.diag(sds**2) @ R[k].T
        ).pdf(ds.X)
        for k in range(3)
    )
    np.testing.assert_allclose(np.exp(ds.extra["log_density"]), pdf, rtol=1e-9)


# --------------------------------------------------------------------------- others


def test_latent_factor_isometry_and_ball():
    ds = g.latent_factor(300, 0, d=4, D=9)
    np.testing.assert_allclose(pdist(ds.X), pdist(ds.latent), atol=1e-10)
    assert ds.meta["ambient_dim"] == 9 and ds.meta["intrinsic_dim"] == 4
    ball = g.uniform_ball(2000, 0, d=3)
    r = np.linalg.norm(ball.X, axis=1)
    assert r.max() <= 1
    # radius CDF of the uniform d-ball is r^d
    assert np.mean(r**3 < 0.5) == pytest.approx(0.5, abs=0.04)


def test_elephant_inside_polygon_and_uniform():
    ds = g.elephant_target(2000, 0)
    poly = ds.extra["polygon"]
    assert 30 <= len(poly) <= 60
    assert MplPath(poly).contains_points(ds.X).all()
    assert ds.X.shape == (2000, 2)
    assert ds.meta["betti"] is None


# --------------------------------------------------------------------------- transforms


def test_random_isometry_preserves_distances():
    ds = g.swiss_roll(300, 0, noise=0.2)
    iso = g.random_isometry(ds, 50, seed=1)
    assert iso.X.shape == (300, 50)
    np.testing.assert_allclose(pdist(iso.X), pdist(ds.X), atol=1e-8)
    assert iso.meta["ambient_dim"] == 50
    assert iso.meta["transforms"] == [{"op": "random_isometry", "D": 50, "seed": 1}]
    assert ds.meta["transforms"] == []  # input untouched
    with pytest.raises(ValueError):
        g.random_isometry(ds, 2, seed=0)


def test_add_noise_dims_keeps_structured_columns():
    ds = g.circle(300, 0, dim=3)
    nd = g.add_noise_dims(ds, 20, sigma=0.5, seed=2)
    assert nd.X.shape == (300, 20)
    np.testing.assert_array_equal(nd.X[:, :3], ds.X)
    assert np.std(nd.X[:, 3:]) == pytest.approx(0.5, rel=0.05)
    assert nd.meta["ambient_dim"] == 20
    assert nd.meta["transforms"][-1] == {
        "op": "add_noise_dims",
        "D": 20,
        "sigma": 0.5,
        "seed": 2,
    }
    with pytest.raises(ValueError):
        g.add_noise_dims(ds, 2, sigma=0.1, seed=0)


def test_transforms_leave_truth_alone_and_chain():
    ds = g.branching(200, 0)
    out = g.add_isotropic_noise(g.add_noise_dims(ds, 30, 0.1, seed=0), 0.2, seed=0)
    assert out.latent is ds.latent and out.clean is ds.clean and out.labels is ds.labels
    assert [t["op"] for t in out.meta["transforms"]] == [
        "add_noise_dims",
        "add_isotropic_noise",
    ]
    assert out.meta["ambient_dim"] == 30
    i = np.arange(200)
    np.testing.assert_array_equal(out.geodesic(i, i[::-1]), ds.geodesic(i, i[::-1]))
    # deterministic, and independent of the generator's own stream
    again = g.add_isotropic_noise(g.add_noise_dims(ds, 30, 0.1, seed=0), 0.2, seed=0)
    np.testing.assert_array_equal(out.X, again.X)


def test_transform_streams_differ_from_generator_stream():
    ds = g.circle(100, 0)
    noise = g.add_isotropic_noise(ds, 1.0, seed=0).X - ds.X
    assert not np.allclose(noise, np.random.default_rng(0).standard_normal((100, 2)))


def test_rotation_invariance_of_isotropic_noise():
    """add_noise_dims vs add_isotropic_noise(random_isometry): equal up to rotation.

    Rotating B = noise(Q x) back by [Q, Q_perp] gives (x + s z1, s z2): the component
    orthogonal to the structure is distributed exactly as add_noise_dims's extra
    columns, and the in-plane component is x plus isotropic noise.
    """
    ds = g.swiss_roll(2000, 0)
    D, s = 30, 0.5
    p = ds.X.shape[1]
    A = g.add_noise_dims(ds, D, s, seed=1).X
    iso = g.random_isometry(ds, D, seed=2)
    B = g.add_isotropic_noise(iso, s, seed=3).X

    QT = np.linalg.lstsq(ds.X, iso.X, rcond=None)[0]  # (p, D) = Q^T
    np.testing.assert_allclose(QT @ QT.T, np.eye(p), atol=1e-10)
    Qp = null_space(QT)  # (D, D - p)

    # off-structure part: same law as A's noise columns (singular values agree)
    sv_a = np.linalg.svd(A[:, p:], compute_uv=False)
    sv_b = np.linalg.svd(B @ Qp, compute_uv=False)
    np.testing.assert_allclose(sv_b, sv_a, rtol=0.06)
    # on-structure part: x + s * N(0, I_p), noise uncorrelated with x
    resid = B @ QT.T - ds.X
    assert np.std(resid) == pytest.approx(s, rel=0.03)
    assert np.abs(np.corrcoef(resid.ravel(), ds.X.ravel())[0, 1]) < 0.03

    # adding the same in-plane noise to A makes the two equal in distribution
    # up to rotation: their pairwise-distance quantiles agree
    A2 = g.add_noise_dims(g.add_isotropic_noise(ds, s, seed=4), D, s, seed=5).X
    sub = np.arange(0, 2000, 4)
    qs = np.linspace(0.01, 0.99, 25)
    qa = np.quantile(pdist(A2[sub]), qs)
    qb = np.quantile(pdist(B[sub]), qs)
    np.testing.assert_allclose(qb, qa, rtol=0.03)
