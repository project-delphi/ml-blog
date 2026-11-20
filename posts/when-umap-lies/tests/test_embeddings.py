"""Tests for umaplab.embeddings: every method, determinism, pca_dims, limits, defaults.

The data are built inline so these tests do not depend on umaplab.generators.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from umaplab import embeddings as E

N = 300


@pytest.fixture(scope="module")
def roll() -> np.ndarray:
    """A 300-point swiss roll in R^3."""
    rng = np.random.default_rng(0)
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, N)
    h = rng.uniform(0.0, 21.0, N)
    return np.c_[t * np.cos(t), h, t * np.sin(t)]


@pytest.fixture(scope="module")
def runs(roll: np.ndarray) -> dict[str, tuple[E.EmbedResult, ...]]:
    """Each method fitted three times: seed 0, seed 0 again, seed 1."""
    return {m: tuple(E.embed(m, roll, s) for s in (0, 0, 1)) for m in E.METHODS}


@pytest.mark.parametrize("method", list(E.METHODS))
def test_every_method_returns_finite_float32(method: str, runs: dict) -> None:
    """Every registry method returns a finite (n, 2) float32 embedding."""
    res = runs[method][0]
    assert res.Y.shape == (N, 2)
    assert res.Y.dtype == np.float32
    assert np.all(np.isfinite(res.Y))
    assert res.seconds >= 0
    assert set(E.METHODS[method].defaults) <= set(res.params)


@pytest.mark.parametrize("method", list(E.METHODS))
def test_same_seed_same_embedding(method: str, runs: dict) -> None:
    """Same seed, same output, for every method (trimap included, at n_jobs=1)."""
    a, b, _ = runs[method]
    np.testing.assert_array_equal(a.Y, b.Y)


@pytest.mark.parametrize("method", ["umap", "tsne", "pacmap", "trimap", "phate"])
def test_different_seed_different_embedding(method: str, runs: dict) -> None:
    """The stochastic methods actually use the seed."""
    a, _, c = runs[method]
    assert not np.allclose(a.Y, c.Y)


def test_trimap_determinism_record(runs: dict) -> None:
    """Trimap 1.2.0 has no seed; our seeding makes it reproducible on one thread."""
    a, b, _ = runs["trimap"]
    assert a.info["trimap_numba_threads"] == 1
    np.testing.assert_array_equal(a.Y, b.Y)


def test_pca_dims_path(roll: np.ndarray) -> None:
    """pca_dims below D pre-reduces (recorded); at or above D it is a no-op."""
    rng = np.random.default_rng(1)
    X = np.c_[roll, 0.1 * rng.standard_normal((N, 17))]
    res = E.embed("umap", X, 0, {"pca_dims": 5})
    assert res.params["pca_dims"] == 5
    assert 0 < res.info["pca_dims_explained_variance"] <= 1
    assert res.Y.shape == (N, 2) and np.all(np.isfinite(res.Y))
    assert not np.allclose(res.Y, E.embed("umap", X, 0).Y)
    full = E.embed("pca", X, 0, {"pca_dims": 50})
    assert "pca_dims_explained_variance" not in full.info


def test_infeasible_raised() -> None:
    """Isomap refuses n above its max_n before doing any work."""
    X = np.zeros((E.METHODS["isomap"].max_n + 1, 3))
    with pytest.raises(E.Infeasible):
        E.embed("isomap", X, 0)


def test_unknown_method() -> None:
    """An unknown method name is a KeyError."""
    with pytest.raises(KeyError):
        E.embed("nope", np.zeros((10, 3)), 0)


def test_overrides_merge_and_resolve(roll: np.ndarray) -> None:
    """Overrides replace defaults; data-dependent defaults are resolved and recorded."""
    res = E.embed("spectral", roll, 0)
    assert res.params["n_neighbors"] == N // 10
    res = E.embed("umap", roll, 0, {"n_neighbors": 5, "min_dist": 0.5})
    assert res.params["n_neighbors"] == 5 and res.params["min_dist"] == 0.5
    assert res.params["init"] == "spectral"
    res = E.embed("pca", roll, 0)
    assert res.params["svd_solver"] == "full"


def test_warnings_are_captured_not_printed(roll: np.ndarray, capsys) -> None:
    """Package warnings and log lines land in info, not on the console."""
    res = E.embed("pacmap", roll, 0)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert any("random state" in w for w in res.info.get("warnings", []))


def test_diffusion_map_of_a_circle_is_a_circle() -> None:
    """On a noise-free circle the 2-D diffusion map is close to a circle."""
    theta = np.random.default_rng(0).uniform(0.0, 2 * np.pi, 500)
    X = np.c_[np.cos(theta), np.sin(theta)]
    Y, lam = E.diffusion_map(X, k=15)
    r = np.linalg.norm(Y - Y.mean(axis=0), axis=1)
    assert r.std() / r.mean() < 0.05
    assert lam[0] == pytest.approx(1.0, abs=1e-8)
    assert lam[1] == pytest.approx(lam[2], rel=1e-2)  # the cos/sin pair


def test_diffusion_map_psi0_is_trivial() -> None:
    """The top eigenvalue is 1 and the embedding avoids the constant eigenvector."""
    X = np.random.default_rng(3).normal(size=(200, 4))
    Y, lam = E.diffusion_map(X)
    assert np.all(np.diff(lam) <= 1e-12)
    assert np.all(np.std(Y, axis=0) > 0)


# Where a registry default deliberately differs from the installed package's own
# signature default (each one is documented in the MethodSpec notes).
_OVERRIDES = {
    ("pca", "svd_solver"): "auto",
    ("umap", "init"): "auto",
    ("umap", "compatibility_layout"): None,
    ("tsne", "early_exaggeration"): "auto",
    ("isomap", "n_neighbors"): 5,
    ("spectral", "eigen_solver"): None,
}
_RENAMED = {("tsne", "init"): "initialization"}


def _signature_defaults(method: str) -> dict:
    import openTSNE
    import pacmap
    import phate
    import trimap
    import umap
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, SpectralEmbedding

    cls = {
        "pca": PCA,
        "umap": umap.UMAP,
        "tsne": openTSNE.TSNE,
        "pacmap": pacmap.PaCMAP,
        "trimap": trimap.TRIMAP,
        "isomap": Isomap,
        "spectral": SpectralEmbedding,
        "phate": phate.PHATE,
    }[method]
    sig = inspect.signature(cls.__init__).parameters
    return {k: v.default for k, v in sig.items()}


@pytest.mark.parametrize("method", [m for m in E.METHODS if m != "diffmap"])
def test_registry_defaults_match_installed_packages(method: str) -> None:
    """Recorded defaults equal the package signature defaults, bar documented changes.

    Guards against a package upgrade silently moving a default we claim to mirror.
    """
    sig = _signature_defaults(method)
    for key, value in E.METHODS[method].defaults.items():
        name = _RENAMED.get((method, key), key)
        if name not in sig:  # an argument of fit_transform (pacmap / trimap init)
            assert key == "init"
            continue
        expected = _OVERRIDES.get((method, key), value)
        got = sig[name]
        if isinstance(got, tuple):
            got = tuple(got)
        assert got == expected, f"{method}.{key}: package {got!r}, registry {value!r}"


def test_measure_memory_reports_growth() -> None:
    """The spawn-subprocess probe returns time and a fit-sized RSS growth."""
    X = np.random.default_rng(0).normal(size=(2000, 10))
    sec, mb = E.measure_memory("isomap", X, 0)
    assert sec > 0
    # Isomap holds at least one dense 2000 x 2000 float64 matrix (30.5 MB).
    assert 30 < mb < 2000
    sec, mb = E.measure_memory("pca", X, 0)
    assert 0 <= mb < 30


@pytest.mark.parametrize("method", sorted(E.METHODS))
def test_every_method_beats_chance_on_a_swiss_roll(method):
    """Guards against a silently broken backend (annoy once returned only the query
    point, leaving trimap at chance-level neighbour preservation)."""
    from umaplab.generators import swiss_roll
    from umaplab.metrics import knn_preservation

    ds = swiss_roll(600, 0)
    Y = E.embed(method, ds.X, 0).Y
    chance = 10 / (ds.X.shape[0] - 1)
    assert knn_preservation(ds.latent, Y, [10])[10] > 10 * chance
