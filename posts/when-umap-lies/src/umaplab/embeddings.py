"""One interface over every dimensionality-reduction method in the series.

Every method is called through :func:`embed`, which merges the caller's overrides into
the package defaults recorded in :data:`METHODS`, applies the optional ``pca_dims``
pre-reduction, times the fit, and captures every warning the package emits into
``EmbedResult.info`` instead of printing it.

The method packages are imported lazily, inside the fit functions. The runner sets
``NUMBA_NUM_THREADS`` and the BLAS thread variables in each worker before anything
numba-based is imported, and that only works if importing this module imports none
of them.

Parameter names follow each package, with one exception: the initialisation is always
``init`` (openTSNE calls it ``initialization``; it is renamed on the way in), so a
config label such as ``tsne·init-random`` means the same thing for every method.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import io
import logging
import multiprocessing as mp
import os
import resource
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

THREAD_ENV_VARS = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Above this many matrix entries PCA switches from the exact LAPACK SVD to the seeded
# randomized solver (n_components=2 is far below the rank, so the randomized solver
# with its default power iterations is accurate to float precision on these data).
PCA_FULL_MAX_ENTRIES = 2_000_000


class Infeasible(Exception):
    """Raised when a method is asked to run beyond its recorded feasibility limits."""


@dataclass(frozen=True)
class MethodSpec:
    """Registry entry: what a method is, its explicit defaults and its limits."""

    name: str
    family: str
    preserves: str
    defaults: dict[str, Any]
    complexity: str
    package: str
    max_n: int | None = None
    max_D: int | None = None
    seeded: bool = True
    notes: str = ""


@dataclass(frozen=True)
class EmbedResult:
    """A 2-D embedding with its fit time, effective params and diagnostics."""

    Y: np.ndarray
    seconds: float
    params: dict[str, Any]
    info: dict[str, Any] = field(default_factory=dict)


def set_thread_env(n: int = 1) -> None:
    """Pin numba / OpenMP / BLAS thread pools to ``n`` via the environment.

    Effective only for libraries not yet imported in this process, and for every
    child process started afterwards.
    """
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n)


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


@contextlib.contextmanager
def _numba_threads(n_jobs: int) -> Iterator[None]:
    """Run the block with numba's thread count set to ``n_jobs`` (restored after)."""
    import numba

    old = numba.get_num_threads()
    cap = numba.config.NUMBA_NUM_THREADS
    want = cap if n_jobs == -1 else max(1, min(n_jobs, cap))
    numba.set_num_threads(want)
    try:
        yield
    finally:
        numba.set_num_threads(old)


class _LogCapture(logging.Handler):
    """Collect WARNING+ log records emitted while a fit runs."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}: {record.getMessage()}")


@contextlib.contextmanager
def _captured(info: dict[str, Any]) -> Iterator[None]:
    """Capture warnings, WARNING+ log records and Python-level stdout/stderr into info.

    Package loggers are made non-propagating for the duration, so nothing reaches the
    console; output written by compiled code straight to the file descriptors is not
    interceptable this way and is left alone.
    """
    handler = _LogCapture()
    root = logging.getLogger()
    quiet = ["pacmap", "pacmap.pacmap", "phate", "graphtools", "tasklogger", "umap"]
    saved = []
    for name in quiet:
        lg = logging.getLogger(name)
        saved.append((lg, lg.propagate))
        lg.addHandler(handler)
        lg.propagate = False
    root.addHandler(handler)
    out, err = io.StringIO(), io.StringIO()
    try:
        with (
            warnings.catch_warnings(record=True) as caught,
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            warnings.simplefilter("always")
            yield
    finally:
        root.removeHandler(handler)
        for lg, prop in saved:
            lg.removeHandler(handler)
            lg.propagate = prop
        msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
        msgs += handler.messages
        # Deduplicate, keep order, cap the length so one noisy fit can't bloat a row.
        uniq = list(dict.fromkeys(m[:500] for m in msgs))
        if uniq:
            info["warnings"] = uniq[:20]
        text = (out.getvalue() + err.getvalue()).strip()
        if text:
            info["output"] = text[-2000:]


# --------------------------------------------------------------------------- fits
# Each fit takes (X, seed, params, n_jobs, info) and returns the (n, 2) embedding.
# ``params`` is the merged effective dict without ``pca_dims``; a fit may add resolved
# values (for example a data-dependent default) to it in place.


def _fit_pca(
    X: np.ndarray, seed: int, p: dict[str, Any], n_jobs: int, info: dict
) -> np.ndarray:
    from sklearn.decomposition import PCA

    if p.get("svd_solver") is None:
        p["svd_solver"] = "full" if X.size <= PCA_FULL_MAX_ENTRIES else "randomized"
    kw = {k: v for k, v in p.items()}
    model = PCA(n_components=2, random_state=seed, **kw)
    Y = model.fit_transform(X)
    info["explained_variance_ratio"] = [
        float(v) for v in model.explained_variance_ratio_
    ]
    return Y


def _fit_umap(
    X: np.ndarray, seed: int, p: dict[str, Any], n_jobs: int, info: dict
) -> np.ndarray:
    import umap

    with _numba_threads(n_jobs):
        model = umap.UMAP(n_components=2, random_state=seed, n_jobs=n_jobs, **p)
        Y = model.fit_transform(X.astype(np.float32))
    # The legacy (compatibility) layout is reproducible only single-threaded, so a
    # seed forces n_jobs=1 inside UMAP whatever we ask for.
    info["umap_n_jobs_effective"] = int(model.n_jobs)
    if p.get("n_epochs") is None and not p.get("densmap"):
        # umap-learn's rule (simplicial_set_embedding): 500 up to n=10,000, else 200.
        info["n_epochs_effective"] = 500 if X.shape[0] <= 10_000 else 200
    return Y


def _fit_tsne(
    X: np.ndarray, seed: int, p: dict[str, Any], n_jobs: int, info: dict
) -> np.ndarray:
    import openTSNE

    kw = dict(p)
    kw["initialization"] = kw.pop("init")
    model = openTSNE.TSNE(n_components=2, random_state=seed, n_jobs=n_jobs, **kw)
    emb = model.fit(X)
    info["kl_divergence"] = float(emb.kl_divergence)
    return np.asarray(emb)


def _fit_pacmap(
    X: np.ndarray, seed: int, p: dict[str, Any], n_jobs: int, info: dict
) -> np.ndarray:
    import pacmap

    kw = dict(p)
    init = kw.pop("init")
    if isinstance(kw.get("num_iters"), list):
        kw["num_iters"] = tuple(kw["num_iters"])
    with _numba_threads(n_jobs):
        model = pacmap.PaCMAP(n_components=2, random_state=seed, verbose=False, **kw)
        # PaCMAP centres and rescales X in place; hand it a private copy.
        return model.fit_transform(X.copy(), init=init)


_trimap_seeders: tuple[Callable[[int], None], Callable[[int], None]] | None = None


def _seed_numba(seed: int) -> None:
    """Seed numba's RNG on the calling thread and on every numba worker thread.

    ``np.random.seed`` from Python seeds only NumPy's generator; numba keeps a separate
    per-thread state, and trimap draws its random triplets inside ``prange`` loops,
    i.e. on the worker threads. Seeding each worker makes those draws reproducible
    provided numba runs one thread (scheduling decides which thread draws what).
    """
    global _trimap_seeders
    if _trimap_seeders is None:
        import numba

        @numba.njit(cache=False)
        def _main(s: int) -> None:  # pragma: no cover - compiled
            np.random.seed(s)

        @numba.njit(parallel=True, cache=False)
        def _workers(s: int) -> None:  # pragma: no cover - compiled
            for _ in numba.prange(numba.get_num_threads()):
                np.random.seed(s)

        _trimap_seeders = (_main, _workers)
    _trimap_seeders[0](seed)
    _trimap_seeders[1](seed)


def _fit_trimap(
    X: np.ndarray, seed: int, p: dict[str, Any], n_jobs: int, info: dict
) -> np.ndarray:
    import trimap
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import NearestNeighbors

    kw = dict(p)
    init = kw.pop("init")
    # annoy 1.17.3, trimap's kNN backend, is miscompiled in this environment: every
    # query returns only the query point, and trimap then reads uninitialised
    # neighbour slots (triplet weights ~1e6, an exploding layout at chance-level kNN
    # preservation). We bypass it with exact kNN through trimap's own knn_tuple path.
    # knn_tuple skips trimap's preprocessing, so it is reproduced here: TruncatedSVD to
    # 100 dims when D > 100 and apply_pca, else min-max rescale and centre. trimap
    # then initialises from PCA of this matrix, as its default path does.
    Xp = np.asarray(X, dtype=np.float32).copy()
    if kw.get("apply_pca", True) and Xp.shape[1] > 100:
        Xp -= Xp.mean(0)
        Xp = TruncatedSVD(n_components=100, random_state=0).fit_transform(Xp)
    else:
        Xp -= Xp.min()
        Xp /= Xp.max()
        Xp -= Xp.mean(0)
    Xp = Xp.astype(np.float32)
    n_extra = min(kw.get("n_inliers", 12) + 50, Xp.shape[0])
    dist, nbrs = NearestNeighbors(n_neighbors=n_extra).fit(Xp).kneighbors(Xp)
    kw["apply_pca"] = False
    info["trimap_knn"] = "exact-sklearn"
    with _numba_threads(n_jobs):
        # trimap 1.2.0 takes no seed. Seed the legacy NumPy generator (initial random
        # layouts) and numba's per-thread generators (triplet sampling) immediately
        # before the fit; deterministic only when numba runs a single thread.
        import numba

        np.random.seed(seed)
        _seed_numba(seed)
        info["trimap_numba_threads"] = int(numba.get_num_threads())
        model = trimap.TRIMAP(
            n_dims=2,
            verbose=False,
            knn_tuple=(nbrs.astype(np.int32), dist.astype(np.float32)),
            **kw,
        )
        return model.fit_transform(Xp, init=init)


def _fit_isomap(
    X: np.ndarray, seed: int, p: dict[str, Any], n_jobs: int, info: dict
) -> np.ndarray:
    from sklearn.manifold import Isomap

    model = Isomap(n_components=2, n_jobs=n_jobs, **p)
    return model.fit_transform(X)


def _fit_spectral(
    X: np.ndarray, seed: int, p: dict[str, Any], n_jobs: int, info: dict
) -> np.ndarray:
    from sklearn.manifold import SpectralEmbedding

    if p.get("n_neighbors") is None:
        # sklearn's own default, made explicit and recorded.
        p["n_neighbors"] = max(int(X.shape[0] / 10), 1)
    model = SpectralEmbedding(n_components=2, random_state=seed, n_jobs=n_jobs, **p)
    return model.fit_transform(X)


def diffusion_map(
    X: np.ndarray,
    *,
    k: int = 15,
    alpha: float = 1.0,
    t: float = 1.0,
    n_components: int = 2,
    seed: int = 0,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Diffusion map with a self-tuning kNN Gaussian kernel (Coifman & Lafon 2006).

    With :math:`d_{ij}` the Euclidean distance and :math:`\varepsilon_i` the distance
    from :math:`x_i` to its k-th nearest neighbour (self excluded):

    1. Kernel, on the symmetrised kNN graph (edge if either point is among the other's
       k neighbours, plus self loops):
       :math:`K_{ij} = \exp\!\big(-d_{ij}^2 / (\varepsilon_i \varepsilon_j)\big)`.
    2. Density normalisation:
       :math:`q_i = \sum_j K_{ij}`,
       :math:`K^{(\alpha)}_{ij} = K_{ij} / (q_i^\alpha q_j^\alpha)`. With
       :math:`\alpha = 1` the limit operator is the Laplace–Beltrami operator whatever
       the sampling density.
    3. Markov normalisation: :math:`d_i = \sum_j K^{(\alpha)}_{ij}`,
       :math:`P = D^{-1} K^{(\alpha)}`.
    4. :math:`P` is similar to the symmetric
       :math:`A = D^{-1/2} K^{(\alpha)} D^{-1/2}`, which ``eigsh`` handles stably.
       With :math:`A \phi_j = \lambda_j \phi_j` (orthonormal :math:`\phi_j`), the right
       eigenvectors of :math:`P` are :math:`\psi_j = D^{-1/2} \phi_j`, rescaled by
       :math:`\sqrt{\sum_i d_i}` so the trivial :math:`\psi_0 \equiv 1` and the
       :math:`\psi_j` are orthonormal under the stationary distribution
       :math:`\pi_i = d_i / \sum d`.
    5. Embedding: :math:`\Psi_t(x_i) = (\lambda_1^t \psi_1(i), \ldots,
       \lambda_m^t \psi_m(i))`, the top non-trivial eigenpairs.

    Returns:
        The (n, n_components) embedding and the eigenvalues
        :math:`\lambda_0 \ge \lambda_1 \ge \ldots` (trivial one first).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    if k >= n:
        raise ValueError(f"diffmap needs k < n (k={k}, n={n})")
    algorithm = "brute" if X.shape[1] > 20 else "auto"
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm, n_jobs=n_jobs).fit(X)
    dist, idx = nn.kneighbors(X)
    # Column 0 is the point itself (distance 0), so column k is the k-th neighbour with
    # self excluded. Self edges are kept: they are the K_ii = 1 diagonal. A zero
    # bandwidth (more than k exact duplicates) is replaced by the smallest positive one.
    eps = dist[:, k]
    eps = np.where(eps > 0, eps, np.min(eps[eps > 0]) if np.any(eps > 0) else 1.0)
    rows = np.repeat(np.arange(n), k + 1)
    cols = idx.ravel()
    vals = np.exp(-(dist.ravel() ** 2) / (eps[rows] * eps[cols]))
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)  # symmetric: the kernel value is the same either way round
    W.setdiag(1.0)
    q = np.asarray(W.sum(axis=1)).ravel()
    qa = q**-alpha
    Ka = sp.diags(qa) @ W @ sp.diags(qa)
    d = np.asarray(Ka.sum(axis=1)).ravel()
    dm = 1.0 / np.sqrt(d)
    A = sp.diags(dm) @ Ka @ sp.diags(dm)
    A = ((A + A.T) * 0.5).tocsr()
    v0 = np.random.default_rng(seed).uniform(0.5, 1.5, size=n)
    m = n_components + 1
    lam, phi = eigsh(A, k=m, which="LA", v0=v0)
    order = np.argsort(lam)[::-1]
    lam, phi = lam[order], phi[:, order]
    psi = phi * dm[:, None] * np.sqrt(d.sum())
    # Fix each eigenvector's sign so the result is deterministic.
    for j in range(m):
        if psi[np.argmax(np.abs(psi[:, j])), j] < 0:
            psi[:, j] *= -1
    Y = psi[:, 1:m] * (np.clip(lam[1:m], 0.0, None) ** t)
    return Y, lam


def _fit_diffmap(
    X: np.ndarray, seed: int, p: dict[str, Any], n_jobs: int, info: dict
) -> np.ndarray:
    unknown = set(p) - set(METHODS["diffmap"].defaults)
    if unknown:
        raise TypeError(f"diffmap got unexpected params {sorted(unknown)}")
    Y, lam = diffusion_map(X, seed=seed, n_jobs=n_jobs, **p)
    info["eigenvalues"] = [float(v) for v in lam]
    return Y


def _fit_phate(
    X: np.ndarray, seed: int, p: dict[str, Any], n_jobs: int, info: dict
) -> np.ndarray:
    import phate

    model = phate.PHATE(
        n_components=2, random_state=seed, n_jobs=n_jobs, verbose=0, **p
    )
    Y = model.fit_transform(X)
    if model.optimal_t is not None:
        info["t_optimal"] = int(model.optimal_t)
    return Y


# Imported before the timer starts, so ``seconds`` never includes an import.
_IMPORTS: dict[str, tuple[str, ...]] = {
    "pca": ("sklearn.decomposition",),
    "umap": ("umap", "numba"),
    "tsne": ("openTSNE",),
    "pacmap": ("pacmap", "numba"),
    "trimap": ("trimap", "numba"),
    "isomap": ("sklearn.manifold",),
    "spectral": ("sklearn.manifold",),
    "diffmap": ("scipy.sparse.linalg", "sklearn.neighbors"),
    "phate": ("phate",),
}

_FITS: dict[str, Callable[..., np.ndarray]] = {
    "pca": _fit_pca,
    "umap": _fit_umap,
    "tsne": _fit_tsne,
    "pacmap": _fit_pacmap,
    "trimap": _fit_trimap,
    "isomap": _fit_isomap,
    "spectral": _fit_spectral,
    "diffmap": _fit_diffmap,
    "phate": _fit_phate,
}

# ----------------------------------------------------------------------- registry
# Defaults are the installed packages' own defaults, read from their signatures and
# source, and passed explicitly so a package upgrade cannot change a result silently.
# Where we deliberately depart from the package, the entry's ``notes`` says so.

METHODS: dict[str, MethodSpec] = {
    "pca": MethodSpec(
        name="pca",
        family="linear",
        preserves="directions of maximal variance (global linear structure)",
        defaults={"svd_solver": None, "whiten": False},
        complexity="O(n D min(n, D)) full SVD; O(n D) per pass randomized",
        package=f"scikit-learn {_version('scikit-learn')}",
        seeded=False,
        notes=(
            "svd_solver=None resolves to 'full' when n*D <= "
            f"{PCA_FULL_MAX_ENTRIES:,} and to seeded 'randomized' above; the resolved "
            "solver is recorded in params. sklearn's own default is 'auto'."
        ),
    ),
    "umap": MethodSpec(
        name="umap",
        family="neighbour-graph SGD",
        preserves="the fuzzy kNN graph (local neighbourhoods), via cross-entropy",
        defaults={
            "n_neighbors": 15,
            "min_dist": 0.1,
            "spread": 1.0,
            "init": "spectral",
            "metric": "euclidean",
            "n_epochs": None,
            "learning_rate": 1.0,
            "negative_sample_rate": 5,
            "repulsion_strength": 1.0,
            "local_connectivity": 1.0,
            "set_op_mix_ratio": 1.0,
            "densmap": False,
            "compatibility_layout": True,
        },
        complexity="O(n^1.14 D) NN-descent kNN (exact below n=4096) + O(n k) per epoch",
        package=f"umap-learn {_version('umap-learn')}",
        notes=(
            "umap-learn 0.6.0 defaults init='auto' and compatibility_layout=None, which "
            "resolve to init='spectral' on the legacy 0.5.x pipeline (with a "
            "FutureWarning that the default will move to recursive init + Adam). We pin "
            "that resolution explicitly. random_state forces n_jobs=1 on this pipeline. "
            "n_epochs=None means 500 for small data, 200 for large."
        ),
    ),
    "tsne": MethodSpec(
        name="tsne",
        family="neighbour-graph gradient descent (KL)",
        preserves="perplexity-calibrated neighbour probabilities (local structure)",
        defaults={
            "perplexity": 30,
            "init": "pca",
            "learning_rate": "auto",
            "early_exaggeration": 12,
            "early_exaggeration_iter": 250,
            "n_iter": 500,
            "exaggeration": None,
            "dof": 1,
            "theta": 0.5,
            "metric": "euclidean",
            "neighbors": "auto",
            "negative_gradient_method": "auto",
        },
        complexity="O(n D k) kNN (k = 3*perplexity) + O(n log n) BH / O(n) FFT per iter",
        package=f"openTSNE {_version('openTSNE')}",
        notes=(
            "learning_rate='auto' is n / exaggeration in each phase (n/12 during early "
            "exaggeration, n after). openTSNE's default early_exaggeration is 'auto' = "
            "max(12, exaggeration); pinned to 12 here. negative_gradient_method 'auto' "
            "is Barnes-Hut below n=10,000 and FFT from 10,000. 'init' is passed as "
            "openTSNE's 'initialization'."
        ),
    ),
    "pacmap": MethodSpec(
        name="pacmap",
        family="neighbour-graph SGD (pair sampling)",
        preserves="neighbour, mid-near and further pairs (local and some global)",
        defaults={
            "n_neighbors": 10,
            "MN_ratio": 0.5,
            "FP_ratio": 2.0,
            "init": "pca",
            "lr": 1.0,
            "num_iters": (100, 100, 250),
            "distance": "euclidean",
            "apply_pca": True,
            "knn_backend": "faiss",
        },
        complexity="O(n D) approximate kNN (faiss HNSW) + O(n k) per iteration",
        package=f"pacmap {_version('pacmap')}",
        notes=(
            "fit_transform's init=None means 'pca'. apply_pca=True reduces D>100 to 100 "
            "by TruncatedSVD inside pacmap; below that X is min-max scaled. kNN is "
            "approximate (faiss HNSW)."
        ),
    ),
    "trimap": MethodSpec(
        name="trimap",
        family="triplet SGD",
        preserves="relative order of distances within sampled triplets",
        defaults={
            "n_inliers": 12,
            "n_outliers": 4,
            "n_random": 3,
            "distance": "euclidean",
            "lr": 0.1,
            "n_iters": 400,
            "weight_temp": 0.5,
            "apply_pca": True,
            "opt_method": "dbd",
            "init": "pca",
        },
        complexity="O(n D log n) Annoy kNN + O(n (k_in k_out + n_random)) per iteration",
        package=f"trimap {_version('trimap')}",
        notes=(
            "No seed argument in 1.2.0. embed seeds np.random and numba's per-thread "
            "generators immediately before the fit; this is reproducible only with one "
            "numba thread (n_jobs=1). apply_pca reduces D>100 to 100 with a fixed seed."
        ),
    ),
    "isomap": MethodSpec(
        name="isomap",
        family="spectral (geodesic MDS)",
        preserves="graph-geodesic distances on the kNN graph",
        defaults={
            "n_neighbors": 10,
            "eigen_solver": "auto",
            "path_method": "auto",
            "neighbors_algorithm": "auto",
            "metric": "minkowski",
            "p": 2,
        },
        complexity="O(n^2 log n) shortest paths, O(n^2) memory, eigensolve of n x n",
        package=f"scikit-learn {_version('scikit-learn')}",
        max_n=5000,
        seeded=False,
        notes="n_neighbors=10 is our choice; sklearn's default is 5.",
    ),
    "spectral": MethodSpec(
        name="spectral",
        family="spectral (Laplacian eigenmaps)",
        preserves="kNN-graph adjacency (smoothest Laplacian eigenvectors)",
        defaults={
            "affinity": "nearest_neighbors",
            "n_neighbors": None,
            "eigen_solver": "arpack",
            "gamma": None,
        },
        complexity="O(n D k) kNN + sparse eigensolve; default k = n/10 so O(n^2) edges",
        package=f"scikit-learn {_version('scikit-learn')}",
        notes=(
            "n_neighbors=None is sklearn's max(int(n/10), 1), resolved and recorded in "
            "params. eigen_solver=None means 'arpack' in sklearn."
        ),
    ),
    "diffmap": MethodSpec(
        name="diffmap",
        family="spectral (diffusion)",
        preserves="diffusion distance at time t (random-walk connectivity)",
        defaults={"k": 15, "alpha": 1.0, "t": 1.0},
        complexity="O(n^2 D) exact kNN + O(n k) per Lanczos iteration",
        package=f"umaplab (scipy {_version('scipy')})",
        notes="Own implementation; see diffusion_map for the formula.",
    ),
    "phate": MethodSpec(
        name="phate",
        family="diffusion + MDS",
        preserves="potential distances of the diffusion operator (global trajectories)",
        defaults={
            "knn": 5,
            "decay": 40,
            "t": "auto",
            "n_landmark": 2000,
            "gamma": 1,
            "n_pca": 100,
            "mds_solver": "sgd",
            "mds": "metric",
            "knn_dist": "euclidean",
            "mds_dist": "euclidean",
        },
        complexity="O(n D k) kNN + landmark diffusion O(n L) + MDS on L=2000 landmarks",
        package=f"phate {_version('phate')}",
    ),
}


def effective_params(method: str, params: dict | None = None) -> dict[str, Any]:
    """The registry defaults for ``method`` merged with ``params``, plus ``pca_dims``."""
    spec = METHODS[method]
    merged: dict[str, Any] = {"pca_dims": None, **spec.defaults}
    merged.update(params or {})
    return merged


def check_feasible(method: str, n: int, D: int) -> None:
    """Raise :class:`Infeasible` if (n, D) exceeds the method's recorded limits."""
    spec = METHODS[method]
    if spec.max_n is not None and n > spec.max_n:
        raise Infeasible(f"{method}: n={n} exceeds max_n={spec.max_n}")
    if spec.max_D is not None and D > spec.max_D:
        raise Infeasible(f"{method}: D={D} exceeds max_D={spec.max_D}")


def embed(
    method: str,
    X: np.ndarray,
    seed: int,
    params: dict | None = None,
    n_jobs: int = 1,
) -> EmbedResult:
    """Embed X into 2-D with ``method``; see the module docstring for the contract.

    Raises:
        KeyError: unknown method.
        Infeasible: (n, D) beyond the method's limits.
        RuntimeError: the method returned a non-finite or mis-shaped embedding.
    """
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; known: {sorted(METHODS)}")
    X = np.asarray(X, dtype=np.float64)
    n, D = X.shape
    check_feasible(method, n, D)
    merged = effective_params(method, params)
    pca_dims = merged.pop("pca_dims")
    info: dict[str, Any] = {}
    fit_params = dict(merged)

    with _captured(info):
        for mod in ("sklearn.decomposition", *_IMPORTS[method]):
            importlib.import_module(mod)
        t0 = time.perf_counter()
        Z = X
        if pca_dims is not None and pca_dims < D:
            from sklearn.decomposition import PCA

            pre = PCA(n_components=pca_dims, svd_solver="randomized", random_state=seed)
            Z = pre.fit_transform(X)
            info["pca_dims_explained_variance"] = float(
                pre.explained_variance_ratio_.sum()
            )
        Y = _FITS[method](np.array(Z, dtype=np.float64), seed, fit_params, n_jobs, info)
        seconds = time.perf_counter() - t0

    Y = np.asarray(Y, dtype=np.float32)
    if Y.shape != (n, 2):
        raise RuntimeError(f"{method} returned shape {Y.shape}, expected {(n, 2)}")
    if not np.all(np.isfinite(Y)):
        raise RuntimeError(
            f"{method} returned {int((~np.isfinite(Y)).sum())} non-finite"
        )
    eff = {"pca_dims": pca_dims, **fit_params}
    return EmbedResult(Y=Y, seconds=seconds, params=eff, info=info)


# Sizes that exercise every compiled code path: umap switches from exact kNN to
# NN-descent above n=4096, and compiling NN-descent costs ~14 s on first use.
_WARMUP_SIZES: dict[str, tuple[int, ...]] = {"umap": (300, 4200)}


def warmup_sizes(method: str, n: int | None = None) -> tuple[int, ...]:
    """The warm-up sample sizes that cover the code paths a fit of size n can take."""
    sizes = _WARMUP_SIZES.get(method, (300,))
    if n is None:
        return sizes
    return tuple(s for s in sizes if s == sizes[0] or s <= n)


def warm_up(
    method: str,
    params: dict | None = None,
    n_jobs: int = 1,
    sizes: tuple[int, ...] | None = None,
) -> None:
    """Import ``method``'s package and compile its numba kernels on throwaway data.

    First calls in a fresh process pay import and JIT costs (5 s for umap at n=300,
    20 s at n=5000 against 6 s warm) that are not the algorithm's; the runner and the
    memory probe call this once per process so ``seconds`` measures the fit alone.
    ``sizes`` defaults to every size in :func:`warmup_sizes`. Failures are ignored:
    the real fit will report them.
    """
    p = {k: v for k, v in (params or {}).items() if k != "pca_dims"}
    for n in sizes if sizes is not None else warmup_sizes(method):
        X = np.random.default_rng(0).normal(size=(n, 10))
        try:
            embed(method, X, 0, p, n_jobs=n_jobs)
        except Exception:  # noqa: BLE001, S110 - warm-up only
            pass


# ------------------------------------------------------------------ memory probe


def _maxrss_mb() -> float:
    """Peak resident set size of this process so far, in MB (macOS bytes, Linux KB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


class _RssSampler:
    """Background thread recording the peak *current* RSS (psutil) while active."""

    def __init__(self, interval: float = 0.002) -> None:
        import threading

        try:
            import psutil
        except ImportError:  # pragma: no cover - psutil ships with the venv
            self._proc = None
        else:
            self._proc = psutil.Process()
        self.interval = interval
        self.peak_mb = self.now_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def now_mb(self) -> float:
        """Current RSS in MB, or NaN without psutil."""
        if self._proc is None:
            return float("nan")
        return self._proc.memory_info().rss / 2**20

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_mb = max(self.peak_mb, self.now_mb())

    def __enter__(self) -> _RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, self.now_mb())


def _memory_child(conn: Any, method: str, X: np.ndarray, seed: int, params: dict | None,
                  warmup: bool) -> None:  # fmt: skip
    """Child body for :func:`measure_memory`: warm up, take a baseline, fit, report.

    ``ru_maxrss`` is a high-water mark that cannot be reset, and the warm-up (numba
    compilation especially) can push it above what a small fit ever needs. So the
    baseline is the *current* RSS just before the fit, and the peak is ``ru_maxrss``
    after the fit when the fit set a new high-water mark (then it is exact), else the
    peak current RSS sampled every 2 ms during the fit.
    """
    import gc

    try:
        set_thread_env(1)
        if warmup:
            warm_up(method, params, sizes=warmup_sizes(method, X.shape[0]))
        gc.collect()
        sampler = _RssSampler()
        base_now = sampler.now_mb()
        base_hw = _maxrss_mb()
        if np.isnan(base_now):
            base_now = base_hw
        with sampler:
            res = embed(method, X, seed, params)
        hw = _maxrss_mb()
        peak = hw if hw > base_hw else sampler.peak_mb
        if np.isnan(peak):
            peak = hw
        conn.send(("ok", res, max(0.0, peak - base_now)))
    except BaseException as exc:  # noqa: BLE001 - report every failure to the parent
        conn.send(("error", f"{type(exc).__name__}: {exc}", None))
    finally:
        conn.close()


def embed_in_subprocess(
    method: str,
    X: np.ndarray,
    seed: int,
    params: dict | None = None,
    *,
    warmup: bool = True,
    timeout: float | None = None,
) -> tuple[EmbedResult, float]:
    """Run :func:`embed` in a fresh spawn process; return the result and peak RSS growth.

    The growth is the child's peak RSS during the fit minus its RSS just before it,
    after imports, data load and (by default) a warm-up fit, so it charges the fit
    alone; see ``_memory_child`` for how the peak is read.
    """
    ctx = mp.get_context("spawn")
    saved = {v: os.environ.get(v) for v in THREAD_ENV_VARS}
    set_thread_env(1)  # inherited by the child before its first import
    try:
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_memory_child,
            args=(child, method, np.asarray(X, dtype=np.float64), seed, params, warmup),
        )
        proc.start()
        child.close()
        if not parent.poll(timeout):
            proc.kill()
            proc.join()
            raise TimeoutError(f"{method} exceeded {timeout}s in the memory probe")
        try:
            status, payload, growth = parent.recv()
        except EOFError as exc:
            proc.join()
            raise RuntimeError(
                f"memory probe for {method} died (exit code {proc.exitcode})"
            ) from exc
        proc.join()
    finally:
        for var, val in saved.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val
    if status != "ok":
        raise RuntimeError(payload)
    return payload, float(growth)


def measure_memory(
    method: str, X: np.ndarray, seed: int, params: dict | None = None
) -> tuple[float, float]:
    """Fit in a fresh spawn subprocess; return (seconds, peak_rss_mb growth of the fit)."""
    res, growth = embed_in_subprocess(method, X, seed, params)
    return res.seconds, growth
