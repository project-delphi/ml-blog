"""Every experiment in the series, as a list of benchmark cells.

Each function returns the cells for one experiment; ``src/run_experiments.py`` runs
them into ``cache/<experiment>/``. The design choices that the posts quote live here,
next to the code that enacts them:

- n = 2000 points everywhere except the scaling study and the real data.
- ``data_seed`` indexes an independent draw of the dataset; in the sweeps the method's
  seed equals it, so one run per draw estimates total variability. ``stability``
  crosses three draws with five method seeds to split the two sources.
- Noise is per-coordinate Gaussian noise of sd ``sigma`` on every added column. Each
  base dataset gets its own ``sigma0``, chosen so that the noise index eta (below)
  equals 1 at D - p = 300, so the transition falls inside the tested range for every
  dataset and the datasets are comparable on one axis.

The noise index. Appending D - p noise columns adds to every squared distance a term
with mean 2 sigma^2 (D - p), the same for every pair, and standard deviation about
sigma^2 sqrt(8 (D - p)). Only the fluctuation can reorder neighbours, so the input
kNN graph should degrade as a function of

    eta = sigma^2 sqrt(8 (D - p)) / r_k^2,

with r_k the median distance to the k-th neighbour (k = 10) in the clean data.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from .benchmark import Cell
from .generators import GENERATORS

N = 2000
SEEDS5 = tuple(range(5))
SEEDS3 = tuple(range(3))
D_GRID = (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
ETA_K = 10
ETA_ANCHOR = 300  # D - p at which eta = 1 for sigma0

# The core comparison: package defaults, plus the PCA-50 pipeline arm for the two
# methods that have no PCA step of their own (PaCMAP, TriMap and PHATE already reduce
# to 100 PCs by default when D > 100).
CORE: tuple[tuple[str, dict], ...] = (
    ("pca", {}),
    ("umap", {}),
    ("tsne", {}),
    ("pacmap", {}),
    ("trimap", {}),
    ("isomap", {}),
    ("spectral", {}),
    ("diffmap", {}),
    ("umap", {"pca_dims": 50}),
    ("tsne", {"pca_dims": 50}),
)
PHATE = (("phate", {}),)
SHOWN = {"pca", "umap", "tsne", "umap·pca50"}  # configs whose embeddings widgets use

# The three base structures for the ambient-dimension sweeps (p = base dims).
SWEEP_BASES: dict[str, dict] = {
    "swiss_roll": {},  # p = 3, intrinsic 2
    "gaussian_clusters": {},  # p = 10, ten clusters
    "branching": {"noise": 0.0},  # p = 10, a trunk and three branches
}


def _label(method: str, params: dict) -> str:
    from .benchmark import config_label

    return config_label(method, params)


@cache
def sigma0(generator: str, params_items: tuple = ()) -> float:
    """Noise sd giving eta = 1 at D - p = ETA_ANCHOR for this base (seed 0 draw).

    r_k is measured on the data as generated (X), so a base with noise of its own (the
    topology circle) gets added noise on the scale of its real neighbourhoods. For
    the noise-free sweep bases X is the clean data, so their sigma is unchanged.
    """
    from .metrics import local_radius

    ds = GENERATORS[generator](N, 0, **dict(params_items))
    rk2 = float(np.median(local_radius(ds.X, ETA_K)) ** 2)
    return float(np.sqrt(rk2 / np.sqrt(8 * ETA_ANCHOR)))


def eta(sigma: float, D: int, p: int, rk2: float) -> float:
    """The noise index for sigma per added column, D total dims, base p dims."""
    return sigma**2 * np.sqrt(8 * max(D - p, 0)) / rk2


def _base_dim(generator: str, params: dict) -> int:
    return GENERATORS[generator](20, 0, **params).X.shape[1]


def _cells(
    experiment: str,
    generator: str,
    gen_params: dict,
    transforms: tuple[dict, ...],
    methods,
    seeds,
    *,
    n: int = N,
    keep=lambda label, seed: False,
    algo_seeds=None,
) -> list[Cell]:
    out = []
    for method, mp in methods:
        label = _label(method, mp)
        for s in seeds:
            for a in algo_seeds if algo_seeds is not None else (s,):
                out.append(
                    Cell(
                        experiment=experiment,
                        generator=generator,
                        gen_params=dict(gen_params),
                        transforms=tuple(transforms),
                        n=n,
                        data_seed=s,
                        method=method,
                        method_params=dict(mp),
                        algo_seed=a,
                        keep_embedding=keep(label, s),
                    )
                )
    return out


def _keep_shown(label: str, seed: int) -> bool:
    return seed == 0 and label in SHOWN


# --------------------------------------------------------------------------- Part 2


def noise_dims() -> list[Cell]:
    """Fixed intrinsic structure, ambient dimension raised by pure-noise columns."""
    cells = []
    for gen, gp in SWEEP_BASES.items():
        p = _base_dim(gen, gp)
        s0 = sigma0(gen, tuple(sorted(gp.items())))
        methods = CORE + (PHATE if gen == "branching" else ())
        for D in D_GRID:
            if D < p:
                continue
            tr = ({"op": "add_noise_dims", "D": D, "sigma": s0},)
            cells += _cells(
                "noise_dims", gen, gp, tr, methods, SEEDS5, keep=_keep_shown
            )
    return cells


def noise_sigma() -> list[Cell]:
    """Half and double sigma0, to test whether eta is the right axis."""
    cells = []
    methods = (("pca", {}), ("umap", {}), ("tsne", {}), ("pacmap", {}))
    methods += (("umap", {"pca_dims": 50}),)
    for gen, gp in SWEEP_BASES.items():
        p = _base_dim(gen, gp)
        s0 = sigma0(gen, tuple(sorted(gp.items())))
        for mult in (0.5, 2.0):
            for D in D_GRID:
                if D <= p:
                    continue
                tr = ({"op": "add_noise_dims", "D": D, "sigma": s0 * mult},)
                cells += _cells("noise_sigma", gen, gp, tr, methods, SEEDS3)
    return cells


def isometry() -> list[Cell]:
    """Control: the same geometry rotated into D dims. Every distance is unchanged."""
    cells = []
    for gen in ("swiss_roll", "gaussian_clusters"):
        p = _base_dim(gen, {})
        for D in D_GRID:
            if D < p:
                continue
            tr = ({"op": "random_isometry", "D": D},)
            cells += _cells("isometry", gen, {}, tr, CORE, SEEDS3)
    return cells


INTRINSIC_D = (2, 5, 10, 20, 50, 100)
INTRINSIC_AMBIENT = 200


def intrinsic() -> list[Cell]:
    """Genuinely more degrees of freedom, at a fixed ambient dimension of 200."""
    cells = []
    for d in INTRINSIC_D:
        rot = ({"op": "random_isometry", "D": INTRINSIC_AMBIENT},)
        cells += _cells(
            "intrinsic", "latent_factor", {"d": d, "D": d}, rot, CORE, SEEDS5
        )
        cells += _cells("intrinsic", "sphere", {"m": d - 1}, rot, CORE, SEEDS5)
        cells += _cells(
            "intrinsic",
            "gaussian_clusters",
            {"dim": d, "n_clusters": 10},
            rot,
            CORE,
            SEEDS5,
        )
    return cells


def scaling_grid() -> list[tuple[str, dict, int, int]]:
    """(method, params, n, seed) for the runtime and memory study (not a Cell grid)."""
    ns = (500, 1000, 2000, 5000, 10_000)
    methods = CORE[:8] + PHATE
    return [(m, mp, n, s) for m, mp in methods for n in ns for s in SEEDS3]


SIGNAL_D = (2, 5, 10, 20)
SIGNAL_AMBIENT = (20, 100, 1000)
SIGNAL_SIGMA = (0.0, 0.15, 0.3, 0.6)
SIGNAL_N = (500, 2000)


def signal_widget() -> list[Cell]:
    """The noise-versus-signal widget: six clusters in d dims, padded with noise."""
    cells = []
    methods = (("pca", {}), ("umap", {}), ("tsne", {}))
    for d in SIGNAL_D:
        for D in SIGNAL_AMBIENT:
            for sg in SIGNAL_SIGMA:
                tr = ({"op": "add_noise_dims", "D": D, "sigma": sg},)
                for n in SIGNAL_N:
                    cells += _cells(
                        "signal_widget",
                        "gaussian_clusters",
                        {"dim": d, "n_clusters": 6},
                        tr,
                        methods,
                        (0,),
                        n=n,
                        keep=lambda label, seed: True,
                    )
    return cells


# --------------------------------------------------------------------------- Part 3

TUNE_BASES: dict[str, dict] = {
    "swiss_roll": {},
    "gaussian_clusters": {"imbalance": 4.0},
    "branching": {"levels": 2, "noise": 0.03},
}

# Every method gets a grid of about the same size around its defaults.
TUNE_GRIDS: dict[str, list[dict]] = {
    "umap": [
        {"n_neighbors": nn, "min_dist": md}
        for nn in (5, 15, 50, 200)
        for md in (0.1, 0.5)
    ]
    + [{"init": "random"}, {"init": "pca"}],
    "tsne": [
        {"perplexity": pp, "exaggeration": ex}
        for pp in (5, 30, 100, 300)
        for ex in (1.0, 4.0)
    ]
    + [{"init": "random"}, {"early_exaggeration": 4.0}, {"early_exaggeration": 36.0}],
    "pacmap": [
        {"n_neighbors": nn, "MN_ratio": mn}
        for nn in (5, 10, 30, 60)
        for mn in (0.5, 2.0)
    ]
    + [{"init": "random"}],
    "trimap": [
        {"n_inliers": ni, "n_outliers": no} for ni in (5, 10, 20, 40) for no in (3, 8)
    ],
    "isomap": [{"n_neighbors": k} for k in (5, 10, 20, 50)],
    "spectral": [{"n_neighbors": k} for k in (5, 15, 50, 200)],
    "diffmap": [{"k": k, "t": t} for k in (5, 15, 50) for t in (1, 4)],
    "phate": [{"knn": k} for k in (3, 5, 15, 30)],
}


def tuning() -> list[Cell]:
    """Hyperparameter grids: select on data seed 0, evaluate on seeds 1 and 2."""
    cells = []
    for gen, gp in TUNE_BASES.items():
        methods = [(m, g) for m, grid in TUNE_GRIDS.items() for g in grid]
        methods += [(m, {}) for m in TUNE_GRIDS] + [("pca", {})]
        cells += _cells("tuning", gen, gp, (), methods, SEEDS3)
    return cells


def stability() -> list[Cell]:
    """Three data draws x five method seeds, embeddings kept for seed-pair metrics."""
    cells = []
    methods = [(m, {}) for m in ("umap", "tsne", "pacmap", "trimap", "phate")]
    methods += [("umap", {"init": "random"}), ("tsne", {"init": "random"})]
    for gen, gp in (("swiss_roll", {}), ("gaussian_clusters", {"imbalance": 4.0})):
        cells += _cells(
            "stability",
            gen,
            gp,
            (),
            methods,
            SEEDS3,
            algo_seeds=SEEDS5,
            keep=lambda label, seed: True,
        )
    return cells


UMAP_NN = (5, 10, 15, 30, 50, 100, 200)
UMAP_MD = (0.0, 0.1, 0.25, 0.5, 0.8, 0.99)


def umap_widget() -> list[Cell]:
    """The UMAP-parameter widget grid, on 1000 points of two datasets."""
    cells = []
    for gen, gp in (("gaussian_clusters", {"imbalance": 4.0}), ("swiss_roll", {})):
        for nn in UMAP_NN:
            for md in UMAP_MD:
                cells += _cells(
                    "umap_widget",
                    gen,
                    gp,
                    (),
                    (("umap", {"n_neighbors": nn, "min_dist": md}),),
                    (0,),
                    n=1000,
                    algo_seeds=SEEDS3,
                    keep=lambda label, seed: True,
                )
    return cells


# --------------------------------------------------------------------------- Part 4

OVERLAP_DELTA = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SMALL_NBHD = (("umap", {"n_neighbors": 5}), ("tsne", {"perplexity": 5}))


def overlap() -> list[Cell]:
    """Two Gaussians whose means are delta apart in 10 dims (delta = 0: one Gaussian)."""
    cells = []
    for delta in OVERLAP_DELTA:
        cells += _cells(
            "overlap",
            "overlapping",
            {"dim": 10, "delta": delta},
            (),
            CORE[:8] + SMALL_NBHD,
            SEEDS5,
            keep=lambda label, seed: seed == 0,
        )
    return cells


DENSITY_RATIO = (1.0, 3.0, 10.0, 30.0)


def density() -> list[Cell]:
    """A flat strip sampled with density varying by density_ratio along its length."""
    cells = []
    methods = CORE[:8] + PHATE + (("umap", {"densmap": True}),)
    for r in DENSITY_RATIO:
        cells += _cells(
            "density",
            "variable_density",
            {"density_ratio": r},
            (),
            methods,
            SEEDS5,
            keep=lambda label, seed: seed == 0,
        )
    return cells


def topology() -> list[Cell]:
    """Circles (clean and noise-padded), a sphere, and one- and two-level trees."""
    cells = []
    methods = (
        CORE[:8] + PHATE + (("umap", {"init": "random"}), ("tsne", {"init": "random"}))
    )
    keep = lambda label, seed: seed == 0  # noqa: E731
    cells += _cells(
        "topology", "circle", {"noise": 0.02}, (), methods, SEEDS5, keep=keep
    )
    s = sigma0("circle", (("noise", 0.02),))
    cells += _cells(
        "topology",
        "circle",
        {"noise": 0.02},
        ({"op": "add_noise_dims", "D": 50, "sigma": s},),
        methods,
        SEEDS5,
        keep=keep,
    )
    cells += _cells("topology", "sphere", {"m": 2}, (), methods, SEEDS5, keep=keep)
    for levels in (1, 2):
        cells += _cells(
            "topology",
            "branching",
            {"levels": levels, "noise": 0.03},
            (),
            methods,
            SEEDS5,
            keep=keep,
        )
    return cells


def real() -> list[Cell]:
    """PBMC 3k (one dataset, five method seeds) and MNIST / Fashion-MNIST subsamples."""
    cells = []
    methods = CORE + PHATE
    keep = lambda label, seed: seed == 0  # noqa: E731
    cells += _cells(
        "real", "pbmc3k", {}, (), methods, (0,), n=0, algo_seeds=SEEDS5, keep=keep
    )
    for name in ("mnist_784", "Fashion-MNIST"):
        cells += _cells("real", name, {}, (), methods, SEEDS3, n=5000, keep=keep)
    return cells


EXPERIMENTS = {
    "noise_dims": noise_dims,
    "noise_sigma": noise_sigma,
    "isometry": isometry,
    "intrinsic": intrinsic,
    "signal_widget": signal_widget,
    "tuning": tuning,
    "stability": stability,
    "umap_widget": umap_widget,
    "overlap": overlap,
    "density": density,
    "topology": topology,
    "real": real,
}
