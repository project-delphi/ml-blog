"""Synthetic datasets with known ground truth, plus ambient transforms.

Every base generator has the signature ``fn(n, seed, **params) -> Dataset`` and draws
from its own ``np.random.default_rng(seed)``. Nothing is standardised silently: each
docstring states the typical nearest-neighbour (NN) spacing of its output at the
default parameters, so a reader can compare a noise level or a ``k`` with the scale
of the data.

Two facts behind several of the stated spacings, for points spread uniformly with
intensity ``lam`` per unit measure on an m-dimensional set:

* the mean NN distance is ``Gamma(1 + 1/m) * (lam * V_m) ** (-1/m)``, where ``V_m`` is
  the volume of the unit m-ball; for m = 1 that is ``1 / (2 lam)``, for m = 2 it is
  ``0.5 / sqrt(lam)``;
* the empirical values quoted were measured at the stated ``n`` and seed 0, and move
  by a few per cent between seeds.

``Dataset.clean`` is the noiseless structure. A generator's own ``noise`` parameter
is applied to ``X`` only, so ``clean`` is ``X`` (the same array) exactly when that
noise is zero and no transform has been applied.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.special import expit, gammaln, logsumexp

__all__ = [
    "GENERATORS",
    "Dataset",
    "add_isotropic_noise",
    "add_noise_dims",
    "branching",
    "circle",
    "elephant_target",
    "gaussian_clusters",
    "latent_factor",
    "overlapping",
    "random_isometry",
    "sphere",
    "swiss_roll",
    "uniform_ball",
    "variable_density",
]


# --------------------------------------------------------------------------------------
# The container
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Dataset:
    """A synthetic dataset: what the method sees, and the truth it should recover."""

    name: str
    X: np.ndarray
    latent: np.ndarray
    clean: np.ndarray
    labels: np.ndarray | None
    density: np.ndarray | None
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def geodesic(self, i: np.ndarray, j: np.ndarray) -> np.ndarray | None:
        """Ground-truth geodesic distances for index pairs, or None if unknown."""
        fn = _GEODESICS.get(self.meta.get("generator", self.name))
        if fn is None:
            return None
        i = np.asarray(i, dtype=np.intp)
        j = np.asarray(j, dtype=np.intp)
        i, j = np.broadcast_arrays(i, j)
        return fn(self, i, j)


def _meta(
    generator: str,
    n: int,
    seed: int,
    params: dict[str, Any],
    intrinsic_dim: int,
    ambient_dim: int,
    betti: tuple[int, ...] | None,
) -> dict[str, Any]:
    """Build the standard meta dict; params appear both flat and under "params"."""
    return {
        "generator": generator,
        **params,
        "params": dict(params),
        "seed": seed,
        "n": n,
        "intrinsic_dim": intrinsic_dim,
        "ambient_dim": ambient_dim,
        "betti": betti,
        "transforms": [],
    }


def _pad(Y: np.ndarray, dim: int) -> np.ndarray:
    """Zero-pad the columns of Y up to dim."""
    if dim < Y.shape[1]:
        raise ValueError(f"dim={dim} is smaller than the structure's {Y.shape[1]} dims")
    out = np.zeros((Y.shape[0], dim))
    out[:, : Y.shape[1]] = Y
    return out


def _with_noise(
    clean: np.ndarray, noise: float, rng: np.random.Generator
) -> np.ndarray:
    """Return clean itself when noise is zero, else clean + noise * N(0, I)."""
    if noise == 0:
        return clean
    return clean + noise * rng.standard_normal(clean.shape)


def _relative(log_density: np.ndarray) -> np.ndarray:
    """Exponentiate a log density on a relative scale (max 1) so it never underflows."""
    return np.exp(log_density - log_density.max())


def _random_orthonormal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    """A (rows x cols) matrix with orthonormal columns, Haar-distributed (QR, sign-fixed)."""
    q, r = np.linalg.qr(rng.standard_normal((rows, cols)))
    return q * np.sign(np.diag(r))


# --------------------------------------------------------------------------------------
# Swiss-roll geometry (shared by swiss_roll and variable_density)
# --------------------------------------------------------------------------------------

_T0 = 1.5 * np.pi  # sklearn's starting angle


def _arc(t: np.ndarray | float) -> np.ndarray:
    """Arc length of the spiral r = t from angle 0: integral of sqrt(1 + t^2)."""
    t = np.asarray(t, dtype=float)
    return 0.5 * (t * np.sqrt(1.0 + t * t) + np.arcsinh(t))


def _roll_length(turns: float) -> float:
    """Arc length of a roll of `turns` turns starting at angle 1.5 pi."""
    return float(_arc(_T0 + 2.0 * np.pi * turns) - _arc(_T0))


def _arc_inverse(s: np.ndarray) -> np.ndarray:
    """Angle t at which the arc length measured from 1.5 pi equals s (Newton)."""
    target = np.asarray(s, dtype=float) + _arc(_T0)
    # _arc(t) >= t^2 / 2, so sqrt(2 * target) lies right of the root; Newton on a convex
    # increasing function started from the right converges monotonically.
    t = np.sqrt(2.0 * target)
    for _ in range(100):
        step = (_arc(t) - target) / np.sqrt(1.0 + t * t)
        t = t - step
        if np.max(np.abs(step), initial=0.0) < 1e-13:
            break
    return t


def _roll_embed(s: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map arc length s and height h onto the roll (t cos t, h, t sin t)."""
    t = _arc_inverse(s)
    return np.column_stack([t * np.cos(t), h, t * np.sin(t)]), t


# --------------------------------------------------------------------------------------
# Base generators
# --------------------------------------------------------------------------------------


def gaussian_clusters(
    n: int,
    seed: int,
    n_clusters: int = 10,
    dim: int = 10,
    separation: float = 6.0,
    imbalance: float = 1.0,
    anisotropy: float = 1.0,
    within_sd: float = 1.0,
) -> Dataset:
    """A Gaussian mixture whose closest two centres sit `separation` sds apart.

    Centres are a seeded N(0, I) draw rescaled so the minimum pairwise centre distance
    is exactly ``separation * within_sd``. Cluster k has size proportional to
    ``imbalance ** (-k / (n_clusters - 1))`` (cluster 0 the largest), rounded by
    largest remainder so the sizes sum to ``n``. Each cluster has its own random
    rotation and per-axis sds log-spaced from ``within_sd / sqrt(anisotropy)`` to
    ``within_sd * sqrt(anisotropy)``.

    ``density`` is the mixture pdf (weights = sizes / n) on a relative scale;
    ``extra["log_density"]`` is the absolute log pdf.

    NN spacing: within a cluster, about ``0.7 * within_sd * sqrt(dim)`` at n = 2000
    with the defaults (measured 2.16 for dim 10, 200 points per cluster); it shrinks
    only slowly with n, as ``n ** (-1 / dim)``.
    """
    if n < n_clusters:
        raise ValueError("need at least one point per cluster")
    if imbalance < 1 or anisotropy < 1:
        raise ValueError("imbalance and anisotropy are ratios >= 1")
    rng = np.random.default_rng(seed)
    params = dict(
        n_clusters=n_clusters,
        dim=dim,
        separation=separation,
        imbalance=imbalance,
        anisotropy=anisotropy,
        within_sd=within_sd,
    )

    centres = rng.standard_normal((n_clusters, dim))
    if n_clusters > 1:
        diff = centres[:, None, :] - centres[None, :, :]
        dist = np.sqrt((diff**2).sum(-1))
        dmin = dist[np.triu_indices(n_clusters, 1)].min()
        centres *= separation * within_sd / dmin
    else:
        centres[:] = 0.0

    sizes = _geometric_sizes(n, n_clusters, imbalance)
    labels = np.repeat(np.arange(n_clusters), sizes)

    sds = within_sd * anisotropy ** np.linspace(-0.5, 0.5, dim)
    rotations = np.stack(
        [_random_orthonormal(rng, dim, dim) for _ in range(n_clusters)]
    )
    z = rng.standard_normal((n, dim)) * sds
    X = centres[labels] + np.einsum("nd,ned->ne", z, rotations[labels])

    # mixture log pdf: for cluster k, x = c_k + R_k (sds * z), so z = (R_k^T (x-c_k)) / sds
    log_w = np.log(sizes / n)
    log_norm = -np.log(sds).sum() - 0.5 * dim * np.log(2 * np.pi)
    comp = np.empty((n, n_clusters))
    for k in range(n_clusters):
        u = ((X - centres[k]) @ rotations[k]) / sds
        comp[:, k] = log_w[k] + log_norm - 0.5 * (u**2).sum(1)
    log_density = logsumexp(comp, axis=1)

    return Dataset(
        name="gaussian_clusters",
        X=X,
        latent=X,
        clean=X,
        labels=labels,
        density=_relative(log_density),
        extra={
            "centres": centres,
            "sizes": sizes,
            "axis_sds": sds,
            "rotations": rotations,
            "log_density": log_density,
        },
        meta=_meta("gaussian_clusters", n, seed, params, dim, dim, (n_clusters,)),
    )


def _geometric_sizes(n: int, k: int, ratio: float) -> np.ndarray:
    """Integer sizes summing to n, geometrically spaced from largest to largest/ratio."""
    if k == 1:
        return np.array([n])
    w = ratio ** (-np.arange(k) / (k - 1))
    raw = n * w / w.sum()
    sizes = np.floor(raw).astype(int)
    short = n - sizes.sum()
    sizes[np.argsort(-(raw - sizes), kind="stable")[:short]] += 1
    sizes = np.maximum(sizes, 1)
    sizes[0] -= sizes.sum() - n
    return sizes


def swiss_roll(
    n: int, seed: int, turns: float = 1.5, height: float = 21.0, noise: float = 0.0
) -> Dataset:
    """A swiss roll sampled uniformly by surface area, in arc-length coordinates.

    The roll is ``(t cos t, h, t sin t)`` with ``t`` from 1.5 pi to
    ``1.5 pi + 2 pi turns`` (sklearn's range at turns = 1.5) and ``h`` uniform on
    ``[0, height]``. Arc length ``s(t) = 1/2 [t sqrt(1+t^2) + asinh t]`` (from 1.5 pi)
    is sampled uniformly and inverted numerically, so the density on the surface is
    uniform and ``latent = (s, h)`` is an isometric flattening: latent Euclidean
    distance is the true geodesic. ``noise`` adds ``noise * N(0, I_3)`` to X.

    NN spacing: the unrolled sheet is ``88.8 x 21`` at the defaults (area 1865), so
    the mean NN distance is ``0.5 * sqrt(1865 / n)``: 0.48 at n = 2000 (measured 0.49),
    0.68 at n = 1000 (measured 0.70).
    """
    rng = np.random.default_rng(seed)
    length = _roll_length(turns)
    s = rng.uniform(0.0, length, n)
    h = rng.uniform(0.0, height, n)
    clean, t = _roll_embed(s, h)
    X = _with_noise(clean, noise, rng)
    return Dataset(
        name="swiss_roll",
        X=X,
        latent=np.column_stack([s, h]),
        clean=clean,
        labels=None,
        density=np.full(n, 1.0 / (length * height)),
        extra={"s": s, "t": t, "h": h},
        meta=_meta(
            "swiss_roll",
            n,
            seed,
            dict(turns=turns, height=height, noise=noise),
            2,
            3,
            (1, 0),
        ),
    )


def circle(
    n: int, seed: int, radius: float = 1.0, noise: float = 0.0, dim: int = 2
) -> Dataset:
    """A circle of `radius` in the first two coordinates of R^dim, uniform in angle.

    ``latent`` is the noiseless circle itself, ``radius * (cos t, sin t)``, so latent
    kNN has no seam at t = 0; the angle is ``extra["theta"]``. ``noise`` adds ``noise * N(0, I_dim)``.

    NN spacing: ``pi * radius / n`` on average (half the arc gap): 0.0031 at n = 1000,
    radius 1.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    clean = _pad(radius * np.column_stack([np.cos(theta), np.sin(theta)]), dim)
    X = _with_noise(clean, noise, rng)
    return Dataset(
        name="circle",
        X=X,
        latent=radius * np.column_stack([np.cos(theta), np.sin(theta)]),
        clean=clean,
        labels=None,
        density=np.full(n, 1.0 / (2.0 * np.pi * radius)),
        extra={"theta": theta},
        meta=_meta(
            "circle", n, seed, dict(radius=radius, noise=noise, dim=dim), 1, dim, (1, 1)
        ),
    )


def sphere(
    n: int, seed: int, m: int = 2, radius: float = 1.0, noise: float = 0.0
) -> Dataset:
    """Points uniform on the m-sphere of `radius` in R^(m+1) (normalised Gaussians).

    ``latent`` holds the unit vectors; the geodesic is the great-circle distance
    ``radius * arccos(u_i . u_j)``. ``noise`` adds ``noise * N(0, I_(m+1))``.

    NN spacing: for S^2, area ``4 pi radius^2``, so ``0.5 * sqrt(4 pi / n) * radius``:
    0.056 at n = 1000 (measured 0.055); for general m use the m-dimensional formula
    in the module docstring with the sphere's surface area.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, m + 1))
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    clean = radius * u
    X = _with_noise(clean, noise, rng)
    log_area = (
        np.log(2.0) + 0.5 * (m + 1) * np.log(np.pi) - gammaln(0.5 * (m + 1))
    ) + m * np.log(radius)
    return Dataset(
        name="sphere",
        X=X,
        latent=u,
        clean=clean,
        labels=None,
        density=np.full(n, np.exp(-log_area)),
        extra={},
        meta=_meta(
            "sphere",
            n,
            seed,
            dict(m=m, radius=radius, noise=noise),
            m,
            m + 1,
            tuple([1] + [0] * (m - 1) + [1]),
        ),
    )


def branching(
    n: int,
    seed: int,
    n_branches: int = 3,
    dim: int = 10,
    branch_length: float = 1.0,
    noise: float = 0.03,
    levels: int = 1,
) -> Dataset:
    """A tree of straight segments in R^dim: a trunk, then branches from its tip.

    Level 1 is ``n_branches`` children of the trunk's tip; each further level splits
    every segment of the level above in two. Directions come from a random
    orthonormal set (or random unit vectors when there are more segments than
    dimensions), each child tilted half-way back toward its parent so the tree reads
    as a forward trajectory. Points are uniform by length over the tree, then get
    ``noise * N(0, I_dim)``.

    ``labels`` is the segment (branch) id, 0 the trunk; ``extra["pseudotime"]`` is the
    path length from the root; ``latent`` is the noiseless tree position (= ``clean``),
    so latent kNN are along-tree neighbours and never jump between branches that share
    a pseudotime. The
    geodesic between samples is the tree path length between their generating
    (noiseless) positions on the tree.

    NN spacing: along the tree it is ``total_length / (2 n)`` (0.002 at n = 1000, four
    unit segments), far below the noise; with noise 0.03 in 10 dims the measured mean
    NN distance at n = 1000 is 0.079, i.e. set by the noise (about 2.6 * noise).
    """
    if levels < 1 or n_branches < 1:
        raise ValueError("levels and n_branches must be >= 1")
    rng = np.random.default_rng(seed)
    params = dict(
        n_branches=n_branches,
        dim=dim,
        branch_length=branch_length,
        noise=noise,
        levels=levels,
    )

    # topology: segments as (start node, end node, parent segment)
    seg_nodes: list[tuple[int, int]] = [(0, 1)]
    parent: list[int] = [-1]
    frontier = [0]
    n_nodes = 2
    for level in range(1, levels + 1):
        fan = n_branches if level == 1 else 2
        nxt = []
        for p in frontier:
            for _ in range(fan):
                seg_nodes.append((seg_nodes[p][1], n_nodes))
                parent.append(p)
                nxt.append(len(seg_nodes) - 1)
                n_nodes += 1
        frontier = nxt
    n_seg = len(seg_nodes)

    if n_seg <= dim:
        basis = _random_orthonormal(rng, dim, n_seg).T
    else:
        g = rng.standard_normal((n_seg, dim))
        basis = g / np.linalg.norm(g, axis=1, keepdims=True)
    directions = np.empty((n_seg, dim))
    nodes = np.zeros((n_nodes, dim))
    for k, (a, b) in enumerate(seg_nodes):
        d = basis[k] if parent[k] < 0 else basis[k] + 0.5 * directions[parent[k]]
        directions[k] = d / np.linalg.norm(d)
        nodes[b] = nodes[a] + branch_length * directions[k]

    seg_nodes_arr = np.array(seg_nodes)
    seg_length = np.full(n_seg, float(branch_length))
    node_depth = np.zeros(n_nodes)
    for k, (a, b) in enumerate(seg_nodes):
        node_depth[b] = node_depth[a] + seg_length[k]
    node_dist = _tree_node_distances(n_nodes, seg_nodes_arr, seg_length)

    seg = rng.choice(n_seg, size=n, p=seg_length / seg_length.sum())
    offset = rng.uniform(0.0, 1.0, n) * seg_length[seg]
    clean = nodes[seg_nodes_arr[seg, 0]] + offset[:, None] * directions[seg]
    X = _with_noise(clean, noise, rng)
    pseudotime = node_depth[seg_nodes_arr[seg, 0]] + offset

    return Dataset(
        name="branching",
        X=X,
        latent=clean,
        clean=clean,
        labels=seg,
        density=np.full(n, 1.0 / seg_length.sum()),
        extra={
            "pseudotime": pseudotime,
            "branch": seg,
            "offset": offset,
            "nodes": nodes,
            "seg_nodes": seg_nodes_arr,
            "seg_length": seg_length,
            "seg_parent": np.array(parent),
            "node_dist": node_dist,
        },
        meta=_meta("branching", n, seed, params, 1, dim, (1, 0)),
    )


def _tree_node_distances(
    n_nodes: int, seg_nodes: np.ndarray, seg_length: np.ndarray
) -> np.ndarray:
    """All node-to-node path lengths on a tree, by breadth-first search from each node."""
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for (a, b), length in zip(seg_nodes, seg_length, strict=True):
        adj[a].append((b, length))
        adj[b].append((a, length))
    dist = np.full((n_nodes, n_nodes), np.inf)
    for src in range(n_nodes):
        dist[src, src] = 0.0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v, length in adj[u]:
                if np.isinf(dist[src, v]):
                    dist[src, v] = dist[src, u] + length
                    queue.append(v)
    return dist


def variable_density(
    n: int,
    seed: int,
    dim: int = 3,
    density_ratio: float = 10.0,
    shape: str = "strip",
) -> Dataset:
    """A sheet sampled with density rising smoothly by `density_ratio` along s.

    The sheet is the default swiss roll's unrolled rectangle, ``s`` in ``[0, 88.8]``
    and ``h`` in ``[0, 21]``. ``s`` has density proportional to
    ``exp(log(density_ratio) * s / L)`` (inverse-CDF sampled), ``h`` is uniform.
    ``shape="strip"`` lays the rectangle flat in the first two coordinates of R^dim;
    ``shape="roll"`` rolls it up exactly as :func:`swiss_roll` does (needs dim >= 3).
    Either way ``latent = (s, h)`` and latent Euclidean distance is the geodesic.
    ``density`` is the true surface density ``p(s) / height`` at each point.

    NN spacing: varies by ``sqrt(density_ratio)`` along s. At n = 2000 and ratio 10
    the local mean NN distance is ``0.5 / sqrt(n * density)``: about 1.0 at s = 0 and
    0.31 at s = L (measured mean over the sheet 0.47).
    """
    if shape not in ("strip", "roll"):
        raise ValueError('shape must be "strip" or "roll"')
    if density_ratio <= 0:
        raise ValueError("density_ratio must be positive")
    rng = np.random.default_rng(seed)
    length = _roll_length(1.5)
    height = 21.0
    c = np.log(density_ratio)
    u = rng.uniform(0.0, 1.0, n)
    if c == 0:
        s = length * u
        pdf_s = np.full(n, 1.0 / length)
    else:
        s = length * np.log1p(u * (density_ratio - 1.0)) / c
        pdf_s = c * np.exp(c * s / length) / (length * (density_ratio - 1.0))
    h = rng.uniform(0.0, height, n)
    if shape == "strip":
        clean = _pad(np.column_stack([s, h]), dim)
    else:
        roll, _ = _roll_embed(s, h)
        clean = _pad(roll, dim)
    return Dataset(
        name="variable_density",
        X=clean,
        latent=np.column_stack([s, h]),
        clean=clean,
        labels=None,
        density=pdf_s / height,
        extra={"s": s, "h": h, "pdf_s": pdf_s},
        meta=_meta(
            "variable_density",
            n,
            seed,
            dict(dim=dim, density_ratio=density_ratio, shape=shape),
            2,
            dim,
            (1, 0),
        ),
    )


def overlapping(
    n: int,
    seed: int,
    dim: int = 10,
    delta: float = 1.0,
    weights: tuple[float, float] = (0.5, 0.5),
) -> Dataset:
    """Two unit Gaussians N(-delta/2 e1, I) (class 0) and N(+delta/2 e1, I) (class 1).

    Labels are drawn with probabilities ``weights``. ``extra["posterior"]`` is the
    exact ``P(class 1 | x) = sigmoid(delta * x_1 + log(w1 / w0))``; ``density`` is the
    mixture pdf on a relative scale, ``extra["log_density"]`` the absolute log pdf.

    NN spacing: measured 1.70 at n = 2000 in 10 dims (1.85 at n = 1000); like any
    full-dimensional Gaussian it falls only as ``n ** (-1 / dim)``.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != (2,) or np.any(w <= 0):
        raise ValueError("weights must be two positive numbers")
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    labels = (rng.uniform(size=n) < w[1]).astype(int)
    mu = np.zeros((2, dim))
    mu[0, 0], mu[1, 0] = -delta / 2.0, delta / 2.0
    X = mu[labels] + rng.standard_normal((n, dim))
    posterior = expit(delta * X[:, 0] + np.log(w[1] / w[0]))
    sq = ((X[:, None, :] - mu[None]) ** 2).sum(-1)
    log_density = logsumexp(np.log(w) - 0.5 * sq, axis=1) - 0.5 * dim * np.log(
        2 * np.pi
    )
    return Dataset(
        name="overlapping",
        X=X,
        latent=X,
        clean=X,
        labels=labels,
        density=_relative(log_density),
        extra={"posterior": posterior, "means": mu, "log_density": log_density},
        meta=_meta(
            "overlapping",
            n,
            seed,
            dict(dim=dim, delta=delta, weights=tuple(float(x) for x in weights)),
            dim,
            dim,
            (1,),
        ),
    )


def latent_factor(
    n: int, seed: int, d: int = 5, D: int | None = None, noise: float = 0.0
) -> Dataset:
    """Z ~ N(0, I_d) mapped into R^D by a random orthonormal (D x d) matrix.

    ``latent`` is Z; since the map is an isometry the geodesic is latent Euclidean
    distance. ``noise`` adds ``noise * N(0, I_D)``. ``density`` is the N(0, I_d) pdf of
    Z (relative scale), the density on the d-dimensional subspace.

    NN spacing: measured 0.70 at n = 1000 for d = 5 (0.61 at n = 2000).
    """
    D = d if D is None else D
    if D < d:
        raise ValueError("D must be >= d")
    rng = np.random.default_rng(seed)
    A = _random_orthonormal(rng, D, d)
    Z = rng.standard_normal((n, d))
    clean = Z @ A.T
    X = _with_noise(clean, noise, rng)
    log_density = -0.5 * (Z**2).sum(1) - 0.5 * d * np.log(2 * np.pi)
    return Dataset(
        name="latent_factor",
        X=X,
        latent=Z,
        clean=clean,
        labels=None,
        density=_relative(log_density),
        extra={"loadings": A, "log_density": log_density},
        meta=_meta("latent_factor", n, seed, dict(d=d, D=D, noise=noise), d, D, (1,)),
    )


def uniform_ball(n: int, seed: int, d: int = 5) -> Dataset:
    """Points uniform in the unit d-ball (Gaussian direction, radius U^(1/d)).

    NN spacing: ``Gamma(1 + 1/d) * n ** (-1/d)`` before boundary effects, which
    lengthen it: measured 0.24 at n = 1000 for d = 5.
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    X = g * rng.uniform(size=n)[:, None] ** (1.0 / d)
    log_volume = 0.5 * d * np.log(np.pi) - gammaln(0.5 * d + 1.0)
    return Dataset(
        name="uniform_ball",
        X=X,
        latent=X,
        clean=X,
        labels=None,
        density=np.full(n, np.exp(-log_volume)),
        extra={},
        meta=_meta("uniform_ball", n, seed, dict(d=d), d, d, (1,)),
    )


# Side view, facing left, traced clockwise from the top of the rump. Units are
# arbitrary; the figure spans about 9.3 x 7.
_ELEPHANT = np.array(
    [
        (9.00, 5.60),  # rump
        (9.35, 5.00),  # tail
        (9.75, 3.60),
        (9.90, 3.30),
        (9.62, 3.35),
        (9.18, 4.55),
        (9.10, 3.70),  # back of hind leg
        (9.00, 2.60),
        (8.95, 0.00),  # hind foot (near)
        (7.95, 0.00),
        (7.95, 1.90),
        (7.75, 1.90),
        (7.65, 0.00),  # hind foot (far)
        (6.85, 0.00),
        (6.85, 2.10),
        (6.00, 2.05),  # belly
        (5.00, 2.10),
        (4.85, 1.95),
        (4.85, 0.00),  # front foot (far)
        (4.05, 0.00),
        (4.10, 1.85),
        (3.85, 1.85),
        (3.80, 0.00),  # front foot (near)
        (2.85, 0.00),
        (2.95, 2.60),
        (2.70, 3.20),  # chest
        (2.15, 3.55),  # jaw
        (1.75, 3.10),  # trunk, back edge
        (1.55, 2.20),
        (1.45, 1.30),
        (1.35, 0.75),
        (1.05, 0.45),  # trunk tip curls forward
        (0.70, 0.55),
        (0.70, 0.80),
        (0.95, 0.85),  # trunk, front edge
        (1.05, 1.40),
        (1.10, 2.30),
        (1.15, 3.30),
        (1.30, 4.40),  # forehead
        (1.60, 5.30),
        (2.05, 5.95),
        (2.55, 6.25),  # crown
        (2.95, 6.85),  # ear rises behind the head
        (3.60, 7.00),
        (4.15, 6.70),
        (4.35, 6.20),
        (5.30, 6.40),  # back
        (6.50, 6.60),
        (7.70, 6.50),
        (8.60, 6.20),
    ]
)


def _polygon_area(vertices: np.ndarray) -> float:
    """Shoelace area of a simple polygon."""
    x, y = vertices[:, 0], vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def elephant_target(n: int, seed: int) -> Dataset:
    """Points uniform inside a hand-drawn elephant silhouette (rejection sampling).

    ``n`` is the number of points (the spec's ``n_points``). The 50-vertex polygon is
    in ``extra["polygon"]``; geodesic is unknown (the shape is not convex).

    NN spacing: the silhouette's area is 40.75 square units, so the mean NN
    distance is ``0.5 * sqrt(40.75 / n)``: 0.10 at n = 1000 (measured 0.104).
    """
    rng = np.random.default_rng(seed)
    path = MplPath(_ELEPHANT)
    lo, hi = _ELEPHANT.min(0), _ELEPHANT.max(0)
    area = _polygon_area(_ELEPHANT)
    accept = area / float(np.prod(hi - lo))
    chunks: list[np.ndarray] = []
    have = 0
    while have < n:
        m = int(1.2 * (n - have) / accept) + 16
        cand = rng.uniform(lo, hi, size=(m, 2))
        keep = cand[path.contains_points(cand)]
        chunks.append(keep)
        have += len(keep)
    X = np.concatenate(chunks)[:n]
    return Dataset(
        name="elephant_target",
        X=X,
        latent=X,
        clean=X,
        labels=None,
        density=np.full(n, 1.0 / area),
        extra={"polygon": _ELEPHANT.copy()},
        meta=_meta("elephant_target", n, seed, {}, 2, 2, None),
    )


GENERATORS: dict[str, Callable[..., Dataset]] = {
    "gaussian_clusters": gaussian_clusters,
    "swiss_roll": swiss_roll,
    "circle": circle,
    "sphere": sphere,
    "branching": branching,
    "variable_density": variable_density,
    "overlapping": overlapping,
    "latent_factor": latent_factor,
    "uniform_ball": uniform_ball,
    "elephant_target": elephant_target,
}


# --------------------------------------------------------------------------------------
# Geodesics
# --------------------------------------------------------------------------------------


def _geo_latent_euclidean(ds: Dataset, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Euclidean distance in the latent coordinates (flat or isometrically flattened)."""
    return np.linalg.norm(ds.latent[i] - ds.latent[j], axis=-1)


def _geo_circle(ds: Dataset, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Radius times the smaller angular gap."""
    theta = ds.extra["theta"]
    gap = np.abs(theta[i] - theta[j]) % (2.0 * np.pi)
    return ds.meta["radius"] * np.minimum(gap, 2.0 * np.pi - gap)


def _geo_sphere(ds: Dataset, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Great-circle distance, radius * arccos(u . v).

    Computed as ``2 atan2(|u - v|, |u + v|)``, the same angle without arccos's loss of
    precision near 0 and pi.
    """
    u, v = ds.latent[i], ds.latent[j]
    angle = 2.0 * np.arctan2(
        np.linalg.norm(u - v, axis=-1), np.linalg.norm(u + v, axis=-1)
    )
    return ds.meta["radius"] * angle


def _geo_tree(ds: Dataset, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Path length along the tree between the samples' generating positions."""
    seg, off = ds.extra["branch"], ds.extra["offset"]
    ends, length, nd = (
        ds.extra["seg_nodes"],
        ds.extra["seg_length"],
        ds.extra["node_dist"],
    )
    sa, sb = seg[i], seg[j]
    ta, tb = off[i], off[j]
    ua, va = ends[sa, 0], ends[sa, 1]
    ub, vb = ends[sb, 0], ends[sb, 1]
    ra, rb = length[sa] - ta, length[sb] - tb
    via = np.minimum.reduce(
        [
            ta + nd[ua, ub] + tb,
            ta + nd[ua, vb] + rb,
            ra + nd[va, ub] + tb,
            ra + nd[va, vb] + rb,
        ]
    )
    return np.where(sa == sb, np.abs(ta - tb), via)


_GEODESICS: dict[str, Callable[[Dataset, np.ndarray, np.ndarray], np.ndarray]] = {
    "swiss_roll": _geo_latent_euclidean,
    "variable_density": _geo_latent_euclidean,
    "latent_factor": _geo_latent_euclidean,
    "uniform_ball": _geo_latent_euclidean,
    "circle": _geo_circle,
    "sphere": _geo_sphere,
    "branching": _geo_tree,
}


# --------------------------------------------------------------------------------------
# Ambient transforms
# --------------------------------------------------------------------------------------

# Each transform mixes a fixed tag into its seed, so a transform seeded like the
# generator does not replay the generator's random stream.
_OP_TAG = {"add_noise_dims": 1, "add_isotropic_noise": 2, "random_isometry": 3}


def _op_rng(op: str, seed: int) -> np.random.Generator:
    """The transform's own generator, derived from (seed, op tag)."""
    return np.random.default_rng(np.random.SeedSequence([seed, _OP_TAG[op]]))


def _transformed(ds: Dataset, X: np.ndarray, record: dict[str, Any]) -> Dataset:
    """A copy of ds with new X, updated ambient_dim and one more transform record."""
    meta = dict(ds.meta)
    meta["ambient_dim"] = X.shape[1]
    meta["transforms"] = [*ds.meta.get("transforms", []), record]
    return replace(ds, X=X, meta=meta)


def add_noise_dims(ds: Dataset, D: int, sigma: float, seed: int) -> Dataset:
    """X <- [X, sigma * N(0,1)^(n x (D - p))]. D is the TOTAL ambient dim; raise if D < p."""
    n, p = ds.X.shape
    if D < p:
        raise ValueError(f"D={D} is smaller than the current ambient dim {p}")
    noise = sigma * _op_rng("add_noise_dims", seed).standard_normal((n, D - p))
    X = np.hstack([ds.X, noise])
    return _transformed(
        ds, X, {"op": "add_noise_dims", "D": D, "sigma": sigma, "seed": seed}
    )


def add_isotropic_noise(ds: Dataset, sigma: float, seed: int) -> Dataset:
    """X <- X + sigma * N(0, I). Used for noise on the structured dims themselves."""
    rng = _op_rng("add_isotropic_noise", seed)
    X = ds.X + sigma * rng.standard_normal(ds.X.shape)
    return _transformed(
        ds, X, {"op": "add_isotropic_noise", "sigma": sigma, "seed": seed}
    )


def random_isometry(ds: Dataset, D: int, seed: int) -> Dataset:
    """X <- X Q^T with Q a (D x p) matrix with orthonormal columns (QR of a Gaussian).

    Pairwise distances are preserved exactly. D total; raise if D < p.
    """
    p = ds.X.shape[1]
    if D < p:
        raise ValueError(f"D={D} is smaller than the current ambient dim {p}")
    Q = _random_orthonormal(_op_rng("random_isometry", seed), D, p)
    return _transformed(ds, ds.X @ Q.T, {"op": "random_isometry", "D": D, "seed": seed})
