"""Persistent homology of point clouds via ripser, on a reproducible subsample.

Normalisation: ``persistence`` takes a farthest-point (maxmin) subsample of m points,
divides it by its own diameter (the largest pairwise distance within the subsample),
and runs ripser on the result. Every finite birth and death is therefore a fraction of
the cloud's diameter, in [0, 1], whatever the units of the input, so one threshold
serves an input in R^1000 and its 2-D embedding alike. The maxmin subsample contains
its seeded start point and the point farthest from it, so its diameter is at least
half the full cloud's.

Lifetimes (death - birth) and thresholds are in those diameter units throughout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import pdist, squareform

if TYPE_CHECKING:
    from .generators import Dataset

DEFAULT_THRESHOLD = 0.1  # a bar counts toward a Betti number above 10% of the diameter

__all__ = [
    "DEFAULT_THRESHOLD",
    "subsample",
    "persistence",
    "betti_estimate",
    "top_persistence",
    "topology_rows",
]


def _cloud(Z: np.ndarray) -> np.ndarray:
    """Return Z as a finite (n, d) float64 array; a 1-D input becomes (n, 1)."""
    A = np.asarray(Z, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2 or A.shape[0] < 1:
        raise ValueError(f"Z must be an (n, d) array, got shape {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError("Z contains NaN or inf")
    return A


def subsample(Z: np.ndarray, m: int, seed: int, method: str = "maxmin") -> np.ndarray:
    """Indices of a reproducible m-point subsample of Z (all n indices when m >= n).

    "maxmin" starts at a seeded random point and repeatedly adds the point farthest
    from those chosen (ties to the lowest index), which spreads the subsample evenly
    over the cloud and keeps its extremes. "random" draws m points without replacement.
    """
    Z = _cloud(Z)
    n = Z.shape[0]
    if int(m) != m or m < 1:
        raise ValueError(f"m must be a positive integer, got {m!r}")
    if method not in ("maxmin", "random"):
        raise ValueError(
            f"unknown subsample method {method!r}; use 'maxmin' or 'random'"
        )
    m = int(m)
    if m >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if method == "random":
        return np.sort(rng.choice(n, size=m, replace=False))
    chosen = np.empty(m, dtype=np.intp)
    chosen[0] = rng.integers(n)
    diff = Z - Z[chosen[0]]
    gap = np.einsum("ij,ij->i", diff, diff)  # squared distance to the chosen set
    for t in range(1, m):
        chosen[t] = int(np.argmax(gap))
        diff = Z - Z[chosen[t]]
        np.minimum(gap, np.einsum("ij,ij->i", diff, diff), out=gap)
    return chosen


def persistence(
    Z: np.ndarray, maxdim: int = 1, m: int = 400, seed: int = 0
) -> list[np.ndarray]:
    """Ripser diagrams (H0..H_maxdim) on a maxmin subsample of Z rescaled to unit diameter.

    See the module docstring for the normalisation. A cloud whose subsample is a
    single repeated point is left unscaled (its diameter is 0).
    """
    from ripser import ripser

    if int(maxdim) != maxdim or maxdim < 0:
        raise ValueError(f"maxdim must be a non-negative integer, got {maxdim!r}")
    Z = _cloud(Z)
    P = Z[subsample(Z, m, seed, method="maxmin")]
    if P.shape[0] < 2:
        raise ValueError("persistence needs at least 2 points")
    D = squareform(pdist(P))
    diam = float(D.max())
    if diam > 0:
        D /= diam
    return list(ripser(D, maxdim=int(maxdim), distance_matrix=True)["dgms"])


def _lifetimes(dgm: np.ndarray) -> np.ndarray:
    """Death minus birth of every bar in one diagram (inf for an infinite bar)."""
    dgm = np.asarray(dgm, dtype=np.float64).reshape(-1, 2)
    return dgm[:, 1] - dgm[:, 0]


def betti_estimate(dgms: Sequence[np.ndarray], threshold: float) -> tuple[int, ...]:
    """Number of bars with persistence (death-birth) > threshold × (diameter scale)
    per dimension; infinite H0 bar counts as 1.

    The diameter scale is 1 for diagrams from ``persistence``, which are already in
    diameter units, so ``threshold`` is a fraction of the diameter. Infinite bars
    always count.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be finite and >= 0, got {threshold!r}")
    return tuple(int((_lifetimes(d) > threshold).sum()) for d in dgms)


def top_persistence(dgms: Sequence[np.ndarray], dim: int, r: int = 3) -> list[float]:
    """Largest r finite lifetimes in dimension dim (diameter units), padded with 0.0.

    The infinite H0 bar is excluded; padding means a missing bar reads as lifetime 0.
    """
    if not 0 <= dim < len(dgms):
        raise ValueError(f"dim {dim} not in diagrams of dimensions 0..{len(dgms) - 1}")
    if int(r) != r or r < 1:
        raise ValueError(f"r must be a positive integer, got {r!r}")
    life = _lifetimes(dgms[dim])
    life = np.sort(life[np.isfinite(life)])[::-1][: int(r)]
    return [float(v) for v in life] + [0.0] * (int(r) - len(life))


def topology_rows(
    ds: Dataset,
    emb: np.ndarray,
    seed: int,
    *,
    m: int = 400,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """Rows {metric: 'h1_top1'|'h1_top2'|'h1_ratio'|'betti0'|'betti1', reference: 'input'|'emb', value}.

    reference="input" is ds.X, reference="emb" the embedding; k is NaN. h1_top1/2 are
    the two longest H1 lifetimes in diameter units; h1_ratio = top1 / top2 (inf when
    only one H1 bar exists, NaN when there is none); betti0/1 use ``threshold``.
    """
    rows = []
    for reference, Z in (("input", ds.X), ("emb", emb)):
        dgms = persistence(Z, maxdim=1, m=m, seed=seed)
        top1, top2 = top_persistence(dgms, 1, r=2)
        ratio = (
            top1 / top2 if top2 > 0 else (float("inf") if top1 > 0 else float("nan"))
        )
        b0, b1 = betti_estimate(dgms, threshold)[:2]
        for metric, value in (
            ("h1_top1", top1),
            ("h1_top2", top2),
            ("h1_ratio", ratio),
            ("betti0", b0),
            ("betti1", b1),
        ):
            rows.append(
                {
                    "metric": metric,
                    "k": float("nan"),
                    "reference": reference,
                    "value": float(value),
                }
            )
    return rows
