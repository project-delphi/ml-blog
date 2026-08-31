"""Masked CP-ALS and a flattened SVD fill for a rating tensor."""

from __future__ import annotations

import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

tl.set_backend("numpy")
SEED = 7


def holdout(mask: np.ndarray, frac: float = 0.2, seed: int = SEED) -> np.ndarray:
    """Hide a random subset of the observed cells. Returns the training mask."""
    rng = np.random.default_rng(seed)
    observed = np.argwhere(mask)
    n_hide = int(frac * len(observed))
    pick = rng.choice(len(observed), size=n_hide, replace=False)
    train = mask.copy()
    for i, j, k in observed[pick]:
        train[i, j, k] = False
    return train


def cp_complete(
    X: np.ndarray,
    mask: np.ndarray,
    rank: int,
    n_outer: int = 12,
    n_inner: int = 15,
    seed: int = SEED,
) -> np.ndarray:
    """Impute-and-project CP-ALS. Observed cells stay put; the holes are updated."""
    fill = float(np.nanmean(np.where(mask, X, np.nan)))
    Y = np.where(mask, X, fill)
    rng = np.random.default_rng(seed)
    recon = Y
    for _ in range(n_outer):
        cp = parafac(
            Y,
            rank=rank,
            n_iter_max=n_inner,
            init="random",
            random_state=int(rng.integers(1_000_000_000)),
            tol=1e-6,
        )
        recon = cp_to_tensor(cp)
        Y = np.where(mask, X, recon)
    return recon


def flatten_svd_complete(
    X: np.ndarray,
    mask: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Average over the time mode, then a rank-k SVD of the user x movie matrix.

    Missing user-movie means (no rating in any month) are filled with the
    global mean before the SVD. The reconstruction is repeated across months.
    """
    n_u, n_m, n_t = X.shape
    month_mask = mask.any(axis=2)
    num = np.where(mask, X, 0.0).sum(axis=2)
    den = mask.sum(axis=2)
    mean = np.divide(num, den, out=np.full((n_u, n_m), np.nanmean(X[mask])), where=den > 0)
    fill = float(np.nanmean(mean[month_mask]))
    M = np.where(month_mask, mean, fill)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(rank, s.size)
    recon2 = (U[:, :k] * s[:k]) @ Vt[:k]
    return np.repeat(recon2[:, :, None], n_t, axis=2)


def rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    err = pred[mask] - truth[mask]
    return float(np.sqrt(np.mean(err**2)))
