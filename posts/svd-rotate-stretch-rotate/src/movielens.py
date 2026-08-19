"""Truncated SVD on real MovieLens ratings, scored on held-out ratings.

The point of this module in the post is one curve: held-out RMSE against
truncation rank. It is the same storage-versus-quality dial as the photograph,
except here the wrong rank does not blur an image -- it recommends the wrong
film, and the U-shape shows exactly where extra ranks stop describing taste and
start memorising noise.

Usage:
    .venv-svd/bin/python posts/svd-rotate-stretch-rotate/src/movielens.py
"""

from __future__ import annotations

from typing import Final

import numpy as np

import data as dataset

SEED: Final[int] = 20260819
TEST_FRACTION: Final[float] = 0.10

# Ranks scored. Dense enough near the minimum to place it, sparse out in the
# tail where the curve is flat and rising.
RANKS: Final[tuple[int, ...]] = (
    1, 2, 3, 5, 7, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200,
)


def _baseline(
    train: np.ndarray, seen: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit the additive baseline every recommender starts from.

    Some users rate generously and some films are simply better; subtracting
    both effects first means the SVD spends its ranks on taste rather than on
    re-deriving those two offsets.

    Args:
        train: Ratings matrix, zero where unobserved.
        seen: Boolean mask of observed entries.

    Returns:
        ``(mu, user_bias, item_bias)``.
    """
    mu = float(train.sum() / seen.sum())
    resid = np.where(seen, train - mu, 0.0)

    user_counts = seen.sum(axis=1)
    user_bias = resid.sum(axis=1) / np.maximum(user_counts, 1)

    resid2 = np.where(seen, resid - user_bias[:, None], 0.0)
    item_counts = seen.sum(axis=0)
    item_bias = resid2.sum(axis=0) / np.maximum(item_counts, 1)

    return mu, user_bias, item_bias


def rmse_by_rank() -> dict[str, object]:
    """Score truncated SVD reconstructions against held-out ratings.

    Returns:
        A dict with ``ranks``, ``rmse``, ``baseline_rmse``, ``best_rank``,
        ``best_rmse`` and the shape/counts the post quotes.
    """
    ratings = dataset.movielens_ratings()
    n_users = int(ratings.user.max())
    n_items = int(ratings.item.max())

    rng = np.random.default_rng(SEED)
    is_test = rng.random(len(ratings)) < TEST_FRACTION

    train_df, test_df = ratings[~is_test], ratings[is_test]

    train = np.zeros((n_users, n_items))
    seen = np.zeros((n_users, n_items), dtype=bool)
    train[train_df.user - 1, train_df.item - 1] = train_df.rating
    seen[train_df.user - 1, train_df.item - 1] = True

    mu, user_bias, item_bias = _baseline(train, seen)
    fitted = mu + user_bias[:, None] + item_bias[None, :]

    # Unobserved cells become zero *after* centring, i.e. "no evidence either
    # way" rather than "rated zero stars". That is what makes a plain SVD a
    # usable collaborative filter on a 94% empty matrix.
    centred = np.where(seen, train - fitted, 0.0)

    u, s, vt = np.linalg.svd(centred, full_matrices=False)

    rows = test_df.user.to_numpy() - 1
    cols = test_df.item.to_numpy() - 1
    actual = test_df.rating.to_numpy().astype(float)
    base_pred = fitted[rows, cols]

    def score(pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((np.clip(pred, 1.0, 5.0) - actual) ** 2)))

    curve = []
    for k in RANKS:
        correction = np.einsum(
            "ij,j,ji->i", u[rows, :k], s[:k], vt[:k, :][:, cols]
        )
        curve.append(score(base_pred + correction))

    best = int(np.argmin(curve))
    return {
        "ranks": list(RANKS),
        "rmse": curve,
        "baseline_rmse": score(base_pred),
        "best_rank": RANKS[best],
        "best_rmse": curve[best],
        "n_users": n_users,
        "n_items": n_items,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "density": float(seen.sum() / seen.size),
    }


def main() -> int:
    """Print the curve."""
    out = rmse_by_rank()
    print(
        f"{out['n_users']} x {out['n_items']}, {out['density'] * 100:.1f}% observed, "
        f"{out['n_train']:,} train / {out['n_test']:,} test"
    )
    print(f"baseline (rank 0)  RMSE {out['baseline_rmse']:.4f}")
    for k, r in zip(out["ranks"], out["rmse"]):
        mark = "  <-- best" if k == out["best_rank"] else ""
        print(f"  k={k:4d}  RMSE {r:.4f}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
