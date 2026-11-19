"""Exhaustive subset fits, computed in batch from Gram matrices.

For centred data, the least-squares fit on a subset S has coefficients
b = G_SS^{-1} c_S with G = X'X and c = X'y, and residual sum of squares
y'y - c_S' b. Stacking every subset's 5x5 block and solving them together
fits all C(30, 5) = 142,506 regressions in a fraction of a second.
"""

from itertools import combinations

import numpy as np


def all_subsets(p: int, k: int) -> np.ndarray:
    """Every k-subset of range(p), one per row, in lexicographic order."""
    return np.array(list(combinations(range(p), k)), dtype=np.int16)


def _blocks(G, c, idx):
    Gs = G[idx[:, :, None], idx[:, None, :]]
    cs = c[idx]
    return Gs, cs


def subset_fits(Xtr, ytr, Xte, yte, k):
    """Train RSS, test MSE and coefficients of every k-variable OLS fit.

    Intercepts are handled by centring on the training means; the test data are
    centred with the same means, so the test error is that of the fitted
    intercept-plus-k-slopes model.
    """
    mx, my = Xtr.mean(0), ytr.mean()
    A, a = Xtr - mx, ytr - my
    B, b = Xte - mx, yte - my
    G, c = A.T @ A, A.T @ a
    Gt, ct = B.T @ B, B.T @ b
    idx = all_subsets(Xtr.shape[1], k)
    Gs, cs = _blocks(G, c, idx)
    coef = np.linalg.solve(Gs, cs[..., None])[..., 0]
    rss = a @ a - np.einsum("ij,ij->i", cs, coef)
    Gts, cts = _blocks(Gt, ct, idx)
    sse_te = (
        b @ b
        - 2 * np.einsum("ij,ij->i", cts, coef)
        + np.einsum("ij,ijk,ik->i", coef, Gts, coef)
    )
    intercept = my - np.einsum("ij,ij->i", coef, mx[idx])
    return {
        "idx": idx,
        "rss": rss,
        "test_mse": sse_te / len(yte),
        "coef": coef,
        "intercept": intercept,
    }


def best_score_subset(X, y, k):
    """Best k-subset for logistic regression by the global score statistic.

    This is the criterion behind SAS PROC LOGISTIC's best-subset search: the
    score chi-square of the subset, evaluated at the intercept-only fit. At that
    point it equals n * R^2 of the least-squares regression of y on the subset,
    so the best subset is the one with the smallest RSS for the 0/1 outcome.
    """
    fits = subset_fits(X, y.astype(float), X[:1], y[:1].astype(float), k)
    return fits["idx"][np.argmin(fits["rss"])]
