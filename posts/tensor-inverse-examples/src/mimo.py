"""FIR tensor fit and two inverses for a MIMO plant.

y(t) = sum_k H[:, :, k] @ u(t-k). H is output x input x lag. Fitting H is
ordinary least squares on a lagged design. Inverting H for a control input
is a different product, and two products disagree.
"""

from __future__ import annotations

import numpy as np

SEED = 7


def lagged_design(U: np.ndarray, n_lags: int) -> np.ndarray:
    """Rows are [u(t), u(t-1), ..., u(t-n_lags+1)] for t >= n_lags-1."""
    T, n_in = U.shape
    rows = T - n_lags + 1
    Phi = np.empty((rows, n_in * n_lags))
    for k in range(n_lags):
        Phi[:, k * n_in : (k + 1) * n_in] = U[n_lags - 1 - k : T - k]
    return Phi


def fit_fir(U: np.ndarray, Y: np.ndarray, n_lags: int) -> np.ndarray:
    """Least-squares H of shape (n_out, n_in, n_lags)."""
    Phi = lagged_design(U, n_lags)
    Yw = Y[n_lags - 1 :]
    # Phi @ vec_lag_major(H[o]) = Yw[:, o]
    coef, *_ = np.linalg.lstsq(Phi, Yw, rcond=None)
    n_in = U.shape[1]
    n_out = Y.shape[1]
    H = np.empty((n_out, n_in, n_lags))
    for o in range(n_out):
        H[o] = coef[:, o].reshape(n_lags, n_in).T
    return H


def predict(H: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Apply the FIR tensor to an input series. Leading lags are left as NaN."""
    n_out, n_in, n_lags = H.shape
    T = U.shape[0]
    Yhat = np.full((T, n_out), np.nan)
    Phi = lagged_design(U, n_lags)
    flat = np.empty((n_in * n_lags, n_out))
    for o in range(n_out):
        flat[:, o] = H[o].T.reshape(-1)
    Yhat[n_lags - 1 :] = Phi @ flat
    return Yhat


def fit_rmse(H: np.ndarray, U: np.ndarray, Y: np.ndarray) -> float:
    Yhat = predict(H, U)
    ok = np.isfinite(Yhat[:, 0])
    err = Yhat[ok] - Y[ok]
    return float(np.sqrt(np.mean(err**2)))


def invert_lag0(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Control inverse that only uses the instantaneous slice H[:, :, 0]."""
    return np.linalg.pinv(H[:, :, 0]) @ y


def invert_stacked(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Moore-Penrose inverse of the flattened (n_out) x (n_in * n_lags) map.

    The returned vector is length n_in * n_lags: one input for each lag.
    The plant is asked to produce y in one shot from a stacked window.
    """
    n_out, n_in, n_lags = H.shape
    # Column-major in lag, matching lagged_design: [u(t), u(t-1), ...].
    M = np.empty((n_out, n_in * n_lags))
    for k in range(n_lags):
        M[:, k * n_in : (k + 1) * n_in] = H[:, :, k]
    return np.linalg.pinv(M) @ y


def einstein_pinv_slice(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Einstein-product pseudoinverse treating H as a map (in x lag) -> out.

    Same numbers as invert_stacked: the Einstein unfolding of an
    (n_out) x (n_in x n_lags) tensor *is* that matrix. Kept as a named
    entry so the post can point at the product.
    """
    n_out, n_in, n_lags = H.shape
    A = H.reshape(n_out, n_in * n_lags)
    return np.linalg.pinv(A).reshape(n_in, n_lags)


def apply_window(H: np.ndarray, u_stacked: np.ndarray) -> np.ndarray:
    """Apply H to a stacked window [u(t), u(t-1), ...] of length n_in * n_lags."""
    n_out, n_in, n_lags = H.shape
    M = np.empty((n_out, n_in * n_lags))
    for k in range(n_lags):
        M[:, k * n_in : (k + 1) * n_in] = H[:, :, k]
    return M @ u_stacked.reshape(-1)
