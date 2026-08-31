"""Tensor products and the inverses they induce.

Four products, four inverses, plus the toy tensor the browser widget and the
post's prose both read. Importing this module in both places is what keeps the
widget's numbers and the post's numbers the same object.

Conventions follow Kolda and Bader (2009) for mode-n unfolding, Kilmer and
Martin (2011) for the t-product, and Brazell, Li, Navasca and Tamon (2013) for
the Einstein product.

Usage:
    from tinv import tprod, tinv, teye, einstein, einstein_inv, unfold
"""

from __future__ import annotations

import math

import numpy as np

SEED = 7

# Toy tensor for the widget and the stage-by-stage section.
TOY_N = 8
TOY_SLICES = 6


# --------------------------------------------------------------------------
# Mode-n product and mode-n pseudoinverse
# --------------------------------------------------------------------------


def unfold(X: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n unfolding: mode `mode` becomes rows, everything else columns."""
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)


def fold(M: np.ndarray, mode: int, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of `unfold` for a known target shape."""
    moved = [shape[mode]] + [s for i, s in enumerate(shape) if i != mode]
    return np.moveaxis(M.reshape(moved), 0, mode)


def mode_dot(X: np.ndarray, M: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n product X x_n M."""
    shape = list(X.shape)
    shape[mode] = M.shape[0]
    return fold(M @ unfold(X, mode), mode, tuple(shape))


def mode_pinv_solve(Y: np.ndarray, X: np.ndarray, mode: int) -> np.ndarray:
    """Regress along one mode: return A with Y ~ X x_mode A.

    `X` is the known design along `mode`; the returned A is
    `pinv(X) @ unfold(Y, mode)`, i.e. the mode-n pseudoinverse applied with
    every other mode stacked as samples.
    """
    return np.linalg.pinv(X) @ unfold(Y, mode)


# --------------------------------------------------------------------------
# t-product (third order) and the inverses it induces
# --------------------------------------------------------------------------


def tprod(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """t-product A * B: FFT along mode 3, per-slice matmul, inverse FFT."""
    Af = np.fft.fft(A, axis=2)
    Bf = np.fft.fft(B, axis=2)
    Cf = np.einsum("ikm,kjm->ijm", Af, Bf)
    out = np.fft.ifft(Cf, axis=2)
    return np.real(out) if np.isrealobj(A) and np.isrealobj(B) else out


def teye(n: int, n3: int) -> np.ndarray:
    """Identity under the t-product: identity in frontal slice 0, zeros after."""
    eye = np.zeros((n, n, n3))
    eye[:, :, 0] = np.eye(n)
    return eye


def tinv(A: np.ndarray) -> np.ndarray:
    """t-inverse: invert every Fourier slice, then transform back."""
    Af = np.fft.fft(A, axis=2)
    Xf = np.stack(
        [np.linalg.inv(Af[:, :, k]) for k in range(A.shape[2])],
        axis=2,
    )
    return np.real(np.fft.ifft(Xf, axis=2))


def tpinv(A: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """t-pseudoinverse: Moore-Penrose per Fourier slice, then transform back."""
    Af = np.fft.fft(A, axis=2)
    Xf = np.stack(
        [np.linalg.pinv(Af[:, :, k], rcond=rcond) for k in range(A.shape[2])],
        axis=2,
    )
    return np.real(np.fft.ifft(Xf, axis=2))


def tsolve(A: np.ndarray, B: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Solve A * X = B slice-wise in the Fourier domain, without forming A^-1."""
    n3 = A.shape[2]
    Af = np.fft.fft(A, axis=2)
    Bf = np.fft.fft(B, axis=2)
    Xf = np.empty((A.shape[1], B.shape[1], n3), dtype=complex)
    for k in range(n3):
        Xf[:, :, k] = np.linalg.pinv(Af[:, :, k], rcond=rcond) @ Bf[:, :, k]
    return np.real(np.fft.ifft(Xf, axis=2))


def treg_solve(A: np.ndarray, B: np.ndarray, lam: float) -> np.ndarray:
    """Tikhonov-regularised t-solve of A * X = B.

    Deconvolution is ill-posed, so both routes in the hyperspectral section get
    the same lambda: per Fourier slice, X = (A^H A + lam I)^-1 A^H B. That
    leaves the band coupling as the only difference between them.
    """
    n3 = A.shape[2]
    Af = np.fft.fft(A, axis=2)
    Bf = np.fft.fft(B, axis=2)
    Xf = np.empty((A.shape[1], B.shape[1], n3), dtype=complex)
    eye = np.eye(A.shape[1])
    for k in range(n3):
        M = Af[:, :, k]
        Xf[:, :, k] = np.linalg.solve(
            M.conj().T @ M + lam * eye,
            M.conj().T @ Bf[:, :, k],
        )
    return np.real(np.fft.ifft(Xf, axis=2))


def band_by_band_solve(A: np.ndarray, B: np.ndarray, lam: float) -> np.ndarray:
    """Undo the spatial blur only, band by band -- the flattening.

    This is what band-by-band 2D deconvolution does. It undoes the spatial blur
    and ignores every other frontal slice, so the cross-band coupling in the
    operator is simply dropped.
    """
    M = A[:, :, 0]
    eye = np.eye(M.shape[1])
    inv = np.linalg.solve(M.T @ M + lam * eye, M.T)
    return np.einsum("ij,jkb->ikb", inv, B)


def slice_conds(A: np.ndarray) -> np.ndarray:
    """Condition number of each Fourier slice of A."""
    Af = np.fft.fft(A, axis=2)
    return np.array([np.linalg.cond(Af[:, :, k]) for k in range(A.shape[2])])


# --------------------------------------------------------------------------
# Einstein product (even order) and the inverses it induces
# --------------------------------------------------------------------------


def einstein(A: np.ndarray, B: np.ndarray, n: int) -> np.ndarray:
    """Einstein product contracting the last n modes of A with the first n of B."""
    lead, mid = A.shape[:n], A.shape[n:]
    if B.shape[:n] != mid:
        raise ValueError(f"cannot contract {A.shape} with {B.shape} over {n} modes")
    tail = B.shape[n:]
    left = A.reshape(math.prod(lead), math.prod(mid))
    right = B.reshape(math.prod(mid), math.prod(tail))
    return (left @ right).reshape(lead + tail)


def einstein_eye(shape: tuple[int, ...]) -> np.ndarray:
    """Identity under the Einstein product on tensors of the given shape."""
    size = math.prod(shape)
    return np.eye(size).reshape(shape + shape)


def einstein_inv(A: np.ndarray, n: int) -> np.ndarray:
    """Einstein-product inverse of a square even-order tensor."""
    lead, mid = A.shape[:n], A.shape[n:]
    if math.prod(lead) != math.prod(mid):
        raise ValueError(f"{A.shape} is not square as an operator over {n} modes")
    flat = np.linalg.inv(A.reshape(math.prod(lead), math.prod(mid)))
    return flat.reshape(mid + lead)


def einstein_pinv(A: np.ndarray, n: int) -> np.ndarray:
    """Multilinear Moore-Penrose inverse under the Einstein product."""
    lead, mid = A.shape[:n], A.shape[n:]
    flat = np.linalg.pinv(A.reshape(math.prod(lead), math.prod(mid)))
    return flat.reshape(mid + lead)


# --------------------------------------------------------------------------
# The toy tensor the widget takes apart
# --------------------------------------------------------------------------


def toy_tensor(rng: np.random.Generator | None = None) -> np.ndarray:
    """Build an 8x8x6 tensor with well-separated Fourier slices.

    Frontal slice 0 is diagonally dominant so the operator is invertible; the
    later slices are small perturbations, which is what makes the six Fourier
    slices differ from one another without any of them being near-singular.
    """
    rng = rng or np.random.default_rng(SEED)
    A = np.zeros((TOY_N, TOY_N, TOY_SLICES))
    A[:, :, 0] = np.eye(TOY_N) * 2.4 + rng.normal(0, 0.35, (TOY_N, TOY_N))
    for k in range(1, TOY_SLICES):
        A[:, :, k] = rng.normal(0, 0.55 / k, (TOY_N, TOY_N))
    return A


def bend_toward_singular(
    A: np.ndarray,
    level: float,
    slice_index: int = 2,
) -> np.ndarray:
    """Drive one Fourier slice of A toward singular.

    `level` runs 0 (untouched) to 1, shrinking that slice's smallest singular
    value by a factor of 10**(12*level), so its condition number sweeps twelve
    orders of magnitude. The tensor comes back in the spatial domain, so
    everything downstream still sees an ordinary real tensor.

    Fourier slices 2 and 4 of a six-slice tensor are a conjugate pair, so the
    edit lands on both and the two condition numbers move together.
    """
    Af = np.fft.fft(A, axis=2)
    U, s, Vh = np.linalg.svd(Af[:, :, slice_index])
    s = s.copy()
    s[-1] = s[-1] * 10.0 ** (-12.0 * level)
    Af[:, :, slice_index] = U @ np.diag(s) @ Vh
    # Keep the spatial tensor real: mirror the edited slice onto its conjugate.
    mirror = (-slice_index) % A.shape[2]
    if mirror != slice_index:
        Af[:, :, mirror] = np.conj(Af[:, :, slice_index])
    return np.real(np.fft.ifft(Af, axis=2))


def widget_state(
    level: float,
    solver: str,
    slice_index: int = 2,
    noise: float = 1e-6,
) -> dict:
    """Every picture and number the widget shows for one conditioning/solver pair.

    Three readings, because they disagree and the disagreement is the point:

    - `residual` checks the identity, ||A * X - I|| / ||I||.
    - `max_abs` is the largest entry of X, where a true inverse blows up.
    - `solve_error` recovers a known X_true from a slightly noisy B = A * X_true.
      That is the reading a practitioner cares about, and the only one on which
      the pseudoinverse wins.
    """
    rng = np.random.default_rng(SEED + 1)
    A = bend_toward_singular(toy_tensor(), level, slice_index)
    inverse = tinv(A) if solver == "inv" else tpinv(A, rcond=1e-8)
    finverse = np.fft.fft(inverse, axis=2)
    product = tprod(A, inverse)
    identity = teye(TOY_N, TOY_SLICES)
    residual = float(np.linalg.norm(product - identity) / np.linalg.norm(identity))

    x_true = rng.normal(0, 1, (TOY_N, 1, TOY_SLICES))
    b = tprod(A, x_true)
    b = b + rng.normal(0, noise * np.abs(b).max(), b.shape)
    x_hat = tprod(inverse, b)
    solve_error = float(np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true))

    return {
        "spatial": A,
        "fourier": np.abs(np.fft.fft(A, axis=2)),
        "conds": slice_conds(A),
        "finverse": np.abs(finverse),
        "inverse": inverse,
        "product": product,
        "residual": residual,
        "max_abs": float(np.abs(inverse).max()),
        "solve_error": solve_error,
    }
