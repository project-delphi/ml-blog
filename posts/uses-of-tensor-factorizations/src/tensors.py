"""Synthetic CP kernels, TT-matrices, and a three-way mixing cube.

Seeded generators plus the closed-form parameter counts the post quotes.
Fitting lives here so ``export_widget_data.py`` and the Quarto document cannot
drift apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

SEED: int = 7
NOISE_FRAC: float = 0.08

# VGG-16 conv5 stand-in (small enough to factorize at render).
CONV_D: int = 3
CONV_C: int = 64
CONV_TRUE_RANK: int = 16
CONV_RANKS: tuple[int, ...] = (4, 8, 16, 24, 32, 48, 64)

# Headline VGG-16 conv5 numbers (closed form; not factorized).
HEAD_D: int = 3
HEAD_C: int = 512
HEAD_CP_RANK: int = 64

# Transformer W_O stand-in, modes 4^4.
TT_MODE: int = 4
TT_ORDER: int = 4
TT_TRUE_RANK: int = 4
TT_RANKS: tuple[int, ...] = (1, 2, 4, 6, 8, 12, 16)
SVD_RANKS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

# Headline transformer W_O, modes 8^4.
HEAD_TT_MODE: int = 8
HEAD_TT_ORDER: int = 4
HEAD_TT_RANK: int = 16

MIX_SHAPE: tuple[int, int, int] = (20, 24, 18)
MIX_RANK: int = 3


def cp_conv_params(d: int, cin: int, cout: int, rank: int) -> int:
    """Parameter count of a CP factorization of a 2-D conv kernel."""
    return int(rank * (2 * d + cin + cout))


def dense_conv_params(d: int, cin: int, cout: int) -> int:
    """Parameter count of a dense 2-D conv kernel."""
    return int(d * d * cin * cout)


def tt_matrix_params(ms: list[int], ns: list[int], rank: int) -> int:
    """Parameter count of a TT-matrix with uniform internal rank."""
    order = len(ms)
    total = 0
    for k in range(order):
        r_left = 1 if k == 0 else rank
        r_right = 1 if k == order - 1 else rank
        total += r_left * ms[k] * ns[k] * r_right
    return int(total)


def svd_params(rows: int, cols: int, rank: int) -> int:
    """Parameter count of a truncated SVD (U, σ, V stored)."""
    return int(rank * (rows + cols + 1))


def rel_fro(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Relative Frobenius error ``||a - b|| / ||a||``."""
    denom = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / denom)


def make_cp_kernel(
    rng: np.random.Generator,
    d: int = CONV_D,
    channels: int = CONV_C,
    rank: int = CONV_TRUE_RANK,
    noise_frac: float = NOISE_FRAC,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Rank-``rank`` CP kernel plus i.i.d. Gaussian noise."""
    factors = [
        rng.normal(size=(d, rank)),
        rng.normal(size=(d, rank)),
        rng.normal(size=(channels, rank)),
        rng.normal(size=(channels, rank)),
    ]
    clean = cp_to_tensor((np.ones(rank), factors))
    noise = noise_frac * float(clean.std()) * rng.normal(size=clean.shape)
    return clean + noise, clean


def fit_cp(tensor: NDArray[np.floating], rank: int) -> NDArray[np.floating]:
    """ALS CP fit; weights absorbed into the reconstruction."""
    weights, factors = parafac(
        tensor, rank=rank, n_iter_max=120, init="svd", random_state=SEED
    )
    return cp_to_tensor((weights, factors))


def random_tt_cores(
    rng: np.random.Generator,
    ms: list[int],
    ns: list[int],
    rank: int,
) -> list[NDArray[np.floating]]:
    """Gaussian TT-matrix cores with uniform internal rank."""
    order = len(ms)
    cores: list[NDArray[np.floating]] = []
    for k in range(order):
        r_left = 1 if k == 0 else rank
        r_right = 1 if k == order - 1 else rank
        core = rng.normal(size=(r_left, ms[k], ns[k], r_right))
        cores.append(core / np.sqrt(core.size))
    return cores


def tt_matrix_to_dense(cores: list[NDArray[np.floating]]) -> NDArray[np.floating]:
    """Contract TT-matrix cores to an ``(M, N)`` matrix."""
    acc = cores[0][0]
    for core in cores[1:]:
        acc = np.tensordot(acc, core, axes=([-1], [0]))
        rows, cols, m_k, n_k, r_next = acc.shape
        acc = np.transpose(acc, (0, 2, 1, 3, 4)).reshape(
            rows * m_k, cols * n_k, r_next
        )
    return np.asarray(acc[..., 0])


def make_tt_matrix(
    rng: np.random.Generator,
    ms: list[int],
    ns: list[int],
    rank: int = TT_TRUE_RANK,
    noise_frac: float = NOISE_FRAC,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """TT-matrix of the given rank plus i.i.d. Gaussian noise."""
    clean = tt_matrix_to_dense(random_tt_cores(rng, ms, ns, rank))
    noise = noise_frac * float(clean.std()) * rng.normal(size=clean.shape)
    return clean + noise, clean


def tt_matrix_svd(
    matrix: NDArray[np.floating],
    ms: list[int],
    ns: list[int],
    max_rank: int,
) -> list[NDArray[np.floating]]:
    """Left-orthogonal TT-SVD of a matrix with the given mode sizes."""
    order = len(ms)
    tensor = matrix.reshape(tuple(ms) + tuple(ns))
    axes = [ax for k in range(order) for ax in (k, order + k)]
    remaining: NDArray[np.floating] = np.transpose(tensor, axes)
    cores: list[NDArray[np.floating]] = []
    ranks = [1]
    for k in range(order - 1):
        r_left = ranks[-1]
        m_k, n_k = ms[k], ns[k]
        rest = remaining.size // (r_left * m_k * n_k)
        unfolding = remaining.reshape(r_left * m_k * n_k, rest)
        u, s, vt = np.linalg.svd(unfolding, full_matrices=False)
        keep = min(max_rank, u.shape[1])
        cores.append(u[:, :keep].reshape(r_left, m_k, n_k, keep))
        remaining = s[:keep, None] * vt[:keep, :]
        ranks.append(keep)
    r_left = ranks[-1]
    m_k, n_k = ms[-1], ns[-1]
    cores.append(remaining.reshape(r_left, m_k, n_k, 1))
    return cores


def truncated_svd(
    matrix: NDArray[np.floating], rank: int
) -> NDArray[np.floating]:
    """Eckart–Young rank-``rank`` approximation."""
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = min(rank, s.size)
    return (u[:, :keep] * s[:keep]) @ vt[:keep, :]


def _bump(n: int, center: float, width: float) -> NDArray[np.floating]:
    grid = np.linspace(0.0, 1.0, n)
    return np.exp(-0.5 * ((grid - center) / width) ** 2)


def make_mixing_cube(
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating], list[NDArray[np.floating]]]:
    """Rank-3 CP cube (sample × emission × excitation) plus noise."""
    n_sample, n_em, n_ex = MIX_SHAPE
    # Overlapping concentrations, separated emission and excitation — the
    # fluorescence layout where unfolding mixes samples and CP does not.
    sample_centres, sample_width = (0.32, 0.50, 0.68), 0.20
    em_centres, em_width = (0.18, 0.50, 0.82), 0.09
    ex_centres, ex_width = (0.20, 0.52, 0.84), 0.09
    factors = [
        np.zeros((n_sample, MIX_RANK)),
        np.zeros((n_em, MIX_RANK)),
        np.zeros((n_ex, MIX_RANK)),
    ]
    terms = []
    for r in range(MIX_RANK):
        a = _bump(n_sample, sample_centres[r], sample_width)
        b = _bump(n_em, em_centres[r], em_width)
        c = _bump(n_ex, ex_centres[r], ex_width)
        factors[0][:, r] = a
        factors[1][:, r] = b
        factors[2][:, r] = c
        terms.append(a[:, None, None] * b[None, :, None] * c[None, None, :])
    clean = sum(terms)
    noise = NOISE_FRAC * float(clean.std()) * rng.normal(size=clean.shape)
    return clean + noise, factors


def align_factors(
    true: NDArray[np.floating], est: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Permute and sign-flip ``est`` columns to match ``true``."""
    t_norm = true / np.linalg.norm(true, axis=0, keepdims=True)
    e_norm = est / np.linalg.norm(est, axis=0, keepdims=True)
    corr = t_norm.T @ e_norm
    row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
    order = np.empty(est.shape[1], dtype=int)
    order[row_ind] = col_ind
    aligned = est[:, order].copy()
    for r in range(est.shape[1]):
        sign = np.sign(np.dot(true[:, r], aligned[:, r]))
        if sign == 0:
            sign = 1.0
        aligned[:, r] *= sign
    a_norm = aligned / np.linalg.norm(aligned, axis=0, keepdims=True)
    col_corr = np.abs((t_norm * a_norm).sum(axis=0))
    return aligned, col_corr


def sweep_cp(kernel: NDArray[np.floating]) -> dict[str, Any]:
    """CP rank sweep on a 4-D conv kernel."""
    d, _, cin, cout = kernel.shape
    dense = dense_conv_params(d, cin, cout)
    ranks = list(CONV_RANKS)
    params = [cp_conv_params(d, cin, cout, r) for r in ranks]
    errors = [rel_fro(kernel, fit_cp(kernel, r)) for r in ranks]
    return {
        "d": d,
        "cin": cin,
        "cout": cout,
        "true_rank": CONV_TRUE_RANK,
        "noise_frac": NOISE_FRAC,
        "dense_params": dense,
        "ranks": ranks,
        "params": params,
        "rel_error": errors,
        "compression": [dense / p for p in params],
        "headline": {
            "d": HEAD_D,
            "cin": HEAD_C,
            "cout": HEAD_C,
            "rank": HEAD_CP_RANK,
            "dense_params": dense_conv_params(HEAD_D, HEAD_C, HEAD_C),
            "cp_params": cp_conv_params(HEAD_D, HEAD_C, HEAD_C, HEAD_CP_RANK),
            "params": [cp_conv_params(HEAD_D, HEAD_C, HEAD_C, r) for r in ranks],
        },
    }


def sweep_tt(matrix: NDArray[np.floating], ms: list[int], ns: list[int]) -> dict[str, Any]:
    """TT-matrix rank sweep plus truncated-SVD baseline."""
    rows, cols = matrix.shape
    dense = rows * cols
    ranks = list(TT_RANKS)
    params = [tt_matrix_params(ms, ns, r) for r in ranks]
    errors = [
        rel_fro(matrix, tt_matrix_to_dense(tt_matrix_svd(matrix, ms, ns, r)))
        for r in ranks
    ]
    svd_ranks = list(SVD_RANKS)
    svd_p = [svd_params(rows, cols, k) for k in svd_ranks]
    svd_err = [rel_fro(matrix, truncated_svd(matrix, k)) for k in svd_ranks]
    head_ms = [HEAD_TT_MODE] * HEAD_TT_ORDER
    head_ns = [HEAD_TT_MODE] * HEAD_TT_ORDER
    head_dense = HEAD_TT_MODE ** (2 * HEAD_TT_ORDER)
    return {
        "ms": ms,
        "ns": ns,
        "true_rank": TT_TRUE_RANK,
        "noise_frac": NOISE_FRAC,
        "dense_params": dense,
        "ranks": ranks,
        "params": params,
        "rel_error": errors,
        "compression": [dense / p for p in params],
        "svd_ranks": svd_ranks,
        "svd_params": svd_p,
        "svd_rel_error": svd_err,
        "headline": {
            "ms": head_ms,
            "ns": head_ns,
            "rank": HEAD_TT_RANK,
            "dense_params": head_dense,
            "tt_params": tt_matrix_params(head_ms, head_ns, HEAD_TT_RANK),
            "params": [tt_matrix_params(head_ms, head_ns, r) for r in ranks],
        },
    }


def mixing_fit(
    cube: NDArray[np.floating], true_factors: list[NDArray[np.floating]]
) -> dict[str, Any]:
    """Rank-3 CP vs mode-0 unfolding SVD on the mixing cube."""
    weights, factors = parafac(
        cube, rank=MIX_RANK, n_iter_max=200, init="svd", random_state=SEED
    )
    recon = cp_to_tensor((weights, factors))
    aligned = []
    corrs = []
    for true, est in zip(true_factors, factors):
        got, corr = align_factors(true, est)
        aligned.append(got)
        corrs.append(corr.tolist())
    unfolding = cube.reshape(cube.shape[0], -1)
    u, _, _ = np.linalg.svd(unfolding, full_matrices=False)
    svd_factors = u[:, :MIX_RANK]
    _, svd_corr = align_factors(true_factors[0], svd_factors)
    return {
        "cp_rel_error": rel_fro(cube, recon),
        "cp_sample": aligned[0].tolist(),
        "true_sample": true_factors[0].tolist(),
        "svd_sample": svd_factors.tolist(),
        "cp_emission": aligned[1].tolist(),
        "true_emission": true_factors[1].tolist(),
        "cp_corr": corrs[0],
        "cp_em_corr": corrs[1],
        "svd_corr": svd_corr.tolist(),
        "mean_cp_corr": float(np.mean(corrs[0])),
        "mean_cp_em_corr": float(np.mean(corrs[1])),
        "mean_svd_corr": float(np.mean(svd_corr)),
    }
