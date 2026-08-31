"""Fourth-order cumulants and a small JADE unmixer.

The mixing matrix is known in this post, so we can score recovery. The
leftover that stays after a successful run is permutation and scale: those
are not fixed by the cumulant tensor.
"""

from __future__ import annotations

import numpy as np

SEED = 7


def centre(X: np.ndarray) -> np.ndarray:
    """X is channels x time."""
    return X - X.mean(axis=1, keepdims=True)


def whiten(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return whitened data and the whitening matrix W with W @ cov(X) @ W.T = I."""
    X = centre(X)
    cov = (X @ X.T) / X.shape[1]
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    W = (evecs / np.sqrt(evals)) @ evecs.T
    return W @ X, W


def fourth_cumulant(Z: np.ndarray) -> np.ndarray:
    """C_ijkl of whitened Z (n x T). For whitened data the pairwise terms are deltas."""
    n, T = Z.shape
    C = np.einsum("it,jt,kt,lt->ijkl", Z, Z, Z, Z) / T
    eye = np.eye(n)
    C = C - np.einsum("ij,kl->ijkl", eye, eye)
    C = C - np.einsum("ik,jl->ijkl", eye, eye)
    C = C - np.einsum("il,jk->ijkl", eye, eye)
    return C


def _givens(A: list[np.ndarray], i: int, j: int) -> tuple[float, float]:
    """Jacobi angle that jointly diagonalizes the (i, j) pair of the slices."""
    g11 = g12 = g22 = 0.0
    for M in A:
        aii, ajj, aij = M[i, i], M[j, j], M[i, j]
        g11 += (aii - ajj) ** 2
        g12 += 2.0 * aij * (aii - ajj)
        g22 += 4.0 * aij**2
    # tan(2θ) from the 2x2 eigenproblem on G.
    if abs(g12) + abs(g11 - g22) < 1e-18:
        return 1.0, 0.0
    eig_off = 0.5 * np.arctan2(2.0 * g12, g11 - g22)
    return float(np.cos(eig_off)), float(np.sin(eig_off))


def joint_diagonalize(mats: list[np.ndarray], sweeps: int = 20) -> np.ndarray:
    """Orthogonal joint diagonalizer of a list of symmetric n x n matrices."""
    n = mats[0].shape[0]
    V = np.eye(n)
    work = [M.copy() for M in mats]
    for _ in range(sweeps):
        for i in range(n):
            for j in range(i + 1, n):
                c, s = _givens(work, i, j)
                if abs(s) < 1e-15:
                    continue
                G = np.array([[c, s], [-s, c]])
                V[:, [i, j]] = V[:, [i, j]] @ G
                for M in work:
                    M[[i, j], :] = G.T @ M[[i, j], :]
                    M[:, [i, j]] = M[:, [i, j]] @ G
    return V


def jade(X: np.ndarray, sweeps: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JADE on channels x time. Returns sources, unmixing, cumulant tensor."""
    Z, W = whiten(X)
    n = Z.shape[0]
    C = fourth_cumulant(Z)
    # Eigenmatrices of the cumulant, one per (p, q) with p <= q.
    slices = []
    for p in range(n):
        for q in range(p, n):
            slices.append(0.5 * (C[:, :, p, q] + C[:, :, q, p]))
    V = joint_diagonalize(slices, sweeps=sweeps)
    # Sources = V.T @ Z. Unmixing of the original mix is V.T @ W.
    S = V.T @ Z
    unmix = V.T @ W
    return S, unmix, C


def align(recovered: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match recovered rows to truth by absolute correlation. Return perm, signs, scales."""
    n = truth.shape[0]
    corr = np.corrcoef(recovered, truth)[:n, n:]
    used = set()
    perm = np.zeros(n, dtype=int)
    signs = np.ones(n)
    scales = np.ones(n)
    for i in np.argsort(-np.max(np.abs(corr), axis=1)):
        j = int(np.argmax(np.abs(corr[i])))
        while j in used:
            corr[i, j] = 0.0
            j = int(np.argmax(np.abs(corr[i])))
        used.add(j)
        perm[i] = j
        signs[i] = 1.0 if corr[i, j] >= 0 else -1.0
        num = recovered[i] @ (signs[i] * truth[j])
        den = recovered[i] @ recovered[i]
        scales[i] = num / den if abs(den) > 1e-18 else 1.0
    return perm, signs, scales


def sir_db(recovered: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Signal-to-interference ratio of each aligned source, in decibels."""
    perm, signs, scales = align(recovered, truth)
    out = []
    for i, (j, sgn, sc) in enumerate(zip(perm, signs, scales)):
        target = sgn * sc * recovered[i]
        err = truth[j] - target
        out.append(10.0 * np.log10((truth[j] @ truth[j]) / (err @ err + 1e-18)))
    return np.asarray(out)


def pca_sources(X: np.ndarray) -> np.ndarray:
    """PCA 'unmix': the whitened channels, no further rotation."""
    Z, _ = whiten(X)
    return Z


def offdiag_energy(C: np.ndarray) -> float:
    """Fraction of ||C||^2 that is not on the superdiagonal C_iiii."""
    n = C.shape[0]
    tot = float((C**2).sum())
    diag = float(sum(C[i, i, i, i] ** 2 for i in range(n)))
    return 1.0 - diag / (tot + 1e-18)


def signed_perm(Ainv: np.ndarray, perm: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Another inverse of the same mix: a signed permutation of A^{-1}."""
    return signs[:, None] * Ainv[perm]


def fastica(X: np.ndarray, n_iter: int = 200, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric kurtosis FastICA on channels x time. Picks one leftover rotation."""
    Z, Ww = whiten(X)
    n, T = Z.shape
    rng = np.random.default_rng(seed)
    W, _ = np.linalg.qr(rng.normal(size=(n, n)))
    for _ in range(n_iter):
        Y = W @ Z
        W = (Y**3) @ Z.T / T - 3.0 * W
        W, _ = np.linalg.qr(W)
    return W @ Z, W @ Ww
