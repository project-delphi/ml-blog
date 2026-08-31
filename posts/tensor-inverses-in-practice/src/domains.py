"""The four problems the post inverts, one builder each.

Kept out of index.qmd so the post shows the inversion rather than the setup.
Every function here is deterministic given SEED in tinv.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tinv import SEED

DATA = Path(__file__).resolve().parent.parent / "data"

# Voigt index map: 0->11, 1->22, 2->33, 3->23, 4->13, 5->12.
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
VOIGT_OF = {}
for _a, (_i, _j) in enumerate(VOIGT):
    VOIGT_OF[(_i, _j)] = _a
    VOIGT_OF[(_j, _i)] = _a

# T300/5208 unidirectional carbon/epoxy ply, room temperature, in Pa.
# Standard reference values, transversely isotropic about the fibre axis.
PLY = {
    "E1": 181.0e9,
    "E2": 10.3e9,
    "G12": 7.17e9,
    "nu12": 0.28,
    "nu23": 0.35,
}


# --------------------------------------------------------------------------
# 1. Continuum mechanics: stiffness and compliance
# --------------------------------------------------------------------------


def voigt_compliance(ply: dict | None = None) -> np.ndarray:
    """Return the 6x6 Voigt compliance of the ply, in engineering shear strains."""
    p = ply or PLY
    e1, e2, g12 = p["E1"], p["E2"], p["G12"]
    nu12, nu23 = p["nu12"], p["nu23"]
    g23 = e2 / (2 * (1 + nu23))
    s = np.zeros((6, 6))
    s[0, 0] = 1 / e1
    s[1, 1] = s[2, 2] = 1 / e2
    s[0, 1] = s[1, 0] = s[0, 2] = s[2, 0] = -nu12 / e1
    s[1, 2] = s[2, 1] = -nu23 / e2
    s[3, 3] = 1 / g23
    s[4, 4] = s[5, 5] = 1 / g12
    return s


def stiffness_to_tensor(c6: np.ndarray) -> np.ndarray:
    """Voigt 6x6 stiffness -> 3x3x3x3 tensor. No factors: stiffness maps straight."""
    c4 = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    c4[i, j, k, m] = c6[VOIGT_OF[(i, j)], VOIGT_OF[(k, m)]]
    return c4


def compliance_to_tensor(s6: np.ndarray, reuter: bool = True) -> np.ndarray:
    """Voigt 6x6 compliance -> 3x3x3x3 tensor.

    Compliance needs the Reuter factors: divide by 2 for each shear index,
    because the Voigt form carries engineering shear strains and the tensor
    form carries tensor strains. Pass `reuter=False` to reproduce the mistake
    the post measures.
    """
    s4 = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    a, b = VOIGT_OF[(i, j)], VOIGT_OF[(k, m)]
                    factor = (2 if a >= 3 else 1) * (2 if b >= 3 else 1)
                    s4[i, j, k, m] = s6[a, b] / (factor if reuter else 1)
    return s4


def rotate4(t4: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate a fourth-order tensor about axis 3 by `degrees`."""
    th = np.radians(degrees)
    c, s = np.cos(th), np.sin(th)
    q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.einsum("ip,jq,kr,ls,pqrs->ijkl", q, q, q, q, t4)


def off_axis_stiffness(degrees: float = 30.0) -> np.ndarray:
    """Return the ply's stiffness tensor, rotated off the fibre axis."""
    return rotate4(stiffness_to_tensor(np.linalg.inv(voigt_compliance())), degrees)


def tensor_to_voigt_stiffness(c4: np.ndarray) -> np.ndarray:
    """3x3x3x3 stiffness -> Voigt 6x6, the inverse of `stiffness_to_tensor`."""
    return np.array(
        [[c4[i, j, k, m] for (k, m) in VOIGT] for (i, j) in VOIGT],
    )


def applied_stress(magnitude: float = 120.0e6) -> np.ndarray:
    """Return a stress state with both normal and shear components, in Pa."""
    sigma = np.zeros((3, 3))
    sigma[0, 0] = magnitude
    sigma[1, 1] = 0.35 * magnitude
    sigma[0, 1] = sigma[1, 0] = 0.45 * magnitude
    return sigma


# --------------------------------------------------------------------------
# 2. Hyperspectral restoration: a t-product blur
# --------------------------------------------------------------------------


def pavia_crop() -> tuple[np.ndarray, dict]:
    """Load the committed 128x128x103 Pavia University crop and its metadata."""
    with np.load(DATA / "paviaU_crop.npz") as f:
        return f["cube"], {
            "origin": f["crop_origin"].tolist(),
            "full_shape": f["full_shape"].tolist(),
            "source": str(f["source"]),
            "retrieved": str(f["retrieved"]),
        }


def blur_operator(
    n: int,
    bands: int,
    width: float = 2.2,
    leak: float = 0.22,
) -> np.ndarray:
    """Build a t-product blur: spatial smear in slice 0, spectral leak in slices +-1.

    Frontal slice 0 is a circulant Gaussian smear along the row mode, the
    optical blur. Slices 1 and n3-1 carry the same smear scaled by `leak`,
    which under the t-product makes each output band a mixture of its
    neighbours -- spectral crosstalk of the kind a real pushbroom sensor has.
    """
    offsets = np.arange(n)
    offsets = np.minimum(offsets, n - offsets)
    kernel = np.exp(-0.5 * (offsets / width) ** 2)
    kernel /= kernel.sum()
    smear = np.array([np.roll(kernel, i) for i in range(n)])
    op = np.zeros((n, n, bands))
    op[:, :, 0] = smear
    op[:, :, 1] = leak * smear
    op[:, :, -1] = leak * smear
    return op


def spectral_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel spectral angle in degrees between two cubes."""
    x = a.reshape(-1, a.shape[-1])
    y = b.reshape(-1, b.shape[-1])
    num = np.sum(x * y, axis=1)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12
    return np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0)))


# --------------------------------------------------------------------------
# 3. Spatiotemporal neuroimaging: mode-n regression
# --------------------------------------------------------------------------


def synthetic_scan(
    side: int = 16,
    frames: int = 120,
    noise: float = 0.4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a 4D scan with a known mixing map: (side, side, side, frames).

    Two boxcar task regressors plus a linear drift and an intercept. Each
    regressor has its own spatial blob, so the true coefficient map is known
    and a recovery can be scored against it.
    """
    rng = rng or np.random.default_rng(SEED)
    t = np.arange(frames)
    design = np.column_stack(
        [
            np.ones(frames),
            ((t // 12) % 2).astype(float),
            ((t // 20) % 2).astype(float),
            np.linspace(-1, 1, frames),
        ],
    )
    grid = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), axis=-1)

    def blob(centre, width):
        d2 = ((grid - np.array(centre)) ** 2).sum(axis=-1)
        return np.exp(-d2 / (2 * width**2))

    beta = np.stack(
        [
            0.5 * np.ones((side, side, side)),
            2.0 * blob((5, 5, 8), 2.4),
            1.6 * blob((11, 10, 7), 2.0),
            0.3 * blob((8, 8, 8), 6.0),
        ],
    )
    clean = np.einsum("tp,pxyz->xyzt", design, beta)
    scan = clean + rng.normal(0, noise, clean.shape)
    return scan, design, beta


# --------------------------------------------------------------------------
# 4. Urban demand: a Kronecker-separable precision
# --------------------------------------------------------------------------


def chicago_counts() -> tuple[np.ndarray, dict]:
    """Load the committed weeks x community area x offence type array."""
    with np.load(DATA / "chicago_counts.npz") as f:
        return f["counts"], {
            "types": [str(t) for t in f["types"]],
            "areas": f["areas"].tolist(),
            "year": int(f["year"]),
            "n_incidents": int(f["n_incidents"]),
            "source": str(f["source"]),
            "retrieved": str(f["retrieved"]),
        }


def _whiten(Y: np.ndarray, covs: list[np.ndarray], skip: int) -> np.ndarray:
    """Whiten every non-sample mode of Y except `skip`, using its covariance."""
    Z = Y
    for j, cov in enumerate(covs):
        if j == skip:
            continue
        w = np.linalg.inv(np.linalg.cholesky(cov))
        Z = np.moveaxis(np.tensordot(w, Z, axes=([1], [j + 1])), 0, j + 1)
    return Z


def flip_flop(Y: np.ndarray, iters: int = 50, ridge: float = 1e-8) -> list[np.ndarray]:
    """Maximum-likelihood mode covariances of a tensor-normal array.

    `Y` is (samples, d1, ..., dK) and already centred. Returns one covariance
    per non-sample mode, so the model is Sigma = Sigma_1 kron ... kron Sigma_K.
    Dutilleul's flip-flop algorithm past two modes: whiten every other mode,
    take the sample covariance of the mode being updated, repeat.

    The scale is shared between modes, so every mode after the first is
    normalised to unit mean variance and mode 0 carries the overall scale.
    """
    dims = Y.shape[1:]
    covs = [np.eye(d) for d in dims]
    for _ in range(iters):
        for k, d in enumerate(dims):
            Z = _whiten(Y, covs, skip=k)
            M = np.moveaxis(Z, k + 1, 0).reshape(d, -1)
            cov = M @ M.T / (M.shape[1])
            cov += ridge * np.trace(cov) / d * np.eye(d)
            if k > 0:
                cov *= d / np.trace(cov)
            covs[k] = cov
    return covs


def separable_logdet(covs: list[np.ndarray]) -> float:
    """log|Sigma_1 kron ... kron Sigma_K| without forming the Kronecker product."""
    total = np.prod([c.shape[0] for c in covs])
    out = 0.0
    for cov in covs:
        d = cov.shape[0]
        out += (total // d) * np.linalg.slogdet(cov)[1]
    return float(out)


def separable_quadform(Z: np.ndarray, covs: list[np.ndarray]) -> np.ndarray:
    """Per-sample x^T (Sigma_1 kron ... )^-1 x, computed mode by mode."""
    W = Z
    for j, cov in enumerate(covs):
        w = np.linalg.inv(np.linalg.cholesky(cov))
        W = np.moveaxis(np.tensordot(w, W, axes=([1], [j + 1])), 0, j + 1)
    return (W**2).reshape(Z.shape[0], -1).sum(axis=1)
