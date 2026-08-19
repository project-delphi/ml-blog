"""The photograph, its SVD, and the two numbers a compression claim needs.

Everything the post says about image compression comes through here, so the
figures, the widget payload and the closing paragraph cannot drift apart: they
all call the same functions.

The image is ``skimage.data.astronaut()`` -- a NASA photograph of astronaut
Eileen Collins, public domain, shipped inside scikit-image. It is converted to
grayscale so the post factors one matrix instead of three.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from skimage import data
from skimage.color import rgb2gray

# Ranks the widget can reach. 100 is past the point of diminishing returns for
# this photograph and keeps the shipped payload well inside its size budget.
MAX_RANK: Final[int] = 100

# The two storage models the post prices. float32 is what you get by saving the
# factors naively; int16 is what the widget actually ships, and is lossless
# enough here to be invisible.
BYTES_FLOAT32: Final[int] = 4
BYTES_INT16: Final[int] = 2


def load_image() -> np.ndarray:
    """Return the photograph as a float array in ``[0, 255]``.

    Returns:
        A ``(512, 512)`` grayscale array.
    """
    return rgb2gray(data.astronaut()) * 255.0


def svd(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full thin SVD of the image.

    Args:
        image: The ``(m, n)`` pixel array.

    Returns:
        ``(u, s, vt)`` with ``s`` in descending order.
    """
    return np.linalg.svd(image, full_matrices=False)


def reconstruct(
    u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int
) -> np.ndarray:
    """Rebuild the image from its leading ``k`` singular triplets.

    Args:
        u: Left singular vectors.
        s: Singular values.
        vt: Right singular vectors, transposed.
        k: Rank to truncate at.

    Returns:
        The rank-``k`` approximation, clipped to the displayable range.
    """
    return np.clip(u[:, :k] * s[:k] @ vt[:k], 0.0, 255.0)


def psnr(original: np.ndarray, approx: np.ndarray) -> float:
    """Peak signal-to-noise ratio in decibels, against an 8-bit peak.

    Args:
        original: Reference image on a 0-255 scale.
        approx: Reconstruction on the same scale.

    Returns:
        PSNR in dB; ``inf`` for an exact match.
    """
    mse = float(np.mean((original - approx) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def stored_bytes(m: int, n: int, k: int, element_bytes: int) -> int:
    """Bytes needed to hold a rank-``k`` factorisation.

    Counts every number that must be written down: ``U_k`` is ``m*k``, ``V_k``
    is ``n*k``, and the ``k`` singular values.

    Args:
        m: Image height.
        n: Image width.
        k: Truncation rank.
        element_bytes: Width of one stored number.

    Returns:
        Byte count.
    """
    return element_bytes * k * (m + n + 1)


def original_bytes(m: int, n: int) -> int:
    """Bytes in the raw 8-bit pixel array -- the baseline every ratio uses."""
    return m * n


def quality_table(
    image: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
    ranks: np.ndarray,
) -> dict[str, np.ndarray]:
    """Price every rank in ``ranks``: quality on one axis, storage on the other.

    Args:
        image: The reference photograph.
        u: Left singular vectors.
        s: Singular values.
        vt: Right singular vectors, transposed.
        ranks: Ranks to evaluate.

    Returns:
        Arrays keyed ``k``, ``psnr``, ``energy`` (fraction of squared Frobenius
        norm retained), ``frobenius`` (relative error), and the two storage
        fractions ``frac_f32`` and ``frac_i16``, each as a fraction of the raw
        pixel array.
    """
    m, n = image.shape
    raw = original_bytes(m, n)
    total_energy = float(np.sum(s**2))

    rows = {key: [] for key in ("psnr", "energy", "frobenius")}
    for k in ranks:
        approx = reconstruct(u, s, vt, int(k))
        rows["psnr"].append(psnr(image, approx))
        rows["energy"].append(float(np.sum(s[:k] ** 2)) / total_energy)
        rows["frobenius"].append(
            float(np.linalg.norm(image - approx) / np.linalg.norm(image))
        )

    return {
        "k": np.asarray(ranks, dtype=int),
        "psnr": np.array(rows["psnr"]),
        "energy": np.array(rows["energy"]),
        "frobenius": np.array(rows["frobenius"]),
        "frac_f32": np.array(
            [stored_bytes(m, n, int(k), BYTES_FLOAT32) / raw for k in ranks]
        ),
        "frac_i16": np.array(
            [stored_bytes(m, n, int(k), BYTES_INT16) / raw for k in ranks]
        ),
    }


def elbow(s: np.ndarray, limit: int = MAX_RANK) -> int:
    """Locate the spectrum's elbow as the point furthest below its own chord.

    The singular values fall off smoothly, so "where the elbow is" needs a
    definition rather than an eyeball. Draw a straight line from the first
    plotted point to the last on a log scale; the elbow is the rank whose
    ``log10(sigma)`` sits furthest below that line.

    The answer depends on how far right you look -- widen ``limit`` and the
    elbow slides right -- so the post always states the window it used.

    Args:
        s: Singular values, descending.
        limit: Largest rank considered, and the right end of the chord.

    Returns:
        The rank at the elbow, 1-indexed.
    """
    logs = np.log10(s[:limit])
    ranks = np.arange(1, limit + 1)
    x = (ranks - ranks[0]) / (ranks[-1] - ranks[0])
    y = (logs - logs[-1]) / (logs[0] - logs[-1])
    return int(ranks[np.argmax((1.0 - x) - y)])


def marginal_rank(psnr_curve: np.ndarray, threshold: float = 0.2) -> int:
    """First rank whose next rank buys less than ``threshold`` dB.

    Unlike :func:`elbow` this needs no window, so the post leans on it for the
    claim about diminishing returns and uses the elbow only to mark the figure.

    Args:
        psnr_curve: PSNR at ranks ``1, 2, ..., len(psnr_curve)``.
        threshold: Decibels per extra rank below which returns are "diminishing".

    Returns:
        The 1-indexed rank at which the gain first falls below ``threshold``.
    """
    gains = np.diff(psnr_curve)
    below = np.flatnonzero(gains < threshold)
    return int(below[0] + 2) if below.size else int(psnr_curve.size)
