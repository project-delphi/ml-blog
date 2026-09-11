"""Gabor enhancement and the FingerCode descriptor: rung 1.

The classical pipeline's insight is that a fingerprint is locally a sinusoid: at
any small patch the ridges run in one direction and repeat at one period. A Gabor
filter is a sinusoid inside a Gaussian envelope, so a bank of them tuned to the
ridge period and stepped through orientation answers "how much ridge flow is
there, in each direction, here?". Applying the filter that matches the local
orientation is what turns a grey smear into legible ridges (Hong, Wan and Jain,
1998).

Jain, Prabhakar, Hong and Pankanti (2000) turned that bank into a fixed-length
descriptor -- the FingerCode. Lay a polar tessellation over the print, measure how
much energy each Gabor orientation puts into each cell, and concatenate. Two
prints of one finger give similar vectors, and turning the finger moves energy
around the tessellation in a predictable, cyclic way. That is what lets the
comparison survive rotation without searching over rotated images: rotating the
*descriptor* is a pair of array rolls.
"""

from __future__ import annotations

import numpy as np
import ridges
from ridges import gabor_kernel  # noqa: F401  (re-exported: the bank is built from it)
from scipy import ndimage

ORIENTATIONS = 8
BANDS = 4
SECTORS = 16


def normalise(img: np.ndarray, period: float = ridges.RIDGE_PERIOD) -> np.ndarray:
    """Zero-mean, unit-variance, with the slow ink-coverage gradient removed.

    Inked cards vary in how hard the finger was pressed, both between prints and
    across one print. Subtracting a local mean and dividing by a local spread puts
    every patch on the same footing before any filter sees it. The window scales
    with the ridge period so it always spans a couple of ridges.
    """
    win = max(5, int(round(2 * period)))
    x = img.astype(np.float64) / 255.0
    x = x - ndimage.uniform_filter(x, win)
    spread = np.sqrt(np.clip(ndimage.uniform_filter(x**2, win), 1e-6, None))
    return x / spread


def _fit(a: np.ndarray, shape) -> np.ndarray:
    """Stretch a block-grid array to pixel `shape` -- blocks do not divide evenly."""
    out = np.zeros(shape, a.dtype)
    h, w = min(a.shape[0], shape[0]), min(a.shape[1], shape[1])
    out[:h, :w] = a[:h, :w]
    if h < shape[0]:
        out[h:, :w] = a[h - 1, :w]
    if w < shape[1]:
        out[:, w:] = out[:, w - 1 : w]
    return out


def gabor_bank(img: np.ndarray, analysis: ridges.Analysis | None = None) -> np.ndarray:
    """Filter `img` with one Gabor per orientation; returns (orientations, H, W).

    Both the period and the envelope come from the print's own ridge spacing, so
    one bank is right for prints scanned at different sizes.
    """
    analysis = analysis or ridges.analyse(img)
    x = normalise(img, analysis.period)
    return np.stack(
        [
            ridges.filter_image(x, gabor_kernel(t, analysis.period))
            for t in np.arange(ORIENTATIONS) * np.pi / ORIENTATIONS
        ],
    )


def enhance(img: np.ndarray, analysis: ridges.Analysis | None = None) -> np.ndarray:
    """Contrast-enhanced ridges: at each pixel, take the best-aligned filter.

    The classical pipeline estimates the local orientation and applies the one
    matching filter. Selecting per pixel from the whole bank reaches the same
    place -- the aligned filter is the one that responds -- using an orientation
    field that has already been computed.
    """
    analysis = analysis or ridges.analyse(img)
    bank = gabor_bank(img, analysis)
    theta = _fit(
        np.kron(analysis.theta, np.ones((ridges.BLOCK, ridges.BLOCK))),
        img.shape,
    )
    idx = np.round(theta / (np.pi / ORIENTATIONS)).astype(int) % ORIENTATIONS
    return np.take_along_axis(bank, idx[None], axis=0)[0]


def tessellation(shape, bands: int = BANDS, sectors: int = SECTORS) -> np.ndarray:
    """Polar cell index for every pixel, or -1 outside the disc.

    Cells are `bands` concentric rings cut into `sectors` wedges, centred on the
    image -- which is the middle of the inked area, because that is where
    `fetch_data.standardise` centred the prints. The innermost disc is left out:
    near the centre of a loop or whorl the ridge direction turns too fast for any
    one filter to mean much.
    """
    h, w = shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot(y - cy, x - cx)
    a = np.arctan2(y - cy, x - cx)

    r_inner, r_outer = 0.12 * min(h, w), 0.5 * min(h, w)
    band = np.floor((r - r_inner) / (r_outer - r_inner) * bands).astype(int)
    sector = np.floor((a + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors

    cell = band * sectors + sector
    cell[(r < r_inner) | (r >= r_outer)] = -1
    return cell


def fingercode(
    img: np.ndarray,
    analysis: ridges.Analysis | None = None,
    bands: int = BANDS,
    sectors: int = SECTORS,
) -> np.ndarray:
    """Fixed-length descriptor: Gabor energy per (cell, orientation).

    Returns an (orientations, bands, sectors) array. The average absolute
    deviation of a filter's response inside a cell is the "how much ridge flow
    runs this way, here" number; stacking them is the whole descriptor.
    """
    bank = gabor_bank(img, analysis)
    cell = tessellation(img.shape, bands, sectors)
    valid = cell >= 0
    flat_cell = cell[valid]
    n_cells = bands * sectors

    counts = np.bincount(flat_cell, minlength=n_cells).astype(float)
    counts[counts == 0] = 1.0

    out = np.empty((len(bank), n_cells))
    for o, response in enumerate(bank):
        v = response[valid]
        mean = np.bincount(flat_cell, weights=v, minlength=n_cells) / counts
        dev = np.abs(v - mean[flat_cell])
        out[o] = np.bincount(flat_cell, weights=dev, minlength=n_cells) / counts

    out = out.reshape(len(bank), bands, sectors)
    return out / (np.linalg.norm(out) + 1e-12)


def rotations(
    code: np.ndarray,
    orientations: int = ORIENTATIONS,
    sectors: int = SECTORS,
):
    """Every whole-sector rotation of a FingerCode.

    Turning the finger by one sector's worth of angle rolls the tessellation by
    one sector and the ridge directions by the matching number of orientation
    bins. Generating those rolls costs nothing next to re-filtering a rotated
    image, which is the point of the descriptor.

    How many bins is "matching" is the easy thing to get wrong here. Sectors
    divide a full turn and orientations divide a half turn, so s sectors of
    rotation is `s * 2 * orientations / sectors` orientation bins -- exactly s
    for the 16/8 default, where both cells span the same pi/8. Rolling by
    `s // (sectors // orientations)` instead moves the histogram half as far as
    the tessellation, which leaves every candidate but s = 0 comparing a rotated
    tessellation against an unrotated set of ridge directions: a print turned 90
    degrees then scores 0.78 against itself where it should score 1.00, and the
    mismatched candidates inflate impostor scores too.

    The ratio is worked out per step and rounded rather than folded into one
    per-sector constant. Integer-dividing it up front truncates to zero for any
    `sectors` above twice `orientations`, which silently restores the bug above,
    and throws away the fraction when `sectors` does not divide
    `2 * orientations`. Rounding each step keeps the error under half a bin
    instead of letting it accumulate; the rotations are exact only when `sectors`
    does divide `2 * orientations`.
    """
    for s in range(sectors):
        bins = int(round(s * 2 * orientations / sectors)) % orientations
        yield np.roll(np.roll(code, s, axis=2), bins, axis=0)


def match(probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Similarity of one probe code against a stack of gallery codes.

    Score is the best dot product over whole-sector rotations of the probe.

    The tessellation's shape is read off the descriptor rather than taken from
    the module constants, so a code built with a non-default `bands`/`sectors`
    is rolled by its own geometry instead of by 16/8's.
    """
    g = gallery.reshape(len(gallery), -1)
    best = np.full(len(gallery), -np.inf)
    for rot in rotations(probe, probe.shape[0], probe.shape[2]):
        np.maximum(best, g @ rot.ravel(), out=best)
    return best
