"""Ridge geometry: what every rung needs before it can do anything.

Four things are read off a print here, and all four come out of one pass:

* the **orientation field** -- the local ridge angle on a coarse block grid;
* the **coherence** -- how single-minded the gradients in a block are, which is
  near 1 on clean parallel ridges and near 0 on blank paper;
* the **ridge mask** -- which blocks hold friction ridges rather than paper, card
  text or a punch hole;
* the **ridge period** -- how many pixels apart the ridges run, which is what
  every filter downstream has to be tuned to.

They are computed together, in `analyse`, because each of them needs the others:
the mask needs the period to tell ridges from print, the period needs the
orientation to know which way to measure, and the core needs all three. Computing
them separately, once per caller, was the first version of this file and it was
three times slower for the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

BLOCK = 8
RIDGE_PERIOD = 9.0  # pixels at 500 dpi, roughly

PERIODS = np.arange(4.0, 17.0, 1.5)
N_ANGLES = 8
ANGLES = np.arange(N_ANGLES) * np.pi / N_ANGLES


def blocks(a: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Block means of `a`, sampled at block centres."""
    return ndimage.uniform_filter(a, block)[block // 2 :: block, block // 2 :: block]


def gabor_kernel(theta: float, period: float, sigma: float | None = None) -> np.ndarray:
    """Real 2D Gabor kernel: a sinusoid of `period` under a Gaussian envelope.

    `theta` is the direction the ridges run. The sinusoid varies *across* the
    ridges, so the wave vector is perpendicular to `theta`. The envelope scales
    with the period, so the filter always spans about the same number of ridges.
    """
    sigma = 0.45 * period if sigma is None else sigma
    half = int(round(3 * sigma))
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-(xr**2 + yr**2) / (2 * sigma**2))
    k = envelope * np.cos(2 * np.pi * yr / period)
    k = k - k.mean()  # no DC, so flat ink coverage scores zero
    # Unit norm, so responses at different periods are on one scale. Without it
    # a filter tuned to a short period -- a physically smaller kernel, with a
    # smaller norm -- cannot be compared with a long one, and the search for the
    # ridge period below just returns the shortest one every time.
    return k / (np.linalg.norm(k) + 1e-12)


def filter_image(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve through the Fourier domain.

    A Gabor tuned to a 16-pixel period needs a kernel about 45 across. Sliding
    that over the image directly costs two thousand multiplies per pixel, and the
    period search below wants seventy of them; through the FFT the whole bank
    takes about as long as one direct convolution.
    """
    return fftconvolve(x, kernel, mode="same")


def orientation(img: np.ndarray, block: int = BLOCK):
    """Ridge orientation field and coherence.

    `theta` is the ridge direction in radians in [0, pi). Direction has no sign --
    a ridge running north-east is the same ridge running south-west -- so the
    gradients are averaged as doubled angles, where those two agree instead of
    cancelling.
    """
    x = img.astype(np.float64) / 255.0
    gy, gx = np.gradient(ndimage.gaussian_filter(x, 1.0))
    gxx, gyy, gxy = (
        blocks(gx * gx, block),
        blocks(gy * gy, block),
        blocks(gx * gy, block),
    )
    num, den = 2 * gxy, gxx - gyy
    theta = 0.5 * np.arctan2(num, den) + np.pi / 2  # gradient normal -> ridge tangent
    coherence = np.hypot(num, den) / (gxx + gyy + 1e-12)
    return np.mod(theta, np.pi), np.clip(coherence, 0, 1)


@dataclass(frozen=True)
class Analysis:
    """One print's ridge geometry, computed once and passed around."""

    theta: np.ndarray  # (h, w) ridge direction per block, radians in [0, pi)
    coherence: np.ndarray  # (h, w) how strongly oriented that block is, 0..1
    mask: np.ndarray  # (h, w) blocks holding friction ridges
    ridgeness: np.ndarray  # (h, w) how ridge-like, masked
    period_map: np.ndarray  # (h, w) local ridge period in pixels
    period: float  # the print's ridge period overall
    shape: tuple  # the image's pixel shape

    def pixel_mask(self) -> np.ndarray:
        """The block mask expanded back to pixels."""
        big = np.kron(self.mask, np.ones((BLOCK, BLOCK), bool))
        out = np.zeros(self.shape, bool)
        h, w = min(big.shape[0], self.shape[0]), min(big.shape[1], self.shape[1])
        out[:h, :w] = big[:h, :w]
        return out


def analyse(img: np.ndarray, block: int = BLOCK) -> Analysis:
    """Orientation, coherence, mask and period for one print, in a single pass.

    The mask is the interesting part. A ridge block is strongly oriented, carries
    energy at the ridge frequency, *and* repeats at a plausible ridge period.
    Printed text on a fingerprint card clears the first two tests and fails the
    third -- letters are not periodic -- which is why the period is measured here
    rather than assumed. What survives is then cut down to its largest connected
    piece: the print is one blob, and the words along the top of the card are
    separate strips.
    """
    theta, coherence = orientation(img, block)

    x = img.astype(np.float64) / 255.0
    x = x - ndimage.uniform_filter(x, 4 * block)

    energy = np.empty((len(PERIODS), N_ANGLES) + theta.shape)
    for pi, p in enumerate(PERIODS):
        for ai, a in enumerate(ANGLES):
            energy[pi, ai] = blocks(np.abs(filter_image(x, gabor_kernel(a, p))), block)

    bin_of = np.round(theta / (np.pi / N_ANGLES)).astype(int) % N_ANGLES
    rows, cols = np.indices(theta.shape)
    at_own_angle = energy[:, bin_of, rows, cols]  # (periods, h, w)
    best = np.argmax(at_own_angle, axis=0)
    period_map = PERIODS[best]
    strength = at_own_angle[best, rows, cols]

    ridgeness = ndimage.gaussian_filter(
        coherence * (strength / (np.percentile(strength, 95) + 1e-12)),
        1.5,
    )
    # Scale by a high percentile rather than the maximum: one hot block -- the
    # rim of a punch hole, the edge of a printed rule -- would otherwise set the
    # scale and push the whole print below the threshold.
    periodic = (period_map > PERIODS[0]) & (period_map < PERIODS[-1])
    mask = (ridgeness > 0.3 * np.percentile(ridgeness, 95)) & periodic
    mask = ndimage.binary_closing(mask, np.ones((3, 3)))
    mask = ndimage.binary_opening(mask, np.ones((3, 3)))

    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        mask = labels == (int(np.argmax(sizes)) + 1)

    good = mask & periodic
    period = float(np.median(period_map[good])) if good.any() else RIDGE_PERIOD
    return Analysis(
        theta, coherence, mask, ridgeness * mask, period_map, period, img.shape
    )


# Closed counter-clockwise walk around a block's 8 neighbours.
_LOOP = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def poincare(theta: np.ndarray) -> np.ndarray:
    """Poincare index of the orientation field, in half-turns.

    Walk a small closed loop around each block and add up the ridge-angle
    changes, wrapping each step into (-pi/2, pi/2] because ridge direction has no
    sign. The total comes to +1/2 at a core -- where ridges curve around a
    turning point -- and -1/2 at a delta, where three ridge flows meet. Anywhere
    the flow is smooth it comes to 0.
    """
    total = np.zeros_like(theta)
    for (dy0, dx0), (dy1, dx1) in zip(_LOOP, _LOOP[1:] + _LOOP[:1]):
        a = np.roll(np.roll(theta, -dy0, 0), -dx0, 1)
        b = np.roll(np.roll(theta, -dy1, 0), -dx1, 1)
        d = b - a
        d = np.mod(d + np.pi / 2, np.pi) - np.pi / 2  # wrap to (-pi/2, pi/2]
        total += d
    total[[0, -1], :] = 0
    total[:, [0, -1]] = 0
    return total / np.pi


def core(analysis: Analysis, block: int = BLOCK) -> tuple[int, int]:
    """Pixel coordinates of the print's registration point.

    Prefer the ridge core -- the singular point a loop or whorl turns around, and
    the landmark classical matchers have registered on since Galton. Arches have
    no core and a smudged scan can hide one, so fall back to the centre of mass
    of the ridge-ness map.
    """
    idx = poincare(ndimage.gaussian_filter(analysis.theta, 0.8, mode="nearest"))
    # Several blocks can trip together on one singularity, and a whorl has two
    # cores; the ridge-weighted mean of them sits inside the pattern.
    w = analysis.ridgeness * ((idx > 0.3) & analysis.mask)
    if w.sum() <= 0:
        w = analysis.ridgeness  # arch, or a scan too smudged to show a core
    if w.sum() <= 0:
        return analysis.shape[0] // 2, analysis.shape[1] // 2

    rows, cols = np.indices(analysis.theta.shape)
    cy = (rows * w).sum() / w.sum()
    cx = (cols * w).sum() / w.sum()
    half = block // 2
    return int(round(cy * block + half)), int(round(cx * block + half))
