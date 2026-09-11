"""Rung 0: compare the pictures.

The bottom of the ladder, and the control the other three have to beat. No ridge
model, no landmarks, no training -- just how well one image lines up with
another once you allow it to slide.

Correlation is computed through the Fourier transform, which gives the score at
every possible shift for the price of one transform per image, and the score is
the best of them. Sliding is allowed because the prints, though centred on the
core, are not centred perfectly. Rotation is not allowed, because handling it
here would mean re-transforming the probe at every angle -- and being unable to
afford that is exactly why the later rungs describe a print in terms that do not
change when the finger turns.
"""

from __future__ import annotations

import enhance
import numpy as np
import ridges


def code(
    img: np.ndarray,
    enhanced: bool = False,
    analysis: ridges.Analysis | None = None,
) -> np.ndarray:
    """Prepare one print for correlation: zero-mean, unit-norm.

    Subtracting the mean is what stops a heavily inked print from correlating
    with everything, and dividing by the norm makes the score a cosine rather
    than a measure of how dark the two prints were.

    With `enhanced`, the Gabor-enhanced ridges are correlated instead of the grey
    image. That separates what the enhancement is worth from what a descriptor is
    worth, and it is the only part of this rung that needs ridge geometry -- pass
    the print's `analysis` if you already have one, because computing it again is
    the expensive half.
    """
    x = enhance.enhance(img, analysis) if enhanced else img.astype(np.float64)
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


def match_all(probe: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Best shift-aligned correlation of one probe against a stack of gallery codes.

    `gallery` is the stack of prepared gallery images; both must be the same
    shape. Returns one score per gallery entry.
    """
    probe_f = np.fft.rfft2(probe)
    gallery_f = np.fft.rfft2(gallery, axes=(-2, -1))
    surface = np.fft.irfft2(gallery_f * np.conj(probe_f), s=probe.shape, axes=(-2, -1))
    return surface.reshape(len(gallery), -1).max(axis=1)
