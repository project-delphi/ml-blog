"""Minutiae: the points where ridges stop or split, and how to compare two sets of them.

Rung 2. This is what a fingerprint examiner means by a fingerprint, and what
every automated system meant by one until the 2010s. Where rung 1 kept a
measurement of the whole pattern, this keeps a short list of landmarks and
throws the image away.

The extraction is the textbook pipeline:

1. **Enhance** with the Gabor bank, so ridges are clean enough to threshold.
2. **Binarise** into ridge and valley.
3. **Thin** each ridge to a one-pixel centreline.
4. **Count crossings.** Walk the eight neighbours of a skeleton pixel and count
   how many times the ring changes between skeleton and background. A pixel in
   the middle of a ridge changes twice. One change is a ridge ending; three is a
   bifurcation. Those are the minutiae.
5. **Clean up.** Thinning turns every speck and gap into a false minutia, so drop
   the ones on the edge of the print and the ones sitting on top of each other.

Matching two lists of points is the harder half. Two impressions of one finger
are related by an unknown rotation and shift, the skin stretches differently
each time, and neither list is complete. Rather than search over transforms,
each minutia gets a description of its own neighbourhood in its own frame of
reference -- how far away its neighbours are, in what direction relative to the
way it points, and which way they point. That description does not change when
the finger is rotated or moved, so the two prints can be compared by matching
descriptions. This is the idea behind Cappelli, Ferrara and Maltoni's Minutia
Cylinder-Code (2010), in a smaller form.

## This rung does not work, and the reason is upstream of the matching

On the committed cache it identifies 3% of probes against a 199-finger gallery.
Chance is 0.5%, so it is not quite guessing, and it is nowhere near the 33% that
plain correlation of the enhanced images gets on the same prints.

The cause is not the matcher. Registering each genuine pair by its ridge flow --
which gives the transform the matcher would have to find -- and then asking how
many landmarks actually land on each other, only about a fifth do, against a
tenth for two unrelated fingers. A descriptor needs the *neighbourhood* to be
stable, so at that rate almost every histogram is built from mostly different
members, and no way of comparing them can recover what was never there.

What was tried against that measurement, and did not move it:

* every scoring rule -- one-to-one greedy pairing, normalising by how many
  landmarks each print had, voting on the implied rotation, and centring the
  descriptors before comparing;
* a full rigid point-pattern search over rotation and translation, which
  identified 2.5% of probes -- if the points had correspondences, this would
  find them;
* three binarisations: the Gabor-selected one, the locally normalised image
  alone, and a blend;
* interpolating the orientation field to pixel resolution and blending between
  adjacent filters, on the theory that the block-stamped field was putting
  staircases in the skeleton for thinning to turn into false landmarks;
* merging clusters to their centre instead of keeping the strongest candidate,
  in case the winner-take-all suppression was picking different members of the
  same cluster in the two impressions;
* smoothing the binary image, thinning at twice the scale, and Lee's thinning
  rule instead of Zhang-Suen.

Two things did help, and neither is enough. Giving the minutia direction a sign
(`direct`, below) roughly tripled the separation, from 0.03 to 0.10 -- see its
docstring for why the unsigned field was fatal. And rebuilding the cache at the
source ridge period of 11.5 pixels rather than 6 raised the repeatability lift by
about a third; resolution is real, but a third of not enough is still not enough.

What would be needed is a different class of extractor -- ridge-following with a
quality map, in the manner of NIST's MINDTCT, rather than crossing numbers on a
thinned skeleton. That is a project, not a parameter. Until then, do not read this
rung as evidence about minutiae matching: it is evidence about this extractor.
Real AFIS systems do well on exactly this imagery, which is why NIST published it.
"""

from __future__ import annotations

from dataclasses import dataclass

import enhance
import numpy as np
import ridges
from scipy import ndimage
from skimage.morphology import skeletonize

# Neighbourhood descriptor: how far, which way round, which way pointing.
RADIUS_RIDGES = 8.0  # neighbourhood radius, in ridge periods
D_BINS, PHI_BINS, PSI_BINS = 4, 8, 8
_DIM = D_BINS * PHI_BINS * PSI_BINS

# The eight neighbours of a pixel, walked in a ring.
_RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


@dataclass(frozen=True)
class Minutiae:
    """A print's landmark list, plus the descriptors used to compare it."""

    xy: np.ndarray  # (M, 2) pixel coordinates
    theta: np.ndarray  # (M,) local ridge direction in radians
    kind: np.ndarray  # (M,) 1 = ridge ending, 3 = bifurcation
    descriptors: np.ndarray  # (M, D) rotation-invariant neighbourhood codes

    def __len__(self) -> int:
        return len(self.xy)


def binarise(img: np.ndarray, analysis: ridges.Analysis) -> np.ndarray:
    """Ridges as True, from the Gabor-enhanced image.

    The enhanced image is already zero-centred -- the filters have no DC term --
    so the threshold is zero rather than anything estimated. Outside the print
    the filters are answering about paper grain, and thresholding that produces a
    field of fake ridges, so anything outside the ridge mask is forced to valley
    before the threshold is taken.
    """
    binary = enhance.enhance(img, analysis) > 0
    binary &= analysis.pixel_mask()
    # A ridge is a long connected run. Specks this side of one are ink noise, and
    # each would otherwise thin down to a pair of false endings.
    labels, n = ndimage.label(binary)
    if n:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        binary = np.isin(labels, np.where(sizes >= 20)[0])
    return binary


def prune(skel: np.ndarray, length: int) -> np.ndarray:
    """Trim skeleton spurs shorter than `length` pixels.

    Thinning a ridge with a ragged edge leaves short whiskers hanging off it.
    Each whisker ends in a crossing number of 1 and would be counted as a ridge
    ending. Repeatedly deleting endpoints removes them; a real ridge is far
    longer than the few pixels this eats off it.
    """
    out = skel.copy()
    for _ in range(length):
        endpoints = out & (crossing_number(out) == 1)
        if not endpoints.any():
            break
        out &= ~endpoints
    return out


def crossing_number(skel: np.ndarray) -> np.ndarray:
    """Half the number of skeleton/background changes around each pixel's ring."""
    ring = np.stack([np.roll(np.roll(skel, -dy, 0), -dx, 1) for dy, dx in _RING])
    changes = np.abs(ring.astype(np.int8) - np.roll(ring, -1, axis=0).astype(np.int8))
    return changes.sum(0) // 2


def extract(
    img: np.ndarray,
    analysis: ridges.Analysis | None = None,
    max_points: int = 80,
) -> Minutiae:
    """Find the minutiae in one print and describe each one's neighbourhood."""
    analysis = analysis or ridges.analyse(img)
    period = analysis.period
    theta_field = analysis.theta

    skel = prune(skeletonize(binarise(img, analysis)), length=int(round(period / 2)))

    # Only trust minutiae well inside the print. Every ridge stops at the edge of
    # the ink, so the border is a ring of endings that say nothing about the
    # finger -- only about where the roll ran out.
    inside = ndimage.binary_erosion(analysis.mask, np.ones((3, 3)), iterations=2)
    inside_px = _fit(np.kron(inside, np.ones((ridges.BLOCK,) * 2, bool)), skel.shape)

    cn = crossing_number(skel)
    found = skel & inside_px & ((cn == 1) | (cn == 3))
    ys, xs = np.where(found)
    if len(ys) == 0:
        empty = np.zeros((0, 2))
        return Minutiae(empty, np.zeros(0), np.zeros(0, int), np.zeros((0, _DIM)))

    b = ridges.BLOCK
    by = np.clip(ys // b, 0, theta_field.shape[0] - 1)
    bx = np.clip(xs // b, 0, theta_field.shape[1] - 1)

    # Where the flow is clearest, the landmark is most likely real. Keeping the
    # best-supported ones bounds the cost of matching and drops the tail of
    # thinning artefacts that survived pruning.
    strength = analysis.coherence[by, bx] * analysis.ridgeness[by, bx]
    order = np.argsort(-strength)
    xy = np.column_stack([xs, ys]).astype(float)[order]
    kind = cn[ys, xs][order]

    xy, kind = _thin_out(xy, kind, min_gap=1.5 * period)
    xy, kind = xy[:max_points], kind[:max_points]

    theta = theta_field[
        np.clip((xy[:, 1] // b).astype(int), 0, theta_field.shape[0] - 1),
        np.clip((xy[:, 0] // b).astype(int), 0, theta_field.shape[1] - 1),
    ]
    theta = direct(skel, xy, theta, period)
    return Minutiae(xy, theta, kind, describe(xy, theta, period))


def direct(
    skel: np.ndarray,
    xy: np.ndarray,
    theta: np.ndarray,
    period: float,
) -> np.ndarray:
    """Give each minutia's ridge angle a sign, reading it off the skeleton.

    The orientation field is not a direction. A ridge running north-east is the
    same ridge running south-west, and `ridges.orientation` picks one of the two
    arbitrarily -- which is all a filter needs, and is fatal here. Every
    descriptor below is built in the minutia's own frame, so if the two
    impressions of a finger make opposite arbitrary choices, every neighbour
    lands in the wrong half of the histogram and the two prints look unrelated.

    The skeleton settles it. A ridge ending has ridge on one side of it and
    nothing on the other, so the centre of mass of the nearby skeleton lies along
    the ridge, pointing away from the ending. A bifurcation has three branches
    and the same centre of mass lands on their bisector -- not the textbook stem
    direction, but a repeatable one, which is all a frame of reference has to be.

    Returns angles in [0, 2*pi).
    """
    radius = max(3.0, 1.5 * period)
    ys, xs = np.nonzero(skel)
    if len(ys) == 0 or len(xy) == 0:
        return np.mod(theta, 2 * np.pi)

    points = np.column_stack([xs, ys]).astype(float)
    out = theta.astype(float).copy()
    for i, centre in enumerate(xy):
        delta = points - centre
        near = (delta**2).sum(1) < radius**2
        if near.sum() < 2:
            continue
        offset = delta[near].mean(axis=0)
        along = offset @ (np.cos(theta[i]), np.sin(theta[i]))
        if along < 0:
            out[i] += np.pi
    return np.mod(out, 2 * np.pi)


def _fit(a: np.ndarray, shape) -> np.ndarray:
    """Pad or trim `a` to `shape` -- block grids do not divide the image evenly."""
    out = np.zeros(shape, bool)
    h, w = min(a.shape[0], shape[0]), min(a.shape[1], shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def _thin_out(xy: np.ndarray, kind: np.ndarray, min_gap: float):
    """Drop minutiae that sit on top of each other.

    A broken ridge yields two endings a pixel apart, and a thinning artefact can
    yield a cluster. Neither is a real landmark, so keep the first of any group
    closer together than one ridge period.
    """
    keep: list[int] = []
    for i in range(len(xy)):
        if not keep or np.min(np.linalg.norm(xy[keep] - xy[i], axis=1)) >= min_gap:
            keep.append(i)
    return xy[keep], kind[keep]


def describe(xy: np.ndarray, theta: np.ndarray, period: float) -> np.ndarray:
    """One rotation-invariant neighbourhood histogram per minutia.

    For each other minutia inside the neighbourhood, measure three things in the
    centre minutia's own frame: how far away it is, the direction to it relative
    to the way the centre points, and the difference between the two directions.
    Bin those, and the result no longer knows which way up the finger was.

    Both angles run over a full turn, because `direct` has already resolved which
    way each minutia points. Folding either into a half turn would throw away the
    difference between a neighbour ahead and the same neighbour behind.
    """
    n = len(xy)
    dim = _DIM
    out = np.zeros((n, dim))
    if n < 2:
        return out

    radius = RADIUS_RIDGES * period
    delta = xy[None, :, :] - xy[:, None, :]  # (centre, neighbour, 2)
    dist = np.linalg.norm(delta, axis=2)
    bearing = np.arctan2(delta[:, :, 1], delta[:, :, 0])

    phi = np.mod(bearing - theta[:, None], 2 * np.pi)
    psi = np.mod(theta[None, :] - theta[:, None], 2 * np.pi)

    near = (dist > 0) & (dist < radius)
    d_bin = np.clip((dist / radius * D_BINS).astype(int), 0, D_BINS - 1)
    p_bin = np.clip((phi / (2 * np.pi) * PHI_BINS).astype(int), 0, PHI_BINS - 1)
    s_bin = np.clip((psi / (2 * np.pi) * PSI_BINS).astype(int), 0, PSI_BINS - 1)
    flat = (d_bin * PHI_BINS + p_bin) * PSI_BINS + s_bin

    for i in range(n):
        sel = near[i]
        if sel.any():
            out[i] = np.bincount(flat[i][sel], minlength=dim)

    # Blur the histogram across neighbouring bins, circularly in the two angles.
    # A neighbour a degree the wrong side of a bin edge is the same neighbour,
    # and hard bins would score it as a different one.
    grid = out.reshape(n, D_BINS, PHI_BINS, PSI_BINS)
    grid = ndimage.gaussian_filter1d(grid, 0.7, axis=1, mode="nearest")
    grid = ndimage.gaussian_filter1d(grid, 0.7, axis=2, mode="wrap")
    grid = ndimage.gaussian_filter1d(grid, 0.7, axis=3, mode="wrap")
    out = grid.reshape(n, dim)

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


def match(probe: Minutiae, gallery: Minutiae, top_k: int = 12) -> float:
    """How alike two minutiae sets are, in [0, 1].

    Every probe minutia is compared with every gallery minutia by descriptor.
    The score is the mean of the strongest few pairings -- the local similarity
    sort of the Cylinder-Code paper. Taking only the best few is what makes the
    comparison survive a partial print: an impression that caught half the finger
    can still put a dozen landmarks in the same relative arrangement.
    """
    if len(probe) < 2 or len(gallery) < 2:
        return 0.0
    similarity = probe.descriptors @ gallery.descriptors.T
    k = min(top_k, similarity.size)
    return float(np.mean(np.sort(similarity, axis=None)[-k:]))


def match_all(probe: Minutiae, gallery: list[Minutiae], top_k: int = 12) -> np.ndarray:
    """One probe against a whole enrolled gallery."""
    return np.array([match(probe, g, top_k) for g in gallery])
