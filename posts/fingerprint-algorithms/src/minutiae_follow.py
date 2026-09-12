"""Minutiae again, from ridge crests instead of a thinned binary image.

``minutiae.py`` is the textbook pipeline and it does not work on this cache: 3%
of probes identified against a 199-finger gallery, against 33% for plain
correlation of the enhanced images. Its docstring traces the cause upstream of
the matching and names the fix -- "a different class of extractor --
ridge-following with a quality map, in the manner of NIST's MINDTCT". This is
that attempt.

What changes, and why each change targets the measured failure:

* **Follow the ridge, do not thin the ink.** Thresholding the enhanced image and
  skeletonizing it turns every ragged edge into a staircase and every staircase
  corner into a candidate landmark. Here the ridge is found as a *crest* -- the
  line where the enhanced image is a local maximum along the direction across
  the ridges -- which is a statement about the ridge flow rather than about where
  a threshold happened to fall.

* **Smooth along the flow first.** An elongated Gaussian, oriented per pixel by
  the orientation field, closes the small ink gaps that a global threshold turns
  into pairs of false endings, without blurring one ridge into its neighbour.

* **Bridge the gaps that survive.** Every crest endpoint looks ahead along its
  own tangent for another endpoint coming the other way. A pair that agrees
  within a narrow cone is one ridge that the ink broke, so it is joined. This is
  the single biggest source of unrepeatable endings: a gap that closes in one
  impression and not the other invents two landmarks in one print and none in
  the other.

* **Carry a quality map.** A landmark sitting where the flow is incoherent, where
  the ridge signal is weak, or within a couple of blocks of the edge of the print
  is not evidence about the finger. Quality orders the candidates and cuts the
  tail, so the list that survives is the part of the print the image supports.

* **Keep the signed direction.** ``minutiae.direct`` is reused unchanged. Signing
  the minutia angle is what took the old extractor's separation from 0.03 to
  0.10, and it would be easy to lose by rewriting the frame of reference.

The descriptors and the matching are ``minutiae.describe`` and
``minutiae.match_all``, untouched, and ``extract`` returns a
``minutiae.Minutiae``. That is deliberate: the question this module exists to
answer is whether the *extractor* was the problem, and changing the comparison at
the same time would make the answer unreadable.

Run the gate before trusting any of it. ``--repeatability`` measures the thing
that diagnosed the old extractor: register each genuine pair by its ridge flow,
so the matcher is handed the transform it would otherwise have to find, then ask
what fraction of landmarks actually land on each other, genuine against
impostor. The old extractor scores about a fifth against a tenth -- not enough
for any descriptor to work with.

    PYTHONPATH=src ../../.venv-fingerprint-algorithms/bin/python src/minutiae_follow.py --repeatability
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import enhance
import minutiae
import numpy as np
import ridges
from scipy import ndimage
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize

N_ORIENT = 8

# Anisotropic smoothing, in ridge periods. Long along the flow to close ink
# gaps, short across it so two ridges never merge into one.
SIGMA_ALONG = 0.75
SIGMA_ACROSS = 0.28

GAP_MAX_PERIODS = 1.25  # how far a bridge may reach, in ridge periods
GAP_CONE_DEG = 35.0  # how far off its own tangent an endpoint may look
MIN_QUALITY = 0.15  # candidates below this are dropped outright
EDGE_BLOCKS = 2.0  # keep landmarks this many blocks inside the print


def _expand(block_array: np.ndarray, shape, block: int = ridges.BLOCK):
    """Blow a block-grid array up to pixel `shape`; blocks do not divide evenly."""
    big = np.kron(block_array, np.ones((block, block), block_array.dtype))
    out = np.zeros(shape, block_array.dtype)
    h, w = min(big.shape[0], shape[0]), min(big.shape[1], shape[1])
    out[:h, :w] = big[:h, :w]
    if h < shape[0]:
        out[h:, :w] = out[h - 1, :w]
    if w < shape[1]:
        out[:, w:] = out[:, w - 1 : w]
    return out


def _ridge_kernel(theta: float, period: float) -> np.ndarray:
    """Gaussian stretched along `theta` and squeezed across it, unit sum."""
    sigma_a = max(1.0, SIGMA_ALONG * period)
    sigma_c = max(0.6, SIGMA_ACROSS * period)
    half = int(round(3 * sigma_a))
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    along = x * np.cos(theta) + y * np.sin(theta)
    across = -x * np.sin(theta) + y * np.cos(theta)
    k = np.exp(-(along**2 / (2 * sigma_a**2) + across**2 / (2 * sigma_c**2)))
    return k / k.sum()


def smooth_along_flow(
    enhanced: np.ndarray,
    analysis: ridges.Analysis,
) -> np.ndarray:
    """Smooth the enhanced image along the local ridge direction.

    One filtered copy per orientation, then each pixel takes the copy whose angle
    matches its own. The same select-from-a-bank trick ``enhance.enhance`` uses,
    for the same reason: the orientation field is already computed, so this costs
    eight convolutions rather than a per-pixel steering.
    """
    bank = np.stack(
        [
            ridges.filter_image(enhanced, _ridge_kernel(t, analysis.period))
            for t in np.arange(N_ORIENT) * np.pi / N_ORIENT
        ],
    )
    theta = _expand(analysis.theta, enhanced.shape)
    idx = np.round(theta / (np.pi / N_ORIENT)).astype(int) % N_ORIENT
    return np.take_along_axis(bank, idx[None], axis=0)[0]


def crests(img_smooth: np.ndarray, analysis: ridges.Analysis) -> np.ndarray:
    """Find the ridge centrelines: local maxima taken across the ridge direction.

    A ridge is a hump in the enhanced image. Walking across it, the centre is
    higher than either flank, so sampling one step to each side along the normal
    and keeping the pixels that beat both finds the crest line. Because the step
    is taken along the *normal*, this is a question about the ridge flow, not
    about where a threshold sat -- which is the whole difference from
    ``minutiae.binarise``.
    """
    theta = _expand(analysis.theta, img_smooth.shape)
    step = max(1.0, 0.3 * analysis.period)
    # Ridges run along (cos t, sin t) in (x, y), so the normal is (-sin t, cos t).
    dy, dx = step * np.cos(theta), -step * np.sin(theta)

    rows, cols = np.indices(img_smooth.shape, dtype=float)
    ahead = ndimage.map_coordinates(
        img_smooth,
        [rows + dy, cols + dx],
        order=1,
        mode="nearest",
    )
    behind = ndimage.map_coordinates(
        img_smooth,
        [rows - dy, cols - dx],
        order=1,
        mode="nearest",
    )
    return (img_smooth >= ahead) & (img_smooth >= behind) & (img_smooth > 0)


def _tangents(skel: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray:
    """Outward unit vector at each endpoint: away from the skeleton behind it.

    An endpoint has ridge on one side and nothing on the other, so the mean
    offset of the nearby skeleton points *backwards* along the ridge. Negating it
    gives the direction the ridge was heading when the ink stopped, which is
    where a bridge has to look.
    """
    ys, xs = np.nonzero(skel)
    cloud = np.column_stack([xs, ys]).astype(float)
    out = np.zeros((len(points), 2))
    for i, centre in enumerate(points):
        delta = cloud - centre
        near = (delta**2).sum(1) < radius**2
        if near.sum() < 2:
            continue
        back = delta[near].mean(axis=0)
        norm = np.hypot(*back)
        if norm > 1e-9:
            out[i] = -back / norm
    return out


def bridge_gaps(skel: np.ndarray, period: float) -> np.ndarray:
    """Join crest endpoints that face each other across a small gap.

    A broken ridge presents two endpoints a short way apart, each pointing at the
    other. A genuine ridge ending has nothing facing it. Requiring both the
    separation and the two tangents to agree is what keeps this from stitching
    unrelated ridges together, and every join removes two false landmarks that
    the other impression of the same finger would not have produced.
    """
    out = skel.copy()
    cn = minutiae.crossing_number(out)
    ys, xs = np.nonzero(out & (cn == 1))
    if len(ys) < 2:
        return out

    ends = np.column_stack([xs, ys]).astype(float)
    tangent = _tangents(out, ends, radius=max(3.0, 1.2 * period))
    gap_max = GAP_MAX_PERIODS * period
    cone = np.cos(np.deg2rad(GAP_CONE_DEG))

    delta = ends[None, :, :] - ends[:, None, :]
    dist = np.hypot(delta[:, :, 0], delta[:, :, 1])
    joined = np.zeros(len(ends), bool)

    order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
    for i, j in order:
        if i >= j or joined[i] or joined[j]:
            continue
        d = dist[i, j]
        if d < 1e-6 or d > gap_max:
            continue
        towards = delta[i, j] / d
        # Each endpoint must be heading at the other, not merely near it.
        if towards @ tangent[i] < cone or (-towards) @ tangent[j] < cone:
            continue
        rr, cc = draw_line(
            int(ends[i, 1]),
            int(ends[i, 0]),
            int(ends[j, 1]),
            int(ends[j, 0]),
        )
        out[rr, cc] = True
        joined[i] = joined[j] = True
    return out


def quality_map(analysis: ridges.Analysis) -> np.ndarray:
    """Per-block confidence that a landmark found here says something about the finger.

    Three things have to hold: the flow is coherent, the ridge signal is strong,
    and the block is not on the rim of the print. The rim matters most -- every
    ridge stops where the roll ran out, so the border of any print is a ring of
    endings that describe the ink and not the skin.
    """
    ridgeness = analysis.ridgeness
    scale = np.percentile(ridgeness[analysis.mask], 90) if analysis.mask.any() else 0.0
    strength = np.clip(ridgeness / (scale + 1e-12), 0, 1)

    # Pad before the distance transform, so the edge of the image counts as
    # outside the print. Without the pad a print that runs to the frame gets
    # credit for a rim that is only an artefact of where the scan was cut, and
    # the whole border ring of endings survives the quality cut.
    padded = np.pad(analysis.mask, 1, constant_values=False)
    inside = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    # A hard margin, not a ramp: nothing within EDGE_BLOCKS of the rim counts at
    # all. Every ridge stops where the roll ran out, so that ring describes the
    # ink rather than the finger, and it is the largest single source of
    # landmarks one impression has and the other does not.
    room = np.clip((inside - EDGE_BLOCKS) / 2.0, 0, 1)
    return np.clip(analysis.coherence, 0, 1) * strength * room


def extract(
    img: np.ndarray,
    analysis: ridges.Analysis | None = None,
    max_points: int = 80,
    return_extras: bool = False,
):
    """Find the minutiae in one print by following ridge crests.

    Drop-in replacement for ``minutiae.extract``: same arguments, same
    ``minutiae.Minutiae`` back, so ``ladder.landmarks`` does not know which one
    it is calling. With `return_extras`, also returns the intermediate images the
    figures need.
    """
    analysis = analysis or ridges.analyse(img)
    period = analysis.period

    enhanced = enhance.enhance(img, analysis)
    smooth = smooth_along_flow(enhanced, analysis)
    crest = crests(smooth, analysis) & analysis.pixel_mask()

    # Thinning a crest map is not thinning ink: the map is already one ridge
    # wide, so this only enforces 8-connectivity rather than inventing a
    # centreline. The spur prune then clears the stubs that enforcement leaves.
    skel = minutiae.prune(skeletonize(crest), length=max(2, int(round(period / 3))))
    skel = bridge_gaps(skel, period)
    skel = minutiae.prune(skel, length=max(2, int(round(period / 3))))

    cn = minutiae.crossing_number(skel)
    found = skel & ((cn == 1) | (cn == 3))
    ys, xs = np.where(found)
    if len(ys) == 0:
        empty = minutiae.Minutiae(
            np.zeros((0, 2)),
            np.zeros(0),
            np.zeros(0, int),
            np.zeros((0, minutiae._DIM)),
        )
        return (
            (empty, {"enhanced": enhanced, "skeleton": skel})
            if return_extras
            else empty
        )

    b = ridges.BLOCK
    quality = quality_map(analysis)
    by = np.clip(ys // b, 0, quality.shape[0] - 1)
    bx = np.clip(xs // b, 0, quality.shape[1] - 1)
    q = quality[by, bx]

    # No fallback when nothing clears the bar. An earlier version kept the best
    # half in that case, which resurrected precisely the landmarks the quality
    # map had just rejected -- on a print whose ridges run off every edge, that
    # returned a ring of border endings and called them minutiae. An empty list
    # is the honest answer, and `minutiae.match` already scores it as no match.
    keep = q >= MIN_QUALITY
    ys, xs, q = ys[keep], xs[keep], q[keep]
    if len(ys) == 0:
        empty = minutiae.Minutiae(
            np.zeros((0, 2)),
            np.zeros(0),
            np.zeros(0, int),
            np.zeros((0, minutiae._DIM)),
        )
        return (
            (empty, {"enhanced": enhanced, "skeleton": skel})
            if return_extras
            else empty
        )

    order = np.argsort(-q)
    xy = np.column_stack([xs, ys]).astype(float)[order]
    kind = cn[ys, xs][order]

    xy, kind = minutiae._thin_out(xy, kind, min_gap=1.5 * period)
    xy, kind = xy[:max_points], kind[:max_points]

    theta_field = analysis.theta
    theta = theta_field[
        np.clip((xy[:, 1] // b).astype(int), 0, theta_field.shape[0] - 1),
        np.clip((xy[:, 0] // b).astype(int), 0, theta_field.shape[1] - 1),
    ]
    theta = minutiae.direct(skel, xy, theta, period)

    found_minutiae = minutiae.Minutiae(
        xy,
        theta,
        kind,
        minutiae.describe(xy, theta, period),
    )
    if return_extras:
        return found_minutiae, {
            "enhanced": enhanced,
            "smooth": smooth,
            "crest": crest,
            "skeleton": skel,
            "quality": quality,
        }
    return found_minutiae


def match_all(probe, gallery, top_k: int = 12) -> np.ndarray:
    """Unchanged from ``minutiae``; re-exported so a caller swaps one module name."""
    return minutiae.match_all(probe, gallery, top_k)


# ----------------------------------------------------------------------------
# The gate: are the landmarks repeatable at all?
# ----------------------------------------------------------------------------


def _unit(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


def _rotation(angle: float) -> np.ndarray:
    """Forward rotation acting on (row, col) coordinates."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])


def rotate_about_centre(img: np.ndarray, angle: float) -> np.ndarray:
    """Turn an image about its centre, so that a feature at `p` lands at `R(p - c) + c`.

    Written as an explicit `affine_transform` rather than `ndimage.rotate`
    because the point map has to agree with the image map exactly. Getting that
    agreement wrong is not a small error: it silently rotates the landmarks the
    wrong way, which makes a genuine pair land no better than an impostor pair
    and reads exactly like an extractor that does not work.
    """
    centre = (np.array(img.shape, float) - 1) / 2
    inverse = _rotation(angle).T
    return ndimage.affine_transform(
        img,
        inverse,
        offset=centre - inverse @ centre,
        order=1,
        mode="nearest",
    )


def best_alignment(
    probe_enhanced: np.ndarray,
    gallery_enhanced: np.ndarray,
    angles: np.ndarray,
):
    """Rotation and shift that line two enhanced prints up, by correlation.

    This hands the matcher the transform it would otherwise have to search for.
    That is the point: if the landmarks still do not coincide once the prints are
    registered, no matching rule can recover a correspondence that is not there.

    Returns `(angle, peak)` where `peak` is the correlation shift in (row, col).
    """
    gallery_f = np.conj(np.fft.rfft2(_unit(gallery_enhanced)))
    best = (-np.inf, 0.0, (0, 0))
    h, w = probe_enhanced.shape
    for angle in angles:
        turned = rotate_about_centre(probe_enhanced, angle)
        surface = np.fft.irfft2(
            np.fft.rfft2(_unit(turned)) * gallery_f,
            s=probe_enhanced.shape,
        )
        dy, dx = np.unravel_index(int(np.argmax(surface)), surface.shape)
        score = float(surface[dy, dx])
        if score > best[0]:
            # A cyclic correlation peak past the halfway point is a negative shift.
            best = (
                score,
                angle,
                (
                    int(dy) - h if dy > h // 2 else int(dy),
                    int(dx) - w if dx > w // 2 else int(dx),
                ),
            )
    return best[1], best[2], best[0]


def transform_points(
    xy: np.ndarray,
    angle: float,
    peak: tuple[int, int],
    shape,
) -> np.ndarray:
    """Put probe landmark coordinates into the gallery's frame.

    Two steps, in the convention `rotate_about_centre` and `best_alignment` fix
    between them. The rotation sends a probe point `p` to `R(p - c) + c`. The
    correlation peak `k` satisfies `turned[n + k] ~ gallery[n]`, so a point at
    `m` in the turned frame is at `m - k` in the gallery's.
    """
    if len(xy) == 0:
        return xy
    centre = (np.array(shape, float) - 1) / 2
    rowcol = np.column_stack([xy[:, 1], xy[:, 0]])  # minutiae are stored (x, y)
    turned = (_rotation(angle) @ (rowcol - centre).T).T + centre
    moved = turned - np.array(peak, float)
    return np.column_stack([moved[:, 1], moved[:, 0]])


def landing_rate(
    probe_xy: np.ndarray,
    gallery_xy: np.ndarray,
    tolerance: float,
) -> float:
    """Fraction of probe landmarks with a gallery landmark inside `tolerance`."""
    if len(probe_xy) == 0 or len(gallery_xy) == 0:
        return 0.0
    d = np.linalg.norm(probe_xy[:, None, :] - gallery_xy[None, :, :], axis=2)
    return float((d.min(axis=1) <= tolerance).mean())


def repeatability(
    prints,
    extractor,
    pairs: int = 60,
    tolerance_periods: float = 1.0,
    seed: int = 7,
) -> dict:
    """Landing rates for genuine and impostor pairs, after registration."""
    rng = np.random.default_rng(seed)
    fingers = np.unique(prints.finger)
    chosen = rng.permutation(fingers)[:pairs]
    angles = np.deg2rad(np.arange(-25, 26, 5.0))

    cache: dict[int, tuple] = {}

    def prepared(index: int):
        if index not in cache:
            img = prints.images[index]
            analysis = ridges.analyse(img)
            cache[index] = (
                img,
                analysis,
                enhance.enhance(img, analysis),
                extractor(img, analysis),
            )
        return cache[index]

    genuine, impostor, counts = [], [], []
    for k, finger in enumerate(chosen):
        idx = np.where(prints.finger == finger)[0]
        if len(idx) < 2:
            continue
        a, b = int(idx[0]), int(idx[1])
        _, a_an, a_enh, a_min = prepared(a)
        _, _, b_enh, b_min = prepared(b)
        counts.extend([len(a_min), len(b_min)])
        tolerance = tolerance_periods * a_an.period

        angle, peak, _ = best_alignment(b_enh, a_enh, angles)
        moved = transform_points(b_min.xy, angle, peak, a_enh.shape)
        genuine.append(landing_rate(moved, a_min.xy, tolerance))

        # An unrelated finger, registered the same way, is the floor: two point
        # sets of this density will always overlap somewhat by accident.
        other = int(chosen[(k + 1) % len(chosen)])
        oidx = np.where(prints.finger == other)[0]
        if len(oidx):
            _, _, o_enh, o_min = prepared(int(oidx[0]))
            angle, peak, _ = best_alignment(o_enh, a_enh, angles)
            moved = transform_points(o_min.xy, angle, peak, a_enh.shape)
            impostor.append(landing_rate(moved, a_min.xy, tolerance))

    return {
        "pairs": len(genuine),
        "genuine": float(np.mean(genuine)) if genuine else 0.0,
        "impostor": float(np.mean(impostor)) if impostor else 0.0,
        "lift": (
            (float(np.mean(genuine)) / float(np.mean(impostor)))
            if impostor and np.mean(impostor) > 0
            else float("nan")
        ),
        "minutiae_median": float(np.median(counts)) if counts else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/prints.npz"))
    ap.add_argument("--repeatability", action="store_true")
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--tolerance", type=float, default=1.0, help="in ridge periods")
    ap.add_argument(
        "--extractor",
        choices=["crossing", "follow", "both"],
        default="both",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import data as data_module

    prints = data_module.load(args.data)
    _, test = data_module.split(prints)

    extractors = {"crossing": minutiae.extract, "follow": extract}
    names = ["crossing", "follow"] if args.extractor == "both" else [args.extractor]

    if not args.repeatability:
        for name in names:
            found = extractors[name](test.images[0])
            print(f"{name:10} {len(found)} minutiae on one print")
        return 0

    print(
        f"Landing rate after registration, {args.pairs} fingers,"
        f" tolerance {args.tolerance} ridge period(s).\n"
        "Genuine is two impressions of one finger; impostor is the floor.\n",
    )
    header = (
        f"{'extractor':<12}{'minutiae':>10}{'genuine':>10}{'impostor':>10}{'lift':>8}"
    )
    print(header)
    print("-" * len(header))
    for name in names:
        r = repeatability(
            test,
            extractors[name],
            pairs=args.pairs,
            tolerance_periods=args.tolerance,
        )
        print(
            f"{name:<12}{r['minutiae_median']:>10.0f}{r['genuine']:>10.3f}"
            f"{r['impostor']:>10.3f}{r['lift']:>8.2f}",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
