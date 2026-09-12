"""Build the post's fingerprint cache from the NIST MINEX III validation imagery.

Run this once, by hand, before running ``src/ladder.py``. The cache it writes
is **not committed** -- ``data/`` is ignored, and 19 MB of derived scans is not
worth carrying in git for code the rendered post does not execute. The post
itself never runs this and never touches the network.

    uv run --with numpy --with scipy python src/fetch_data.py

The source is ``usnistgov/minex`` on GitHub: 801 raw 8-bit grayscale scans of
inked fingerprint cards at 500 dpi, released into the public domain as a US
Government work. Files are named ``<set><subject>_<position>.gray`` -- ``a001_02``
is set A, subject 001, ANSI/NIST finger position 02 (right index). Sets A and B
are two impressions of the same finger, which is what makes the file a matching
benchmark at all; see the module docstring test in ``data.py``.

Per-image width and height are not in the files. They live in a C table,
``minexiii_validation_data.h``, alongside a quality grade and the finger
position, and this script parses that table to know how to reshape each file.

The checkout is ~190 MB and is deleted afterwards. What it leaves behind, under
an ignored ``data/``, is one 192x192 square per scan, resampled so that every
print has the same ridge period and centred on the inked area -- see
``standardise`` and the ``--centre`` flag for why not the ridge core. Nothing
else is taken out: rotation, ink coverage, elastic distortion, damage and how
much of the finger the roll caught are all still there, and are what the
matchers have to survive.

Two earlier versions of this crop were worse, and both failures are worth
knowing. Centring on raw local contrast lands on the printed words at the top of
the card, not on the finger. And cutting a small window at full resolution --
sensor-sized, which sounds right -- throws away the global ridge pattern that
tells a loop from a whorl, and costs about four times the identification
accuracy of keeping the whole print.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import ridges
from scipy import ndimage

REPO = "https://github.com/usnistgov/minex.git"
IMAGERY = "minexiii/validation/validation_imagery_raw"
HEADER = "minexiii/validation/minexiii_validation_data.h"

# {"a001_02.gray", {329, 503, MINEX_QUALITY_FAIR, MINEX_FINGER_RIGHT_INDEX, ...
ENTRY_RE = re.compile(
    r'\{"([ab]\d{3}_\d{2}\.gray)",\s*\{(\d+),\s*(\d+),\s*MINEX_QUALITY_(\w+),'
    r"\s*MINEX_FINGER_(\w+),",
    re.S,
)

QUALITY = {"UNKNOWN": 0, "POOR": 1, "FAIR": 2, "GOOD": 3, "VERYGOOD": 4, "EXCELLENT": 5}

SIZE = 192  # pixels per side of the standardised square

# Every print is resampled to this ridge period, in pixels. The source scans run
# a median of 11.5, so 6.0 halves them and keeps about a quarter of the pixels.
# That is a deliberate trade, not a default nobody revisited: it fits the whole
# print inside a 192-pixel square, which is what the rungs measuring global ridge
# flow need -- a sensor-sized window at full resolution costs them about four
# times the identification accuracy. What it costs is rung 2, whose ridge endings
# and bifurcations are no longer well resolved at this period; rebuilding at 11.5
# measurably improves their repeatability without coming close to fixing it, and
# src/minutiae.py carries the numbers. Raising this means raising SIZE in step,
# and roughly doubles the archive this writes.
TARGET_PERIOD = 6.0
CENTRE = "ink"  # "ink" (centre of the inked area) or "core" (the singular point)


def clone(dest: Path) -> str:
    """Shallow-clone the MINEX repo into `dest`; return the commit sha."""
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", REPO, str(dest)],
        check=True,
    )
    sha = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sha.stdout.strip()


def read_table(repo: Path) -> dict[str, tuple[int, int, int, str]]:
    """Map each .gray filename to (width, height, quality, finger position)."""
    text = (repo / HEADER).read_text()
    return {
        m.group(1): (int(m.group(2)), int(m.group(3)), QUALITY[m.group(4)], m.group(5))
        for m in ENTRY_RE.finditer(text)
    }


def standardise(
    img: np.ndarray,
    size: int = SIZE,
    period: float = TARGET_PERIOD,
    centre: str = CENTRE,
):
    """Put every scan on one scale and one centre.

    Two things vary between cards for reasons that have nothing to do with whose
    finger it is: how large the print was rolled, and where on the card it
    landed. Both are taken out here, and nothing else is.

    Scale is fixed by the ridges themselves -- each print is resampled until its
    own ridge period is `period` pixels, so one filter bank is tuned for all of
    them. Position is fixed by `centre` -- the middle of the inked area by
    default, the ridge core on request. What stays in is the part a matcher has
    to survive: rotation, ink coverage, elastic distortion, damage, and how much
    of the finger the roll actually caught.

    Returns the standardised square and the print's original ridge period.
    """
    own = ridges.analyse(img).period
    zoom = period / own
    # Anti-alias before shrinking. Resampling a 10-pixel ridge period down to 6
    # without it folds the ridges back as high-frequency noise, and the period
    # search then reports that noise instead of the ridges -- which empties the
    # ridge mask and, with it, everything downstream.
    source = img.astype(np.float64)
    if zoom < 1:
        source = ndimage.gaussian_filter(source, 0.5 / zoom)
    scaled = np.clip(ndimage.zoom(source, zoom, order=1), 0, 255).astype(np.uint8)

    analysis = ridges.analyse(scaled)
    if centre == "core":
        cy, cx = ridges.core(analysis)
    else:
        # The middle of the inked area. Less anatomical than the core, but it
        # keeps the most print inside the frame, and how much of the pattern the
        # window holds turns out to matter more than what it is centred on.
        rows, cols = np.indices(analysis.mask.shape)
        w = analysis.mask.astype(float)
        if w.sum() == 0:
            w = np.ones_like(w)
        b = ridges.BLOCK
        cy = int(round((rows * w).sum() / w.sum() * b + b // 2))
        cx = int(round((cols * w).sum() / w.sum() * b + b // 2))
    half = size // 2
    pad = np.full(
        (scaled.shape[0] + size, scaled.shape[1] + size),
        int(np.median(scaled)),
        np.uint8,
    )
    pad[half : half + scaled.shape[0], half : half + scaled.shape[1]] = scaled
    return pad[cy : cy + size, cx : cx + size], own


def build(
    repo: Path,
    out: Path,
    size: int,
    centre: str = CENTRE,
    period: float = TARGET_PERIOD,
    limit: int | None = None,
) -> dict:
    table = read_table(repo)
    raw = repo / IMAGERY

    # A finger is (subject, position). Keep only the ones scanned in both sets --
    # seven of the 801 scans have no mate and cannot be scored, leaving 794.
    fingers = sorted(
        {
            (n[1:4], n[5:7])
            for n in table
            if (raw / ("a" + n[1:])).exists() and (raw / ("b" + n[1:])).exists()
        },
    )

    if limit:
        fingers = fingers[:limit]

    images, finger_id, impression = [], [], []
    subject, position, quality, names, periods = [], [], [], [], []
    for fid, (subj, pos) in enumerate(fingers):
        for imp, setname in enumerate("ab"):
            name = f"{setname}{subj}_{pos}.gray"
            w, h, qual, _ = table[name]
            img = np.frombuffer((raw / name).read_bytes(), np.uint8).reshape(h, w)
            std, own = standardise(img, size, period=period, centre=centre)
            images.append(std)
            periods.append(own)
            finger_id.append(fid)
            impression.append(imp)
            subject.append(int(subj))
            position.append(int(pos))
            quality.append(qual)
            names.append(name)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=np.stack(images),
        finger=np.array(finger_id, np.int16),
        impression=np.array(impression, np.int8),
        subject=np.array(subject, np.int16),
        position=np.array(position, np.uint8),
        quality=np.array(quality, np.uint8),
        source_period=np.array(periods, np.float32),
        name=np.array(names),
    )
    return {
        "fingers": len(fingers),
        "images": len(images),
        "size": size,
        "centre": centre,
        "period": period,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        type=Path,
        help="existing MINEX checkout; cloned to a tempdir if unset",
    )
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--centre", choices=["ink", "core"], default=CENTRE)
    ap.add_argument(
        "--period",
        type=float,
        default=TARGET_PERIOD,
        help="ridge period every print is resampled to, in pixels. Below the"
        " source period (median 11.5 at 500 dpi) this discards detail: ridge"
        " flow survives it, individual endings and bifurcations do not",
    )
    ap.add_argument(
        "--limit",
        type=int,
        help="build from the first N fingers only, for testing",
    )
    ap.add_argument("--out", type=Path, default=Path("data/prints.npz"))
    args = ap.parse_args()

    tmp = None
    if args.repo:
        repo, sha = args.repo, "(local checkout)"
    else:
        tmp = Path(tempfile.mkdtemp(prefix="minex-"))
        repo, sha = tmp, clone(tmp)

    try:
        stats = build(
            repo,
            args.out,
            args.size,
            args.centre,
            args.period,
            args.limit,
        )
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    manifest = {
        "source": "NIST MINEX III validation imagery",
        "repository": REPO,
        "commit": sha,
        "path": IMAGERY,
        "licence": "Public domain (US Government work); see LICENSE.md in the repository",
        "retrieved": date.today().isoformat(),
        "resolution_dpi": 500,
        "size_px": stats["size"],
        "target_ridge_period_px": stats["period"],
        "standardisation": (
            "resampled to a common ridge period, then a square window centred on"
            f" the {stats['centre']} centre; see standardise() and src/ridges.py"
        ),
        "fingers": stats["fingers"],
        "images": stats["images"],
        "impressions_per_finger": 2,
        "impression_sets": {"0": "set A", "1": "set B"},
        "file": args.out.name,
        "sha256": digest,
        "bytes": args.out.stat().st_size,
    }
    (args.out.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )

    print(f"{stats['images']} images from {stats['fingers']} fingers -> {args.out}")
    print(f"{args.out.stat().st_size / 1e6:.2f} MB  sha256 {digest[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
