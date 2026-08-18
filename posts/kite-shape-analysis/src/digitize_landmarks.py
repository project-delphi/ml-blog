"""Record the hand-digitised sail landmarks and draw the verification overlays.

Six landmarks per frame, all corners of the *band stack* -- the block of
coloured chevrons that makes up the rigid sail. Nothing is placed on a
streamer: the tails flap between frames, so a point on one means something
different in every photograph and destroys the correspondence Procrustes
needs.

Going round the sail: ``nose`` is the outer apex where the two dark navy
leading edges meet; ``armA_out``/``armB_out`` are the far ends of those two
edges; ``armA_in``/``armB_in`` are the far ends of the band stack's inner
boundary on each arm; and ``apex_in`` is the inner apex where those two inner
boundaries meet.

The coordinates below were digitised by hand off the committed half-resolution
photographs, the way a morphometrician would place them in tpsDig, then checked
against the overlays this script writes. Placement is good to roughly +/-5 px on
a sail 150-200 px across. That is ample for the question the post asks -- the
shape differences between frames are tens of percent -- but it is not precise
enough to separate an affine camera model from a projective one, and the post
says so.

Not run by the render::

    .venv-kite/bin/python posts/kite-shape-analysis/src/digitize_landmarks.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

POST = Path(__file__).resolve().parent.parent
PHOTOS = POST / "photos"
CSV_OUT = POST / "landmarks.csv"
OVERLAY_OUT = POST / "fig-landmarks.png"

# Traversal order round the sail. Keeping it fixed is what makes landmark i
# mean the same thing in every frame.
ORDER = ["nose", "armA_out", "armA_in", "apex_in", "armB_in", "armB_out"]

COLOURS = dict(
    zip(ORDER, ["#FFFFFF", "#FF3B30", "#FF9500", "#34C759", "#00C7FF", "#BF5AF2"]),
)

# Hand-digitised pixel coordinates, in the frame of photos/kite-NN.jpg.
LANDMARKS: dict[int, dict[str, tuple[int, int]]] = {
    1: {
        "nose": (805, 971),
        "armA_out": (722, 1012),
        "armA_in": (745, 1017),
        "apex_in": (801, 1022),
        "armB_in": (845, 1035),
        "armB_out": (872, 1050),
    },
    2: {
        "nose": (818, 826),
        "armA_out": (714, 858),
        "armA_in": (732, 872),
        "apex_in": (807, 866),
        "armB_in": (855, 890),
        "armB_out": (881, 901),
    },
    3: {
        "nose": (775, 608),
        "armA_out": (662, 546),
        "armA_in": (690, 588),
        "apex_in": (714, 624),
        "armB_in": (706, 657),
        "armB_out": (718, 700),
    },
    4: {
        "nose": (870, 1078),
        "armA_out": (868, 924),
        "armA_in": (822, 968),
        "apex_in": (820, 1020),
        "armB_in": (748, 1032),
        "armB_out": (720, 1057),
    },
    # Arms A and B are swapped relative to how this frame was read off the
    # photograph. The sail is bilaterally symmetric and its colour bands are
    # symmetric too, so nothing in a single image says which physical wing is
    # which; the labelling is free. Choosing it so every frame's traversal runs
    # the same way round keeps the five configurations comparable without a
    # reflection that would only be an artefact of that free choice.
    5: {
        "nose": (833, 1307),
        "armA_out": (958, 1288),
        "armA_in": (938, 1264),
        "apex_in": (858, 1250),
        "armB_in": (856, 1204),
        "armB_out": (819, 1187),
    },
}


def configuration(frame: int) -> np.ndarray:
    """Return one frame's landmarks as a 6x2 array in ``ORDER``."""
    return np.array([LANDMARKS[frame][name] for name in ORDER], dtype=float)


def signed_area(config: np.ndarray) -> float:
    """Twice the signed area of the landmark polygon (the shoelace formula).

    The sign says which way round the traversal runs in image coordinates. All
    five come out negative, because frame 5's arm labels are swapped above to
    make them. That is the check on that swap: a positive value here would mean
    one frame's landmarks run the opposite way round the sail from the rest.

    It is not, on its own, a reason to allow reflection downstream. The reason
    for that is the same free choice seen from the other side -- nothing in a
    photograph says which physical wing is which -- and the post's alignment
    allows reflection so the answer does not depend on how this swap went.
    """
    x, y = config[:, 0], config[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def write_csv() -> None:
    """Write ``landmarks.csv`` in long form, one row per landmark."""
    with CSV_OUT.open("w", newline="") as handle:
        # Explicit LF: csv.writer defaults to CRLF, which git rewrites on every
        # checkout and shows the file as modified when nothing has changed.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "landmark", "x", "y"])
        for frame in sorted(LANDMARKS):
            for name in ORDER:
                x, y = LANDMARKS[frame][name]
                writer.writerow([frame, name, x, y])


def write_overlay() -> None:
    """Draw every frame's landmarks on its photograph, for checking by eye."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    flat = axes.ravel()
    for frame in sorted(LANDMARKS):
        image = Image.open(PHOTOS / f"kite-{frame:02d}.jpg").convert("RGB")
        config = configuration(frame)
        x0, y0 = config.min(axis=0) - 30
        x1, y1 = config.max(axis=0) + 30
        ax = flat[frame - 1]
        ax.imshow(
            np.asarray(image),
            extent=(0, image.width, image.height, 0),
            interpolation="lanczos",
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        loop = np.vstack([config, config[0]])
        ax.plot(loop[:, 0], loop[:, 1], "-", color="white", lw=1.2, alpha=0.8)
        for name in ORDER:
            ax.plot(
                *LANDMARKS[frame][name],
                "o",
                color=COLOURS[name],
                ms=10,
                mec="black",
                mew=1.0,
            )
        ax.set_title(f"frame {frame}", fontsize=12)
        ax.axis("off")

    legend = flat[5]
    legend.axis("off")
    for row, name in enumerate(ORDER):
        legend.plot(
            0.08,
            0.86 - row * 0.13,
            "o",
            color=COLOURS[name],
            ms=12,
            mec="black",
            transform=legend.transAxes,
        )
        legend.text(
            0.17,
            0.86 - row * 0.13,
            name,
            fontsize=12,
            va="center",
            transform=legend.transAxes,
        )
    fig.tight_layout()
    # 72 rather than a higher dpi on purpose: at 110 this file was 1.3 MB and
    # the largest asset in the repository, for a figure nobody zooms into.
    fig.savefig(OVERLAY_OUT, dpi=72)
    plt.close(fig)


def main() -> int:
    """Write the landmark table and its overlay.

    Returns
    -------
        Process exit code.
    """
    write_csv()
    write_overlay()
    for frame in sorted(LANDMARKS):
        area = signed_area(configuration(frame))
        print(
            f"frame {frame}: signed area {area:9.1f}  ({'CW' if area > 0 else 'CCW'})",
        )
    print(f"\nwrote {CSV_OUT.name} and {OVERLAY_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
