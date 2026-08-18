"""Draw the post's social card in the blog's house style.

Matches the existing cards (see ``posts/metagenomics``): a solid #4A3AA7 field,
a translucent rounded category pill top-left, the white triple-ring logomark
that the site favicon also uses, and bold centred title text. Not run by the
render -- regenerate the card by hand with the post's own venv, which carries
matplotlib for the in-post figures::

    .venv-kite/bin/python \
        posts/kite-shape-analysis/src/make_cover.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "cover.png"

PURPLE = "#4A3AA7"
PILL = "#6A5CBB"
WHITE = "#FFFFFF"

WIDTH, HEIGHT = 1200, 630
DPI = 100

BADGE = "STATISTICS & GEOMETRY"
TITLE = ["Five Photos of One Kite,", "Five Different Shapes"]


def main() -> int:
    """Write ``cover.png``.

    Returns
    -------
        Process exit code.
    """
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=PURPLE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_facecolor(PURPLE)

    # Category pill, top-left.
    pill_w = 330
    ax.add_patch(
        FancyBboxPatch(
            (54, 40),
            pill_w,
            56,
            boxstyle="round,pad=0,rounding_size=28",
            facecolor=PILL,
            edgecolor="none",
        ),
    )
    ax.text(
        54 + pill_w / 2,
        70,
        BADGE,
        color=WHITE,
        fontsize=15,
        fontweight="bold",
        ha="center",
        va="center",
        family="DejaVu Sans",
    )

    # Triple-ring logomark: three overlapping circles on a triangle.
    cx, cy, r = WIDTH / 2, 200, 88
    offsets = [(-64, -34), (64, -34), (0, 62)]
    for dx, dy in offsets:
        ax.add_patch(
            Circle(
                (cx + dx, cy + dy),
                r,
                facecolor="none",
                edgecolor=WHITE,
                linewidth=11,
                alpha=0.82,
            ),
        )

    # Title, centred, one line per entry.
    for k, line in enumerate(TITLE):
        ax.text(
            WIDTH / 2,
            446 + k * 74,
            line,
            color=WHITE,
            fontsize=32,
            fontweight="bold",
            ha="center",
            va="center",
            family="DejaVu Sans",
        )

    fig.savefig(OUT, dpi=DPI, facecolor=PURPLE)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
