# mypy: disable-error-code=import-not-found
"""Draw the post's social card in the blog's house style."""

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


def main() -> int:
    """Write the cover image."""
    figure = plt.figure(
        figsize=(WIDTH / DPI, HEIGHT / DPI),
        dpi=DPI,
        facecolor=PURPLE,
    )
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
    axes.invert_yaxis()
    axes.axis("off")
    axes.set_facecolor(PURPLE)

    axes.add_patch(
        FancyBboxPatch(
            (54, 40),
            310,
            56,
            boxstyle="round,pad=0,rounding_size=28",
            facecolor=PILL,
            edgecolor="none",
        ),
    )
    axes.text(
        209,
        70,
        "SCIENTIFIC PYTHON",
        color=WHITE,
        fontsize=15,
        fontweight="bold",
        ha="center",
        va="center",
        family="DejaVu Sans",
    )

    center_x, center_y, radius = WIDTH / 2, 200, 88
    for offset_x, offset_y in [(-64, -34), (64, -34), (0, 62)]:
        axes.add_patch(
            Circle(
                (center_x + offset_x, center_y + offset_y),
                radius,
                facecolor="none",
                edgecolor=WHITE,
                linewidth=11,
                alpha=0.82,
            ),
        )

    for index, line in enumerate(["NumPy to JAX", "The Array Learned New Tricks"]):
        axes.text(
            WIDTH / 2,
            446 + index * 74,
            line,
            color=WHITE,
            fontsize=38 if index == 0 else 32,
            fontweight="bold",
            ha="center",
            va="center",
            family="DejaVu Sans",
        )

    figure.savefig(OUT, dpi=DPI, facecolor=PURPLE)
    plt.close(figure)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
