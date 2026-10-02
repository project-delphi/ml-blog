r"""Write fig-strassen-blocks.png: Strassen's seven blocks and their sum.

One column per term, one row per slice of the cube. The eighth column is the
sum of the seven, which is the multiplication tensor. The factors are typed in
here rather than read from model.js, so the figure is a second check on them.

Usage (from repo root):

    uv run --no-project --with matplotlib python \
        posts/alphatensor-seven-multiplications/src/make_figures.py

`--no-project` matters: a plain `uv run` here deletes and re-syncs the repo venv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

POST = Path(__file__).resolve().parent.parent

INK = "#1F2430"
MUTED = "#5F6672"
RULE = "#D8DBE2"
PURPLE = "#4A3AA7"
AMBER = "#9A5B00"
SUB = "₀₁₂₃₄₅₆₇₈₉"

# Rows are the seven products; columns weight (a11, a12, a21, a22) and so on.
U = np.array(
    [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
)
V = np.array(
    [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
)
W = np.array(
    [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
)


def tensor(n: int = 2) -> np.ndarray:
    """Return the n x n matrix multiplication tensor, indexed [a, b, c]."""
    size = n * n
    t = np.zeros((size, size, size), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                t[i * n + j, j * n + k, i * n + k] = 1
    return t


def grid(ax: plt.Axes, x0: float, y0: float, values: np.ndarray) -> None:
    """Draw one 4 x 4 slice with its top-left corner at (x0, y0)."""
    for a in range(4):
        for b in range(4):
            v = values[a, b]
            face = PURPLE if v > 0 else AMBER if v < 0 else "white"
            ax.add_patch(
                Rectangle(
                    (x0 + b, y0 - a - 1),
                    0.92,
                    0.92,
                    facecolor=face,
                    edgecolor=RULE,
                    lw=0.8,
                )
            )


def main() -> None:
    """Draw the seven blocks and their sum and save the figure beside the post."""
    blocks = [np.einsum("a,b,c->abc", U[r], V[r], W[r]) for r in range(7)]
    total = sum(blocks)
    assert np.array_equal(total, tensor()), (
        "Strassen's seven blocks must sum to the tensor"
    )
    fig, ax = plt.subplots(figsize=(12, 6.3), dpi=100, facecolor="white")
    step_x, step_y = 5.3, 4.9
    for col, values in enumerate([*blocks, total]):
        x0 = col * step_x + (1.6 if col == 7 else 0.0)
        title = f"m{SUB[col + 1]}" if col < 7 else "sum"
        ax.text(
            x0 + 2,
            1.0,
            title,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color=INK,
        )
        for c in range(4):
            grid(ax, x0, -c * step_y, values[:, :, c])
    for c in range(4):
        label = f"slice c{SUB[c // 2 + 1]}{SUB[c % 2 + 1]}"
        ax.text(
            -0.8,
            -c * step_y - 2,
            label,
            ha="right",
            va="center",
            fontsize=13,
            color=MUTED,
        )
    ax.text(
        7 * step_x + 0.4,
        -1.5 * step_y - 2.2,
        "=",
        ha="center",
        va="center",
        fontsize=22,
        color=MUTED,
    )
    ax.set_xlim(-6.5, 7 * step_x + 6.4)
    ax.set_ylim(-3 * step_y - 4.6, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(POST / "fig-strassen-blocks.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
