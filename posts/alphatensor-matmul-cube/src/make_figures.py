r"""Write fig-cube.png: the 2 x 2 matrix multiplication tensor as a stack of trays.

A static copy of the post's 3D scene, for readers whose browser cannot draw it
and as the source of the cover. The tensor is rebuilt here from its definition
rather than read from model.js, so the figure is a second check on it.

Usage (from repo root):

    uv run --no-project --with matplotlib python \
        posts/alphatensor-matmul-cube/src/make_figures.py

`--no-project` matters: a plain `uv run` here deletes and re-syncs the repo venv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

POST = Path(__file__).resolve().parent.parent

INK = "#1F2430"
MUTED = "#5F6672"
RULE = "#C4C8D0"
PURPLE = "#4A3AA7"
GAP = 1.7
SUB = "₀₁₂₃₄₅₆₇₈₉"


def name(letter: str, idx: int, n: int = 2) -> str:
    """Return the name of a matrix entry, such as a₁₂ for flat index 1."""
    return f"{letter}{SUB[idx // n + 1]}{SUB[idx % n + 1]}"


def tensor(n: int = 2) -> np.ndarray:
    """Return the n x n matrix multiplication tensor, indexed [a, b, c]."""
    size = n * n
    t = np.zeros((size, size, size), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                t[i * n + j, j * n + k, i * n + k] = 1
    return t


def main() -> None:
    """Draw the cube and save it beside the post."""
    t = tensor()
    assert t.sum() == 8
    size = t.shape[0]
    fig = plt.figure(figsize=(12, 6.6), dpi=100, facecolor="white")
    ax = fig.add_axes((0.0, -0.06, 1.0, 1.1), projection="3d")
    ax.set_proj_type("ortho")
    for c in range(size):
        z = (size - 1 - c) * GAP
        edge = [-0.5, size - 0.5, size - 0.5, -0.5, -0.5]
        ax.plot(
            edge,
            [-0.5, -0.5, size - 0.5, size - 0.5, -0.5],
            [z] * 5,
            color=RULE,
            lw=1.1,
        )
        for a in range(size):
            for b in range(size):
                x, y = b, size - 1 - a
                if t[a, b, c]:
                    ax.bar3d(
                        x - 0.33,
                        y - 0.33,
                        z,
                        0.66,
                        0.66,
                        0.66,
                        color=PURPLE,
                        shade=True,
                    )
                else:
                    ax.scatter([x], [y], [z + 0.2], color=RULE, s=9, depthshade=False)
        # Beside the tray's right-hand corner, at the tray's own height.
        ax.text(
            size + 0.1,
            size - 0.5,
            z,
            name("c", c),
            color=INK,
            fontsize=15,
            fontweight="bold",
            va="center",
        )
    for i in range(size):
        ax.text(
            i, -1.3, 0, name("b", i), color=MUTED, fontsize=13, ha="center", va="top"
        )
        ax.text(
            -1.3,
            size - 1 - i,
            0,
            name("a", i),
            color=MUTED,
            fontsize=13,
            ha="right",
            va="center",
        )
    ax.set_xlim(-2, size)
    ax.set_ylim(-2, size)
    ax.set_zlim(0, (size - 1) * GAP + 1)
    ax.set_box_aspect((size + 2, size + 2, (size - 1) * GAP + 1), zoom=1.0)
    ax.view_init(elev=15, azim=-62)
    ax.set_axis_off()
    fig.savefig(POST / "fig-cube.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
