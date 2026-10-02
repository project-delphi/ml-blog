r"""Write fig-uphill.png: what is left of the cube, move by move, on two routes.

The schoolbook route clears one cell a move and wins in eight. Strassen's route
wins in seven and has more nonzero cells after its first move than before it.
Both trails are recomputed here from the factors rather than read from
model.js, so the figure is a second check on them.

Usage (from repo root):

    uv run --no-project --with matplotlib python \
        posts/alphatensor-tensor-game/src/make_figures.py

`--no-project` matters: a plain `uv run` here deletes and re-syncs the repo venv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

POST = Path(__file__).resolve().parent.parent

INK = "#1F2430"
MUTED = "#5F6672"
RULE = "#D8DBE2"
PURPLE = "#4A3AA7"
AMBER = "#9A5B00"

# Rows are Strassen's seven products; columns weight (a11, a12, a21, a22) and so on.
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


def trail(blocks: list[np.ndarray]) -> list[int]:
    """Return the nonzero cells left after each block is subtracted in turn."""
    left = tensor()
    out = [int(np.count_nonzero(left))]
    for block in blocks:
        left = left - block
        out.append(int(np.count_nonzero(left)))
    return out


def main() -> None:
    """Draw both trails and save the figure beside the post."""
    strassen = trail([np.einsum("a,b,c->abc", U[r], V[r], W[r]) for r in range(7)])
    cells = np.argwhere(tensor() == 1)
    single = []
    for a, b, c in cells:
        block = np.zeros((4, 4, 4), dtype=int)
        block[a, b, c] = 1
        single.append(block)
    school = trail(single)
    assert strassen[-1] == 0 and school[-1] == 0
    fig, ax = plt.subplots(figsize=(12, 6.3), dpi=100, facecolor="white")
    ax.plot(range(len(school)), school, color=AMBER, lw=3, marker="o", ms=9)
    ax.plot(range(len(strassen)), strassen, color=PURPLE, lw=3, marker="o", ms=9)
    ax.annotate(
        "one cell at a time:\ndownhill every move, done in 8",
        xy=(5, school[5]),
        xytext=(4.6, 0.6),
        color=AMBER,
        fontsize=15,
        fontweight="bold",
        ha="right",
    )
    ax.annotate(
        "Strassen's moves:\nuphill first, done in 7",
        xy=(2, strassen[2]),
        xytext=(3.4, 11.6),
        color=PURPLE,
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("moves played", fontsize=14, color=INK)
    ax.set_ylabel("nonzero cells left in the cube", fontsize=14, color=INK)
    ax.set_xticks(range(9))
    ax.set_yticks(range(0, 14, 2))
    ax.set_ylim(-0.6, 13.4)
    ax.tick_params(colors=MUTED, labelsize=13)
    ax.grid(axis="y", color=RULE, lw=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.12)
    fig.savefig(POST / "fig-uphill.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
