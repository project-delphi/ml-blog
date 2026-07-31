"""House style, seeding, and the source-echo helper.

Two things here are load-bearing for the post rather than cosmetic.

``rng_for`` derives an independent generator per experiment *by name*, so adding
an experiment never shifts the random stream of an existing one — a figure does
not change because something unrelated was inserted above it.

``show_source`` renders a function's real source into the page via
``inspect.getsource``. Every code block in the post that shows a method comes
through it, so the code the reader sees is by construction the code that ran.
There is no second, prettified copy to drift out of sync.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown

SEED = 20260731

# Palette. INK/MUTED/GRID/SURFACE follow the site's other computational posts;
# the three data hues are chosen to stay distinguishable in greyscale and to
# clear WCAG AA against SURFACE for anything that also appears in running text.
INK = "#171C1B"
MUTED = "#67706E"
GRID = "#DEDFDA"
SURFACE = "#FCFCFB"

BLUE = "#1D5C6E"  # gradient-based methods
PLUM = "#7A3B6B"  # removal-based methods
OCHRE = "#8A5E06"  # axiomatic / integrated methods
SLATE = "#3C4B67"  # everything statistical

# Diverging map for signed attributions, sequential for magnitudes. Attribution
# maps must never use a rainbow: hue ordering is not perceptually monotone and
# invents structure that is not in the numbers.
CMAP_SIGNED = "RdBu_r"
CMAP_MAG = "magma"

SPECIES_COLORS = {"setosa": BLUE, "versicolor": OCHRE, "virginica": PLUM}


def use_house_style() -> None:
    """Apply the blog's figure defaults to the global matplotlib state."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.dpi": 140,
        }
    )


def rng_for(name: str) -> np.random.Generator:
    """Return a generator seeded from SEED and the experiment's name.

    Args:
        name: Identifier of the experiment, e.g. ``"occlusion"``.

    Returns:
        A NumPy generator whose stream depends only on SEED and ``name``.
    """
    digest = hashlib.sha256(f"{SEED}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _strip_docstring(src: str) -> str:
    """Remove a function's docstring, keeping every other line verbatim.

    Deleting the line range the parser reports, rather than round-tripping
    through ``ast.unparse``, is what keeps the inline comments — which are
    frequently the part worth reading.

    Args:
        src: Source of a single function or class.

    Returns:
        The same source with its leading docstring removed.
    """
    tree = ast.parse(src)
    node = tree.body[0]
    doc = node.body[0] if node.body else None
    if not (isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)):
        return src
    lines = src.splitlines()
    del lines[doc.lineno - 1 : doc.end_lineno]
    return "\n".join(lines)


def show_source(*objs: Callable, docstrings: bool = False) -> Markdown:
    """Render the real source of one or more callables as a Python code block.

    Every method shown in the post goes through here, so the code on the page is
    the code that ran. Docstrings are dropped by default only because the prose
    around the block is already saying what the docstring says; nothing else is
    altered, including the comments.

    Args:
        *objs: Functions or classes to display.
        docstrings: Whether to keep the docstring.

    Returns:
        A Markdown object holding a fenced Python block.
    """
    blocks = []
    for obj in objs:
        src = textwrap.dedent(inspect.getsource(obj))
        blocks.append(src if docstrings else _strip_docstring(src))
    return Markdown("```python\n" + "\n\n".join(b.rstrip() for b in blocks) + "\n```")


def md_table(df, floatfmt: str = ".3f", index: bool = False) -> Markdown:
    """Render a DataFrame as a Markdown table.

    Args:
        df: The frame to render.
        floatfmt: Format string passed to tabulate.
        index: Whether to include the index column.

    Returns:
        A Markdown object holding the table.
    """
    return Markdown(df.to_markdown(index=index, floatfmt=floatfmt))


def digit_axis(ax, image: np.ndarray, title: str | None = None) -> None:
    """Draw a 28x28 MNIST image on a bare axis.

    Args:
        ax: Target matplotlib axis.
        image: Array of shape (28, 28) in [0, 1].
        title: Optional axis title.
    """
    ax.imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)


def overlay_axis(
    ax,
    image: np.ndarray,
    heat: np.ndarray,
    title: str | None = None,
    signed: bool = False,
    alpha: float = 0.75,
) -> None:
    """Draw an attribution heatmap over a faint copy of the digit.

    The digit is drawn at low alpha underneath so the reader can see *where* on
    the stroke the attribution lands; the heatmap carries the colour.

    Args:
        ax: Target matplotlib axis.
        image: Array of shape (28, 28) in [0, 1].
        heat: Attribution map of shape (28, 28).
        title: Optional axis title.
        signed: If True, use a diverging map centred on zero; else a magnitude map.
        alpha: Opacity of the heatmap layer.
    """
    ax.imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest", alpha=0.30)
    if signed:
        lim = float(np.abs(heat).max()) or 1.0
        ax.imshow(
            heat, cmap=CMAP_SIGNED, vmin=-lim, vmax=lim, alpha=alpha, interpolation="bilinear"
        )
    else:
        ax.imshow(heat, cmap=CMAP_MAG, alpha=alpha, interpolation="bilinear")
    # The stroke outline goes on top. A diverging map is mostly white in its
    # middle and swallows the digit underneath it, which makes every "the
    # attribution sits on the loop" claim in the prose unverifiable by eye.
    ax.contour(image, levels=[0.5], colors=[INK], linewidths=0.7, alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)


def save(fig: plt.Figure, path) -> None:
    """Write a figure to disk with the house background.

    Args:
        fig: Figure to write.
        path: Destination path.
    """
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
