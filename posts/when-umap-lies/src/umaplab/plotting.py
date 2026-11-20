"""House figure style for the series, and the fixed method-to-colour map.

Figures sit on a white card in both site themes, so they use the light half of the
site's categorical tokens (``--w-c1``..``--w-c6`` in ``theme/light.scss``). Six hues
go to the methods the series argues about; the three spectral-family methods share a
muted grey and are told apart by line style and marker; a PCA-50 pipeline arm takes
its parent's hue with a dashed line. A method keeps its colour in every figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

INK, MUTED, RULE = "#1F2430", "#5F6672", "#D8DBE2"
C1, C2, C3, C4, C5, C6 = (
    "#4A3AA7",
    "#1D6E6E",
    "#9A5B00",
    "#A8437A",
    "#0072B2",
    "#009E73",
)
GREY = MUTED

# config label -> (colour, linestyle, marker, display name)
METHOD_STYLE: dict[str, tuple[str, str, str, str]] = {
    "umap": (C1, "-", "o", "UMAP"),
    "umap·pca50": (C1, "--", "o", "UMAP on PCA-50"),
    "umap·densmap": (C1, ":", "o", "densMAP"),
    "umap·nn5": (C1, "--", "o", "UMAP, n_neighbors = 5"),
    "tsne·perp5": (C3, "--", "s", "t-SNE, perplexity = 5"),
    "tsne": (C3, "-", "s", "t-SNE"),
    "tsne·pca50": (C3, "--", "s", "t-SNE on PCA-50"),
    "pca": (C5, "-", "D", "PCA"),
    "pacmap": (C4, "-", "^", "PaCMAP"),
    "trimap": (C6, "-", "v", "TriMap"),
    "phate": (C2, "-", "P", "PHATE"),
    "isomap": (GREY, "-", "x", "Isomap"),
    "spectral": (GREY, ":", "+", "Laplacian eigenmaps"),
    "diffmap": (GREY, "-.", "*", "Diffusion map"),
    "input": (INK, (0, (1, 1)), "", "Input kNN graph"),
}

ORDER = list(METHOD_STYLE)

# Class identity (cell types, digits, clusters): a validated eight-hue categorical
# order, used only for labels in the data, never for methods. More than eight classes
# fall back to two neutrals with direct labels.
CLASS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
    "#5F6672",
    "#1F2430",
]


def class_colours(labels: np.ndarray) -> np.ndarray:
    """One CLASS colour per point, in label order."""
    return np.array(CLASS, dtype=object)[np.asarray(labels) % len(CLASS)]


# Sequential: one hue (c1), light to dark. Diverging: c5 / grey / c3.
SEQ = LinearSegmentedColormap.from_list("seq", ["#F4F5F7", C1])
DIV = LinearSegmentedColormap.from_list("div", [C5, "#E9EAEE", C3])


def setup() -> None:
    """Apply the blog's house matplotlib style (white card, recessive axes)."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.grid": True,
            "grid.color": RULE,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
        }
    )


def style(config: str) -> dict:
    """Matplotlib kwargs for a config label (unknown labels fall back to grey)."""
    colour, ls, marker, name = METHOD_STYLE.get(config, (GREY, "-", "o", config))
    return dict(color=colour, linestyle=ls, marker=marker, label=name)


def name(config: str) -> str:
    """Display name of a config label."""
    return METHOD_STYLE.get(config, (None, None, None, config))[3]


def band(ax, x, lo, hi, config: str, alpha: float = 0.15) -> None:
    """Shade an uncertainty band in the config's colour."""
    ax.fill_between(x, lo, hi, color=style(config)["color"], alpha=alpha, lw=0)


def summarise(df: pd.DataFrame, by: list[str], value: str = "value") -> pd.DataFrame:
    """Median, min and max over seeds — the series reports ranges, not just means."""
    g = df.groupby(by)[value]
    return g.agg(
        median="median", lo="min", hi="max", mean="mean", n="count"
    ).reset_index()


def embedding_panel(ax, Y: np.ndarray, c=None, cmap=None, s: float = 2.0, title=""):
    """A scatter of a 2-D embedding: equal aspect, no axes (coordinates mean nothing)."""
    ax.scatter(Y[:, 0], Y[:, 1], c=c, cmap=cmap, s=s, lw=0, rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, color=INK)


def align_to(ref: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Procrustes-align Y to ref for display (rotation, reflection, scale, shift)."""
    from .stability import procrustes_align

    return procrustes_align(ref, Y)[0]
