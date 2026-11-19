"""Figure style and table helper shared by the series."""

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown

INK, MUTED, RULE = "#1F2430", "#5F6672", "#D8DBE2"
PURPLE, TEAL, AMBER, PINK, BLUE = "#4A3AA7", "#1D6E6E", "#9A5B00", "#A8437A", "#0072B2"
GREEN = "#009E73"


def setup() -> None:
    """Apply the blog's house matplotlib style."""
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
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def md(df: pd.DataFrame, **kw) -> Markdown:
    """Render a DataFrame as a Markdown table without re-parsing numbers."""
    kw.setdefault("index", False)
    return Markdown(df.to_markdown(disable_numparse=True, **kw))


def pct(x: float, d: int = 1) -> str:
    """Format a proportion as a percentage string."""
    return f"{100 * x:.{d}f}%"
