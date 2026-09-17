"""House palette and matplotlib defaults, matching the rest of the blog."""

from __future__ import annotations

import matplotlib.pyplot as plt

INK = "#1F2430"
ACCENT = "#4A3AA7"
TEAL = "#2A9D8F"
CORAL = "#C45C26"
GOLD = "#E8A33D"
MUTED = "#5F6672"
RULE = "#D8DBE2"

# Arms keep one colour each across every figure, so a reader who learns the
# mapping in the raster does not relearn it in the power curve.
ARM_COLOUR = {"broad": ACCENT, "focal": CORAL, "none": MUTED}


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": RULE,
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        },
    )
