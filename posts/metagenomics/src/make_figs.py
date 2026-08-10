"""Draw the three figures for the metagenomics post.

Only the first figure plots measured numbers, and every one of them is
transcribed from Table 1 of Amann, Ludwig & Schleifer (1995), *Microbiological
Reviews* 59(1):143-169, with the source recorded beside the data below.
Refreshing the post therefore means editing this file and re-running it. The
other two figures are schematics: they carry no data, only the shape of an
argument made in the prose.

This post has no venv of its own -- run it with any venv that has matplotlib,
or ephemerally::

    uv run --with matplotlib python posts/metagenomics/src/make_figs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent.parent

# House hue plus two steps of it, and a single warm accent for the one thing a
# reader should look at first in each figure. Checked for contrast against the
# light page surface the post renders on.
PURPLE = "#4A3AA7"
PURPLE_MID = "#6A5CBB"
PURPLE_LIGHT = "#B7AEE4"
ACCENT = "#C2570A"
INK = "#1F1D24"
MUTED = "#6B6673"
GRID = "#DCDAE2"
PALE = "#EFEDF6"
WHITE = "#FFFFFF"

WIDTH = 9.0
DPI = 130

# --------------------------------------------------------------------------
# Figure 1 -- culturability by habitat, the great plate count anomaly.
# Source: Amann, Ludwig & Schleifer 1995, Microbiol. Rev. 59(1):143-169,
# Table 1. "Culturability" there is colony-forming units expressed as a
# percentage of the cells counted directly under the microscope in the same
# sample. Single values are point estimates in the table; pairs are the ranges
# it reports across studies of that habitat.
# --------------------------------------------------------------------------
CULTURABILITY = {
    "source": "Amann, Ludwig & Schleifer 1995, Microbiol. Rev. 59(1), Table 1",
    # habitat, low %, high % (equal when the table gives a single value)
    "rows": [
        ("Seawater", 0.001, 0.1),
        ("Freshwater", 0.25, 0.25),
        ("Sediments", 0.25, 0.25),
        ("Soil", 0.3, 0.3),
        ("Mesotrophic lake", 0.1, 1.0),
        ("Unpolluted estuarine water", 0.1, 3.0),
        ("Activated sludge", 1.0, 15.0),
    ],
}


def _style(ax):
    """Apply the shared recessive-axis styling to ``ax``."""
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", colors=MUTED, length=0, labelsize=9.5)


def fig_culturability() -> None:
    """Log-scaled ranges: what fraction of counted cells will grow on a plate."""
    rows = CULTURABILITY["rows"]
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2), dpi=DPI)

    for i, (_, lo, hi) in enumerate(rows):
        if hi > lo:
            ax.plot([lo, hi], [i, i], color=PURPLE, linewidth=7, solid_capstyle="round")
            label = f"{lo:g}–{hi:g}%"
        else:
            label = f"{lo:g}%"
        ax.plot([lo], [i], "o", color=PURPLE, markersize=7)
        if hi > lo:
            ax.plot([hi], [i], "o", color=PURPLE, markersize=7)
        ax.text(hi * 1.5, i, label, va="center", fontsize=9, color=INK)

    # The reference the whole chart exists against: every cell you can see.
    ax.axvline(100, color=ACCENT, linewidth=1.6, linestyle=(0, (4, 3)))
    # Parked low and right, in the empty quadrant the point-estimate rows leave.
    ax.text(
        88,
        2.4,
        "100% = every cell\ncounted under\nthe microscope",
        fontsize=8.8,
        color=ACCENT,
        va="center",
        ha="right",
    )

    ax.set_xscale("log")
    ax.set_xlim(0.0005, 260)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.set_xticks([0.001, 0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"])
    ax.set_xlabel(
        "colony-forming units as a share of cells counted directly (log scale)",
        fontsize=9.5,
        color=MUTED,
    )
    ax.set_title(
        "Almost nothing grows: culturability by habitat",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    _style(ax)
    fig.text(
        0.005,
        0.012,
        f"Source: {CULTURABILITY['source']}.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUT / "fig-plate-count.png", dpi=DPI, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {OUT / 'fig-plate-count.png'}")


# --------------------------------------------------------------------------
# Figure 2 -- schematic: three ways to read the same scoop of DNA.
# No data; the layout is the argument. Four grey bars stand for four genomes
# present in the sample, and what is drawn on top of them is what each method
# actually recovers.
# --------------------------------------------------------------------------
# Shotgun fragments, hand-placed so the panel reads as random-but-even coverage
# rather than a repeating pattern. (genome index, start, length), axis units.
SHOTGUN = [
    (0, 0.04, 0.13), (0, 0.22, 0.10), (0, 0.38, 0.15), (0, 0.60, 0.11), (0, 0.78, 0.14),
    (1, 0.02, 0.11), (1, 0.19, 0.16), (1, 0.41, 0.09), (1, 0.56, 0.13), (1, 0.75, 0.17),
    (2, 0.07, 0.15), (2, 0.27, 0.12), (2, 0.45, 0.14), (2, 0.66, 0.10), (2, 0.82, 0.12),
    (3, 0.03, 0.12), (3, 0.20, 0.14), (3, 0.40, 0.11), (3, 0.58, 0.16), (3, 0.80, 0.13),
]
# What binning gets back: two near-complete genomes, two partial ones. The
# gaps are the point -- a MAG is rarely the whole chromosome.
BINNED = [
    (0, [(0.02, 0.44), (0.50, 0.48)]),
    (1, [(0.02, 0.96)]),
    (2, [(0.06, 0.30), (0.44, 0.22)]),
    (3, [(0.02, 0.35), (0.62, 0.24)]),
]


def _genome_rows(ax, n=4):
    """Draw ``n`` grey bars standing for genomes present in the sample."""
    for i in range(n):
        y = n - 1 - i
        ax.add_patch(
            Rectangle((0.0, y + 0.32), 1.0, 0.36, facecolor=PALE, edgecolor="none")
        )


def fig_three_reads() -> None:
    """Three panels: amplicon, shotgun, and genome-resolved, on one sample."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.2), dpi=DPI)
    titles = [
        ("Amplicon (16S)", "one marker gene, amplified", "who is here"),
        ("Shotgun", "fragments of everything", "who is here, and what they can do"),
        ("Genome-resolved", "assembled, then binned", "draft genomes of the uncultured"),
    ]

    for ax, (head, sub, gets) in zip(axes, titles):
        _genome_rows(ax)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.75, 5.0)
        ax.axis("off")
        ax.text(0, 4.45, head, fontsize=11, color=INK, fontweight="bold", va="bottom")
        ax.text(0, 4.05, sub, fontsize=8.8, color=MUTED, va="bottom")
        ax.text(0, -0.62, f"→ {gets}", fontsize=9, color=ACCENT, va="bottom")

    # Panel 1: the same short marker gene, in the same place, on every genome.
    for i in range(4):
        y = 3 - i
        axes[0].add_patch(
            Rectangle((0.44, y + 0.32), 0.09, 0.36, facecolor=ACCENT, edgecolor="none")
        )

    # Panel 2: fragments scattered across all four genomes.
    for gi, start, length in SHOTGUN:
        y = 3 - gi
        axes[1].add_patch(
            Rectangle(
                (start, y + 0.32), length, 0.36, facecolor=PURPLE_MID, edgecolor="none"
            )
        )

    # Panel 3: those fragments stitched into contigs and grouped per genome.
    for gi, pieces in BINNED:
        y = 3 - gi
        for start, length in pieces:
            axes[2].add_patch(
                Rectangle(
                    (start, y + 0.32), length, 0.36, facecolor=PURPLE, edgecolor="none"
                )
            )
    # A bin is one organism's genome, so the unit is the row, not a box drawn
    # round several of them. The gaps within a row are the honest part, and the
    # footnote below explains them rather than an annotation competing for the
    # little free space this panel has.

    fig.suptitle(
        "Three ways to read the same scoop of DNA",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        x=0.005,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.005,
        0.012,
        "Schematic. Grey bars are genomes present in the sample; colour is what the method recovers. "
        "Gaps in the third panel are what the assembly could not reconstruct.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT / "fig-three-reads.png", dpi=DPI, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {OUT / 'fig-three-reads.png'}")


# --------------------------------------------------------------------------
# Figure 3 -- schematic: where each named tool sits in the pipeline.
# --------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, sub=None, face=PALE, edge=PURPLE_LIGHT, color=INK):
    """Draw one rounded stage box with centred text and an optional sub-label."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.6",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
        )
    )
    ty = y + h / 2 + (1.9 if sub else 0)
    ax.text(
        x + w / 2, ty, text, ha="center", va="center", fontsize=9.6, color=color,
    )
    if sub:
        ax.text(
            x + w / 2,
            y + h / 2 - 2.6,
            sub,
            ha="center",
            va="center",
            fontsize=9.0,
            color=ACCENT,
            fontweight="bold",
        )


def _arrow(ax, x1, y1, x2, y2):
    """Draw one connector between stages."""
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            color=PURPLE_MID,
            linewidth=1.3,
            shrinkA=0,
            shrinkB=0,
        )
    )


def fig_pipeline() -> None:
    """Flow diagram: bench work, then the two computational routes off the reads."""
    fig, ax = plt.subplots(figsize=(WIDTH, 5.6), dpi=DPI)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # The wet-lab band: everything that happens before a computer is involved.
    _box(
        ax,
        2,
        88,
        96,
        9,
        "At the bench:  collect the sample  →  extract all the DNA in it  →  sequence everything",
        face=PURPLE,
        edge=PURPLE,
        color=WHITE,
    )
    _arrow(ax, 50, 88, 50, 81.5)
    _box(ax, 28, 71, 44, 10, "Hundreds of millions of short reads")

    # Split into the two routes.
    _arrow(ax, 50, 71, 50, 66)
    ax.plot([24, 76], [66, 66], color=PURPLE_MID, linewidth=1.3)
    _arrow(ax, 24, 66, 24, 60.5)
    _arrow(ax, 76, 66, 76, 60.5)

    left = [
        ("Match reads against a reference database", "Kraken2  ·  MetaPhlAn"),
        ("Relative abundance for each taxon", None),
        ("Who is here — fast, but only what the\ndatabase already knows", None),
    ]
    right = [
        ("Assemble reads into long contigs", "MEGAHIT  ·  metaSPAdes"),
        ("Bin contigs by coverage and composition", None),
        ("Draft genomes (MAGs) — including of\norganisms nobody has ever cultured", None),
    ]
    ys = [48, 32, 15]
    for lane_x, lane in ((2, left), (54, right)):
        for k, (text, sub) in enumerate(lane):
            h = 12.5 if k == 0 else 11.5
            _box(ax, lane_x, ys[k], 44, h, text, sub=sub)
            if k:
                _arrow(ax, lane_x + 22, ys[k - 1], lane_x + 22, ys[k] + h)

    ax.text(
        50,
        7.5,
        "Wrapped end to end, with a record of exactly what ran:   "
        "QIIME 2 (amplicon)   ·   nf-core/mag (assembly → binning → annotation)",
        ha="center",
        va="center",
        fontsize=9.2,
        color=MUTED,
    )

    fig.suptitle(
        "Where the named tools sit",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        x=0.005,
        ha="left",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "fig-pipeline.png", dpi=DPI, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {OUT / 'fig-pipeline.png'}")


def main() -> int:
    """Write all three figures.

    Returns:
        Process exit code.
    """
    fig_culturability()
    fig_three_reads()
    fig_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
