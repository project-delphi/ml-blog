"""High-resolution assay schematic for the unmixing section.

Usage (from repo root):

    .venv-tensor-factorizations/bin/python \\
        posts/uses-of-tensor-factorizations/src/make_mixing_poster.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
    Wedge,
)
from matplotlib.patheffects import withStroke
from matplotlib.transforms import Affine2D
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensors as T

INK = "#1B2433"
MUTED = "#5C6570"
RULE = "#D5D9E2"
CARD = "#F4F6FA"
ACCENT = "#4A3AA7"
TEAL = "#2A9D8F"
CORAL = "#E07A5F"
GOLD = "#E0B44A"
SKIN = "#C9956C"
HAIR = "#2B211C"
COAT = "#F4F6FA"
DYE = (ACCENT, TEAL, CORAL)
NAMES = ("dye A", "dye B", "dye C")
POST = Path(__file__).resolve().parent.parent
OUT = POST / "media" / "mixing-assay.png"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 13.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _cmap(hex_color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "dye", ["#FFFFFF", hex_color + "22", hex_color]
    )


def _to_rgb(z: np.ndarray, cmap="inferno") -> np.ndarray:
    z = np.abs(np.asarray(z, dtype=float))
    z = z / (float(z.max()) + 1e-12)
    rgba = plt.colormaps[cmap](z)
    return (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)


def _hi(z: np.ndarray, factor: float = 10.0) -> np.ndarray:
    return zoom(np.asarray(z, dtype=float), factor, order=3)


def iso_xy(x, y, z, origin, scale=1.0, depth=0.40):
    return np.array(
        [
            origin[0] + scale * (x + depth * z),
            origin[1] + scale * (y + 0.48 * depth * z),
        ]
    )


def imshow_quad(ax, image, bl, br, tr, tl, zorder=5, lw=0.7):
    image = np.clip(np.asarray(image), 0, 255).astype(np.uint8)
    bl, br, tr, tl = map(np.asarray, (bl, br, tr, tl))
    vx, vy = br - bl, tl - bl
    trans = Affine2D.from_values(
        vx[0], vx[1], vy[0], vy[1], bl[0], bl[1]
    ) + ax.transData
    im = ax.imshow(
        image,
        origin="upper",
        extent=(0, 1, 0, 1),
        transform=trans,
        interpolation="bicubic",
        aspect="auto",
        zorder=zorder,
        clip_on=True,
    )
    im.set_clip_path(Polygon([bl, br, tr, tl], closed=True).get_path(), ax.transData)
    ax.add_patch(
        Polygon(
            [bl, br, tr, tl],
            closed=True,
            fill=False,
            edgecolor=INK,
            lw=lw,
            zorder=zorder + 1,
        )
    )
    return im


def _header(ax, letter: str, title: str, *, y: float = 1.08, va: str = "bottom") -> None:
    """Letter badge + title. Default sits above the axes so it never covers the drawing."""
    ax.text(
        0.0,
        y,
        f"  {letter}  ",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color="white",
        ha="left",
        va=va,
        zorder=20,
        clip_on=False,
        bbox=dict(boxstyle="square,pad=0.38", facecolor=ACCENT, edgecolor="none"),
    )
    ax.text(
        0.11,
        y,
        title,
        transform=ax.transAxes,
        fontsize=16.5,
        color=INK,
        ha="left",
        va=va,
        zorder=20,
        clip_on=False,
        fontweight="bold",
    )


def _arrow(ax, p0, p1, color, lw=2.0, style="-|>", z=8):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=13,
            lw=lw,
            color=color,
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _load():
    rng = np.random.default_rng(T.SEED)
    cube, true = T.make_mixing_cube(rng)
    dye_eems = [np.outer(true[1][:, r], true[2][:, r]) for r in range(3)]
    return cube, true, dye_eems


def _scientist(ax, x: float, y: float, s: float = 1.0) -> None:
    """Stylized scientist in a lab coat, facing the instrument (right)."""
    ax.add_patch(Rectangle((x - 0.22 * s, y), 0.18 * s, 1.05 * s, facecolor="#2A3140", edgecolor=INK, lw=0.65, zorder=5))
    ax.add_patch(Rectangle((x + 0.06 * s, y), 0.18 * s, 1.05 * s, facecolor="#2A3140", edgecolor=INK, lw=0.65, zorder=5))
    ax.add_patch(Ellipse((x - 0.13 * s, y), 0.20 * s, 0.09 * s, facecolor="#1A1F28", edgecolor=INK, lw=0.4, zorder=6))
    ax.add_patch(Ellipse((x + 0.15 * s, y), 0.20 * s, 0.09 * s, facecolor="#1A1F28", edgecolor=INK, lw=0.4, zorder=6))
    coat = [
        (x - 0.52 * s, y + 0.98 * s),
        (x + 0.54 * s, y + 0.98 * s),
        (x + 0.62 * s, y + 2.48 * s),
        (x - 0.56 * s, y + 2.48 * s),
    ]
    ax.add_patch(Polygon(coat, closed=True, facecolor=COAT, edgecolor=INK, lw=1.15, zorder=6))
    ax.add_patch(
        Rectangle((x - 0.045 * s, y + 1.05 * s), 0.09 * s, 1.32 * s, facecolor="#E2E6EF", edgecolor=INK, lw=0.5, zorder=7)
    )
    ax.add_patch(
        Polygon(
            [
                (x + 0.48 * s, y + 2.22 * s),
                (x + 1.12 * s, y + 1.82 * s),
                (x + 1.16 * s, y + 2.02 * s),
                (x + 0.54 * s, y + 2.42 * s),
            ],
            closed=True,
            facecolor=COAT,
            edgecolor=INK,
            lw=0.85,
            zorder=7,
        )
    )
    ax.add_patch(Circle((x + 1.20 * s, y + 1.90 * s), 0.12 * s, facecolor=SKIN, edgecolor=INK, lw=0.6, zorder=8))
    ax.add_patch(Circle((x + 0.04 * s, y + 2.86 * s), 0.34 * s, facecolor=SKIN, edgecolor=INK, lw=0.85, zorder=8))
    ax.add_patch(Wedge((x + 0.04 * s, y + 2.90 * s), 0.36 * s, 10, 176, facecolor=HAIR, edgecolor=INK, lw=0.5, zorder=9))
    ax.add_patch(Ellipse((x + 0.04 * s, y + 3.14 * s), 0.48 * s, 0.18 * s, facecolor=HAIR, edgecolor=INK, lw=0.4, zorder=9))
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.82 * s, y + 1.52 * s),
            0.50 * s,
            0.68 * s,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor="#FFFDF7",
            edgecolor=INK,
            lw=0.75,
            zorder=8,
        )
    )
    ax.text(x - 0.57 * s, y + 1.92 * s, "EEM", ha="center", va="center", fontsize=9, color=MUTED, zorder=9, fontweight="bold")


def _fluorometer(ax, x0: float, y0: float, s: float = 1.0, eem: np.ndarray | None = None) -> None:
    """Bench fluorometer: lamp, sample in the beam path, detector at 90°."""
    W, H = 7.35 * s, 5.85 * s

    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            W,
            H,
            boxstyle="round,pad=0.03,rounding_size=0.18",
            facecolor="#243044",
            edgecolor=INK,
            lw=1.45,
            zorder=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x0 + 0.22 * s, y0 + 0.30 * s),
            W - 0.44 * s,
            H - 0.62 * s,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#1A2333",
            edgecolor="#3A465A",
            lw=0.85,
            zorder=3,
        )
    )
    ax.text(
        x0 + 0.50 * W,
        y0 + H - 0.48 * s,
        "spectrofluorometer",
        ha="center",
        fontsize=12,
        color="#D7DCE6",
        fontweight="bold",
        zorder=12,
    )

    # Lamp
    ax.add_patch(
        FancyBboxPatch(
            (x0 + 0.40 * s, y0 + 2.05 * s),
            1.28 * s,
            1.72 * s,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            facecolor="#3D3420",
            edgecolor=GOLD,
            lw=1.15,
            zorder=4,
        )
    )
    ax.add_patch(
        Circle((x0 + 1.04 * s, y0 + 2.95 * s), 0.34 * s, facecolor=GOLD, edgecolor="#F6E7A8", lw=0.7, zorder=5, alpha=0.95)
    )
    ax.add_patch(Circle((x0 + 1.04 * s, y0 + 2.95 * s), 0.52 * s, facecolor=GOLD, edgecolor="none", lw=0, zorder=4, alpha=0.18))
    ax.text(x0 + 1.04 * s, y0 + 2.22 * s, "lamp", ha="center", fontsize=10, color="#F6E7A8", zorder=6)

    # Excitation beam through the cuvette
    beam = [
        (x0 + 1.70 * s, y0 + 2.72 * s),
        (x0 + 1.70 * s, y0 + 3.18 * s),
        (x0 + 4.05 * s, y0 + 3.28 * s),
        (x0 + 4.05 * s, y0 + 2.62 * s),
    ]
    ax.add_patch(Polygon(beam, closed=True, facecolor=GOLD, edgecolor="none", alpha=0.62, zorder=5))
    ax.add_patch(Polygon(beam, closed=True, facecolor="#FFF3B8", edgecolor="none", alpha=0.28, zorder=5))
    ax.text(
        x0 + 2.70 * s,
        y0 + 3.52 * s,
        r"excitation beam  $\lambda_{\mathrm{ex}}$",
        fontsize=11,
        color=GOLD,
        ha="center",
        zorder=12,
        fontweight="bold",
    )

    # Sample cuvette in the beam
    cx, cy = x0 + 4.22 * s, y0 + 2.28 * s
    ax.add_patch(
        Rectangle((cx, cy), 0.82 * s, 1.52 * s, facecolor="#8FD0E8", edgecolor="#EAF6FB", lw=1.2, alpha=0.90, zorder=7)
    )
    ax.add_patch(
        Rectangle((cx - 0.07 * s, cy + 1.44 * s), 0.96 * s, 0.18 * s, facecolor="#D7DEE8", edgecolor=INK, lw=0.65, zorder=8)
    )
    for dx, dy, col, r in (
        (0.24 * s, 0.58 * s, ACCENT, 0.22 * s),
        (0.46 * s, 0.44 * s, TEAL, 0.20 * s),
        (0.38 * s, 0.82 * s, CORAL, 0.17 * s),
    ):
        ax.add_patch(Circle((cx + dx, cy + dy), r, facecolor=col, edgecolor="white", lw=0.45, alpha=0.58, zorder=9))
    ax.text(
        cx + 0.41 * s,
        cy - 0.38 * s,
        "sample",
        ha="center",
        fontsize=11,
        color="#E8EDF5",
        zorder=12,
        fontweight="bold",
    )

    # Emission beam at 90°
    em = [
        (cx + 0.24 * s, cy + 1.52 * s),
        (cx + 0.58 * s, cy + 1.52 * s),
        (cx + 0.70 * s, y0 + 4.42 * s),
        (cx + 0.18 * s, y0 + 4.42 * s),
    ]
    ax.add_patch(Polygon(em, closed=True, facecolor=CORAL, edgecolor="none", alpha=0.55, zorder=6))
    ax.text(
        cx + 1.22 * s,
        y0 + 3.92 * s,
        r"emission beam  $\lambda_{\mathrm{em}}$",
        fontsize=11,
        color="#F3C1B3",
        ha="left",
        zorder=12,
        fontweight="bold",
    )

    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.18 * s, y0 + 4.32 * s),
            1.18 * s,
            0.62 * s,
            boxstyle="round,pad=0.015,rounding_size=0.07",
            facecolor="#3A2A58",
            edgecolor=ACCENT,
            lw=1.15,
            zorder=8,
        )
    )
    ax.text(cx + 0.41 * s, y0 + 4.63 * s, "detector", ha="center", va="center", fontsize=10, color="#EDE4FF", zorder=9)

    # Live EEM on the instrument screen
    sx, sy, sw, sh = x0 + 5.55 * s, y0 + 1.15 * s, 1.42 * s, 1.62 * s
    ax.add_patch(
        FancyBboxPatch(
            (sx, sy),
            sw,
            sh,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#0E131C",
            edgecolor="#6B7384",
            lw=0.85,
            zorder=6,
        )
    )
    if eem is not None:
        pad = 0.10 * s
        ax.imshow(
            _to_rgb(np.flipud(_hi(eem, 8))),
            origin="upper",
            extent=(sx + pad, sx + sw - pad, sy + pad, sy + sh - 0.32 * s),
            aspect="auto",
            interpolation="bicubic",
            zorder=7,
        )
    ax.text(sx + 0.50 * sw, sy + sh - 0.16 * s, "live EEM", ha="center", fontsize=9, color="#AEB6C4", zorder=8)


def _draw_assay(ax, true: list[np.ndarray], cube: np.ndarray) -> None:
    ax.set_xlim(0, 13.4)
    ax.set_ylim(-0.05, 11.2)
    ax.set_facecolor(CARD)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RULE)
    _header(ax, "A", "A scientist records one map per well")

    _scientist(ax, 1.55, 3.15, s=1.42)
    _fluorometer(ax, 4.55, 3.05, s=1.08, eem=cube[10])

    ax.text(
        8.4,
        2.55,
        "Shine one colour.  Collect another.  Sweep both.",
        ha="center",
        va="top",
        fontsize=13.5,
        color=INK,
    )

    amounts = true[0] / (true[0].max(axis=0, keepdims=True) + 1e-12)
    n_s = cube.shape[0]
    ax.text(0.55, 1.15, f"{n_s} wells", ha="left", va="center", fontsize=12, color=MUTED, fontweight="bold")
    ax.text(
        8.4,
        1.55,
        "One well  →  one emission × excitation map.",
        ha="center",
        fontsize=13.5,
        color=INK,
    )
    x0, y0, rad = 2.35, 0.42, 0.18
    span = 13.4 - x0 - 0.55
    step = span / n_s
    for i in range(n_s):
        w = amounts[i]
        w = w / (w.sum() + 1e-12)
        rgb = np.clip(
            w[0] * np.array([0x4A, 0x3A, 0xA7]) / 255
            + w[1] * np.array([0x2A, 0x9D, 0x8F]) / 255
            + w[2] * np.array([0xE0, 0x7A, 0x5F]) / 255,
            0,
            1,
        )
        ax.add_patch(
            Circle(
                (x0 + (i + 0.5) * step, y0),
                rad,
                facecolor=rgb,
                edgecolor="white",
                lw=0.6,
                zorder=4,
            )
        )


def _dim_arrow(ax, p0, p1, label, outward=0.22):
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.35, shrinkA=0, shrinkB=0),
        zorder=14,
    )
    mid = (p0 + p1) / 2
    v = p1 - p0
    n = np.array([-v[1], v[0]])
    n = n / (float(np.linalg.norm(n)) + 1e-9)
    loc = mid + n * outward
    ax.text(loc[0], loc[1], label, ha="center", va="center", fontsize=13.5, color=INK, fontweight="bold", zorder=15)


def _draw_cube(ax, cube: np.ndarray) -> None:
    ax.set_xlim(0, 11.6)
    ax.set_ylim(-0.15, 11.4)
    ax.set_facecolor(CARD)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RULE)
    n_s, n_em, n_ex = cube.shape
    _header(ax, "B", rf"Those maps stack into a tensor  ${n_s}\times {n_em}\times {n_ex}$")

    i_s, i_em, i_ex = 10, int(0.50 * (n_em - 1)), int(0.52 * (n_ex - 1))
    front = np.flipud(_hi(cube[i_s]))
    right = np.flipud(_hi(cube[:, :, i_ex].T))
    top = _hi(cube[:, i_em, :])

    origin = np.array([2.35, 3.35])
    scale, W, H, D = 3.15, 1.55, 1.70, 1.45
    p = lambda x, y, z: iso_xy(x, y, z, origin, scale=scale)

    sh = [p(0.14, -0.10, 0.14), p(W + 0.14, -0.10, 0.14), p(W + 0.14, -0.10, D + 0.14), p(0.14, -0.10, D + 0.14)]
    ax.add_patch(Polygon(sh, closed=True, facecolor="#DEE2EA", edgecolor="none", zorder=1, alpha=0.95))

    tl, tr, trz, tlz = p(0, H, 0), p(W, H, 0), p(W, H, D), p(0, H, D)
    br, brz = p(W, 0, 0), p(W, 0, D)
    bl = p(0, 0, 0)

    imshow_quad(ax, _to_rgb(top), tl, tr, trz, tlz, zorder=2, lw=0.85)
    imshow_quad(ax, _to_rgb(right), br, brz, trz, tr, zorder=3, lw=0.85)
    imshow_quad(ax, _to_rgb(front), bl, br, tr, tl, zorder=6, lw=1.0)

    _dim_arrow(ax, bl + np.array([0.0, -0.62]), br + np.array([0.0, -0.62]), f"excitation   {n_ex}", outward=-0.50)
    ax.annotate(
        "",
        xy=tl + np.array([-0.72, 0.0]),
        xytext=bl + np.array([-0.72, 0.0]),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.45, shrinkA=0, shrinkB=0),
        zorder=14,
    )
    ax.text(
        *( (bl + tl) / 2 + np.array([-1.28, 0.0]) ),
        f"emission   {n_em}",
        ha="center",
        va="center",
        fontsize=13.5,
        color=INK,
        fontweight="bold",
        rotation=90,
        zorder=15,
    )
    p0 = br + np.array([0.78, -0.28])
    p1 = brz + np.array([1.05, 0.22])
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.45, shrinkA=0, shrinkB=0),
        zorder=14,
    )
    mid = (p0 + p1) / 2
    ax.text(
        mid[0] + 0.92,
        mid[1] + 0.08,
        f"sample   {n_s}",
        ha="center",
        va="center",
        fontsize=13.5,
        color=INK,
        fontweight="bold",
        zorder=15,
        rotation=28,
    )

    ax.text(
        *( (bl + tr) / 2 + np.array([0.0, 0.12]) ),
        "front  =  one well",
        fontsize=11,
        color="white",
        ha="center",
        zorder=12,
        path_effects=[withStroke(linewidth=3.2, foreground=INK)],
    )

    ax.text(
        5.9,
        1.55,
        rf"$\mathcal{{X}}\in\mathbb{{R}}^{{{n_s}\times {n_em}\times {n_ex}}}$",
        ha="center",
        fontsize=18,
        color=ACCENT,
        fontweight="bold",
    )
    ax.text(
        5.9,
        0.95,
        r"sample $\times$ emission $\times$ excitation",
        ha="center",
        fontsize=14.5,
        color=MUTED,
    )
    ax.text(
        5.9,
        0.32,
        r"$\mathcal{X}\;\approx\;a_1\circ b_1\circ c_1\;+\;a_2\circ b_2\circ c_2\;+\;a_3\circ b_3\circ c_3$",
        ha="center",
        fontsize=13.5,
        color=INK,
    )


def _draw_dyes(fig, gs, true, dye_eems) -> None:
    inner = gs.subgridspec(3, 3, height_ratios=[0.52, 1.45, 0.95], hspace=0.38, wspace=0.32)
    ax_tag = fig.add_subplot(inner[0, :])
    ax_tag.set_xlim(0, 1)
    ax_tag.set_ylim(0, 1)
    ax_tag.axis("off")
    ax_tag.set_facecolor("white")
    _header(ax_tag, "C", "Each dye is one outer product — a map times an amount", y=0.42, va="center")

    em = true[1] / (np.max(np.abs(true[1]), axis=0, keepdims=True) + 1e-12)
    ex = true[2] / (np.max(np.abs(true[2]), axis=0, keepdims=True) + 1e-12)
    x_em = np.linspace(0, 1, em.shape[0])
    x_ex = np.linspace(0, 1, ex.shape[0])

    for r, color in enumerate(DYE):
        ax = fig.add_subplot(inner[1, r])
        z = np.flipud(_hi(dye_eems[r], 12))
        ax.imshow(z, origin="upper", aspect="auto", cmap=_cmap(color), interpolation="bicubic")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"excitation $\rightarrow$", fontsize=11)
        if r == 0:
            ax.set_ylabel(r"emission $\rightarrow$", fontsize=11)
        ax.set_title(NAMES[r], fontsize=13, color=color, pad=8, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(RULE)

        ax2 = fig.add_subplot(inner[2, r])
        ax2.plot(x_ex, ex[:, r], color=color, lw=2.3, label="excitation")
        ax2.plot(x_em, em[:, r], color=color, lw=1.7, ls="--", alpha=0.9, label="emission")
        ax2.fill_between(x_ex, ex[:, r], color=color, alpha=0.12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.12)
        ax2.set_xlabel(r"colour $\rightarrow$", fontsize=11)
        ax2.set_yticks([])
        ax2.spines["left"].set_visible(False)
        if r == 2:
            ax2.legend(frameon=False, fontsize=10, loc="upper left")


def _draw_amounts(ax, true) -> None:
    amounts = true[0] / (np.max(np.abs(true[0]), axis=0, keepdims=True) + 1e-12)
    x = np.arange(amounts.shape[0])
    _header(ax, "D", "Amounts overlap — most wells are mixes")
    for r, color in enumerate(DYE):
        ax.plot(x, amounts[:, r], color=color, lw=2.6, label=NAMES[r], zorder=3)
        ax.fill_between(x, amounts[:, r], color=color, alpha=0.08, zorder=2)
    ax.set_xlim(-0.4, 19.4)
    ax.set_ylim(-0.04, 1.42)
    ax.set_xlabel("well (sample)", fontsize=12)
    ax.set_ylabel("amount", fontsize=12)
    ax.yaxis.grid(True, color="#EEF0F4", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=11, loc="upper right")
    ax.tick_params(axis="x", length=0, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)


def draw_poster(fig: plt.Figure | None = None) -> plt.Figure:
    _style()
    cube, true, dye_eems = _load()
    size = (16.2, 12.4)
    if fig is None:
        fig = plt.figure(figsize=size, facecolor="white", dpi=240)
    else:
        fig.set_size_inches(*size)
        fig.set_facecolor("white")
    fig.clf()
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.85, 1.55, 1.05],
        width_ratios=[1.20, 1.22],
        hspace=0.48,
        wspace=0.18,
        left=0.042,
        right=0.975,
        top=0.915,
        bottom=0.052,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_assay(ax_a, true, cube)
    _draw_cube(ax_b, cube)

    _draw_dyes(fig, gs[1, :], true, dye_eems)

    ax_d = fig.add_subplot(gs[2, :])
    _draw_amounts(ax_d, true)
    return fig


def write_png(path: Path | None = None) -> Path:
    dest = Path(path) if path is not None else OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig = draw_poster()
    fig.savefig(dest, dpi=280, facecolor="white", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return dest


def main() -> None:
    path = write_png(OUT)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)  {__import__('PIL').Image.open(path).size}")


if __name__ == "__main__":
    main()
