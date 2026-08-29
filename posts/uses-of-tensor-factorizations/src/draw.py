"""Drawing kit for the unmixing figures.

Flat-ink illustrations — lamp, monochromator, well, detector — plus the palette
and card furniture shared by ``make_mixing_poster.py`` and
``make_mixing_gif.py``, so the still and the animation cannot drift apart.

Every glyph draws into an axes whose data units are inches, centred on
``(x, y)`` and scaled by ``s``. Nothing here touches the data; the numbers all
come from ``tensors.py``.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Wedge,
)
from matplotlib.path import Path

INK = "#1B2433"
MUTED = "#5C6570"
RULE = "#DCE0E8"
PAGE = "#F7F8FB"
CARD = "#FFFFFF"
SHADOW = "#E1E5EC"
ACCENT = "#4A3AA7"
TEAL = "#2A9D8F"
CORAL = "#E07A5F"
DYE = (ACCENT, TEAL, CORAL)
NAMES = ("dye A", "dye B", "dye C")
GLASS = "#E8EEF6"
STEEL = "#C3CDDB"
AMBER = "#F4B942"
GREEN = "#2F855A"
RED = "#C1483F"
# Excitation sweep, short to long wavelength. Drawn, not measured.
SPECTRUM = ("#6C3DD1", "#3D6BD1", "#2AA1C0", "#2A9D8F", "#8FBE3A", "#F4B942", "#E07A5F")

LW = 1.6
FONT = ["DejaVu Sans"]

# Hex string, or the RGB triple wash()/blend()/saturate() return.
Color = Union[str, tuple[float, float, float]]


# --- colour ----------------------------------------------------------------


def wash(color: Color, amount: float) -> tuple[float, float, float]:
    """Blend ``color`` toward white; ``amount=1`` is white."""
    r, g, b = to_rgb(color)
    t = float(np.clip(amount, 0.0, 1.0))
    return (r + (1.0 - r) * t, g + (1.0 - g) * t, b + (1.0 - b) * t)


def blend(colors: Sequence[Color], weights: Sequence[float] | None = None):
    """Weighted mean of colours — what a mixture of dyes looks like."""
    w = np.ones(len(colors)) if weights is None else np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    rgb = np.array([to_rgb(c) for c in colors])
    return tuple(float(v) for v in (w[:, None] * rgb).sum(axis=0))


def saturate(color: Color, amount: float = 0.35):
    """Push a colour away from its own grey — mixed dyes read as mud otherwise."""
    rgb = np.array(to_rgb(color), dtype=float)
    grey = float(rgb.mean())
    return tuple(
        float(v) for v in np.clip(grey + (rgb - grey) * (1.0 + amount * 4.0), 0, 1)
    )


def _ink(lw: float = LW, z: int = 10) -> dict:
    return dict(
        edgecolor=INK,
        linewidth=lw,
        joinstyle="round",
        capstyle="round",
        zorder=z,
    )


# --- furniture -------------------------------------------------------------


def card(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fc: str = CARD,
    ec: str = RULE,
    lw: float = 1.1,
    z: int = 1,
    rs: float = 0.16,
    shadow: bool = True,
) -> None:
    """Draw a rounded panel with a soft offset shadow."""
    if shadow:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.06, y - 0.07),
                w,
                h,
                boxstyle=f"round,pad=0.01,rounding_size={rs}",
                facecolor=SHADOW,
                edgecolor="none",
                zorder=z - 1,
            ),
        )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.01,rounding_size={rs}",
            facecolor=fc,
            edgecolor=ec,
            lw=lw,
            zorder=z,
        ),
    )


def pill(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    fc="#EDEAF8",
    tc: str = ACCENT,
    fs: float = 11.0,
    pad: float = 0.34,
    h: float = 0.46,
    z: int = 12,
    weight: str = "bold",
) -> None:
    """Small rounded tag with centred text, sized from the string."""
    w = pad * 2 + fs / 72.0 * 0.56 * len(text)
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0.01,rounding_size={h / 2}",
            facecolor=fc,
            edgecolor="none",
            zorder=z,
        ),
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=tc,
        fontweight=weight,
        zorder=z + 1,
    )


def badge(
    ax: Axes,
    x: float,
    y: float,
    label: str,
    color: Color = ACCENT,
    r: float = 0.23,
    fs: float = 13.0,
    z: int = 12,
) -> None:
    """Numbered (or lettered) ring, as down the left of a conference poster."""
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=z))
    ax.text(
        x,
        y - 0.012,
        label,
        ha="center",
        va="center",
        fontsize=fs,
        color="white",
        fontweight="bold",
        zorder=z + 1,
    )


def glow(
    ax: Axes,
    x: float,
    y: float,
    r: float,
    color: Color,
    layers: int = 14,
    alpha: float = 0.13,
    z: int = 4,
) -> None:
    """Soft halo from stacked translucent circles."""
    for i in range(layers, 0, -1):
        frac = i / layers
        ax.add_patch(
            Circle(
                (x, y),
                r * frac,
                facecolor=color,
                edgecolor="none",
                alpha=alpha * (1.0 - frac) ** 1.3,
                zorder=z,
            ),
        )


def arrow(
    ax: Axes,
    p0,
    p1,
    color: Color = INK,
    lw: float = 1.5,
    rad: float = 0.0,
    z: int = 14,
    ls: str = "-",
    head: float = 9.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=f"-|>,head_length={head * 0.55},head_width={head * 0.30}",
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            lw=lw,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
            mutation_scale=1.0,
        ),
    )


# --- light -----------------------------------------------------------------


def beam(
    ax: Axes,
    p0,
    p1,
    w0: float,
    w1: float,
    color: Color,
    alpha: float = 0.5,
    z: int = 6,
) -> None:
    """Tapered light path from ``p0`` to ``p1``."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    d = p1 - p0
    n = np.array([-d[1], d[0]])
    n = n / (np.linalg.norm(n) + 1e-12)
    quad = [p0 + n * w0 / 2, p1 + n * w1 / 2, p1 - n * w1 / 2, p0 - n * w0 / 2]
    ax.add_patch(
        Polygon(
            quad,
            closed=True,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=z,
        ),
    )


def fan(
    ax: Axes,
    apex,
    x_end: float,
    half: float,
    colors: Sequence[Color] = SPECTRUM,
    alpha: float = 0.85,
    z: int = 6,
) -> None:
    """Spread white light into a colour fan — one monochromator, drawn."""
    apex = np.asarray(apex, dtype=float)
    edges = np.linspace(apex[1] - half, apex[1] + half, len(colors) + 1)
    for i, color in enumerate(colors):
        ax.add_patch(
            Polygon(
                [apex, (x_end, edges[i]), (x_end, edges[i + 1])],
                closed=True,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=z,
            ),
        )


# --- instrument ------------------------------------------------------------


def lamp(
    ax: Axes,
    x: float,
    y: float,
    s: float = 1.0,
    z: int = 10,
    soft: bool = True,
) -> None:
    """Xenon lamp in its housing, mouth facing right."""
    if soft:
        glow(ax, x + 0.04 * s, y, 0.66 * s, AMBER, alpha=0.22, z=z - 3)
    ax.add_patch(
        Wedge(
            (x - 0.02 * s, y),
            0.46 * s,
            100,
            260,
            width=0.11 * s,
            facecolor=STEEL,
            **_ink(1.2, z),
        ),
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.52 * s, y - 0.50 * s),
            1.0 * s,
            1.0 * s,
            boxstyle=f"round,pad=0.01,rounding_size={0.14 * s}",
            facecolor="none",
            **_ink(1.4, z - 1),
        ),
    )
    ax.add_patch(
        Circle((x, y), 0.27 * s, facecolor=wash(AMBER, 0.35), **_ink(1.5, z + 1)),
    )
    for ang in range(0, 360, 45):
        rad = np.deg2rad(ang)
        v = np.array([np.cos(rad), np.sin(rad)])
        ax.add_patch(
            PathPatch(
                Path([tuple([x, y] + v * 0.07 * s), tuple([x, y] + v * 0.17 * s)]),
                facecolor="none",
                edgecolor=INK,
                lw=1.3,
                capstyle="round",
                zorder=z + 2,
            ),
        )


def prism(ax: Axes, x: float, y: float, s: float = 1.0, z: int = 10) -> None:
    """Dispersing prism, apex up."""
    tri = [
        (x - 0.50 * s, y - 0.40 * s),
        (x + 0.50 * s, y - 0.40 * s),
        (x, y + 0.50 * s),
    ]
    ax.add_patch(Polygon(tri, closed=True, facecolor=wash(GLASS, 0.25), **_ink(1.6, z)))
    ax.add_patch(
        Polygon(
            [
                (x, y + 0.50 * s),
                (x - 0.25 * s, y + 0.05 * s),
                (x - 0.02 * s, y - 0.40 * s),
                (x - 0.50 * s, y - 0.40 * s),
            ],
            closed=True,
            facecolor="white",
            edgecolor="none",
            alpha=0.55,
            zorder=z + 1,
        ),
    )


def slit(
    ax: Axes,
    x: float,
    y: float,
    s: float = 1.0,
    gap: float = 0.16,
    z: int = 10,
) -> None:
    """Plate with a slit: one colour out of the fan gets through."""
    for sign in (1.0, -1.0):
        ax.add_patch(
            FancyBboxPatch(
                (
                    (x - 0.07 * s, y + sign * (gap * s / 2))
                    if sign > 0
                    else (x - 0.07 * s, y - 0.52 * s)
                ),
                0.14 * s,
                0.52 * s - gap * s / 2,
                boxstyle=f"round,pad=0.005,rounding_size={0.03 * s}",
                facecolor=STEEL,
                **_ink(1.3, z),
            ),
        )


def well(
    ax: Axes,
    x: float,
    y: float,
    s: float = 1.0,
    weights: Sequence[float] = (1.0, 1.0, 1.0),
    z: int = 10,
    soft: bool = True,
    dots: bool = True,
) -> None:
    """One well of mixed dye: glass vessel, blended liquid, three dye specks."""
    mix = saturate(blend(DYE, weights), 0.60)
    if soft:
        glow(ax, x, y - 0.06 * s, 0.80 * s, mix, alpha=0.16, z=z - 3)
    body = FancyBboxPatch(
        (x - 0.28 * s, y - 0.46 * s),
        0.56 * s,
        0.92 * s,
        boxstyle=f"round,pad=0.005,rounding_size={0.09 * s}",
        facecolor=wash(GLASS, 0.4),
        **_ink(1.6, z),
    )
    ax.add_patch(body)
    liquid = Polygon(
        [
            (x - 0.26 * s, y - 0.42 * s),
            (x + 0.26 * s, y - 0.42 * s),
            (x + 0.26 * s, y + 0.16 * s),
            (x - 0.26 * s, y + 0.16 * s),
        ],
        closed=True,
        facecolor=mix,
        edgecolor="none",
        alpha=0.72,
        zorder=z + 1,
    )
    ax.add_patch(liquid)
    liquid.set_clip_path(body)
    ax.add_patch(
        Ellipse(
            (x, y + 0.16 * s),
            0.52 * s,
            0.10 * s,
            facecolor="white",
            alpha=0.5,
            edgecolor="none",
            zorder=z + 2,
        ),
    )
    if dots:
        spots = ((-0.12, -0.20), (0.10, -0.05), (-0.02, -0.30))
        for (dx, dy), color in zip(spots, DYE):
            ax.add_patch(
                Circle(
                    (x + dx * s, y + dy * s),
                    0.052 * s,
                    facecolor=color,
                    edgecolor="white",
                    lw=0.8,
                    zorder=z + 3,
                ),
            )


def detector(
    ax: Axes,
    x: float,
    y: float,
    s: float = 1.0,
    reading: str = "",
    z: int = 10,
) -> None:
    """Photomultiplier: lens, body, and a readout showing one number."""
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.40 * s, y - 0.40 * s),
            0.88 * s,
            0.80 * s,
            boxstyle=f"round,pad=0.01,rounding_size={0.10 * s}",
            facecolor=wash(STEEL, 0.35),
            **_ink(1.6, z),
        ),
    )
    ax.add_patch(
        Polygon(
            [
                (x - 0.40 * s, y - 0.26 * s),
                (x - 0.62 * s, y - 0.34 * s),
                (x - 0.62 * s, y + 0.34 * s),
                (x - 0.40 * s, y + 0.26 * s),
            ],
            closed=True,
            facecolor=wash(GLASS, 0.2),
            **_ink(1.5, z),
        ),
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.26 * s, y - 0.22 * s),
            0.60 * s,
            0.44 * s,
            boxstyle=f"round,pad=0.005,rounding_size={0.05 * s}",
            facecolor=INK,
            edgecolor="none",
            zorder=z + 1,
        ),
    )
    if reading:
        ax.text(
            x + 0.04 * s,
            y,
            reading,
            ha="center",
            va="center",
            fontsize=11.5 * s,
            color=AMBER,
            fontweight="bold",
            zorder=z + 2,
        )


def droplet(ax: Axes, x: float, y: float, s: float, color: str, z: int = 10) -> None:
    """Bezier teardrop."""
    pts = np.array(
        [
            [0.00, 0.62],
            [0.22, 0.22],
            [0.40, 0.02],
            [0.40, -0.14],
            [0.40, -0.44],
            [0.18, -0.60],
            [0.00, -0.60],
            [-0.18, -0.60],
            [-0.40, -0.44],
            [-0.40, -0.14],
            [-0.40, 0.02],
            [-0.22, 0.22],
            [0.00, 0.62],
        ],
    )
    codes = [Path.MOVETO] + [Path.CURVE4] * 12
    ax.add_patch(
        PathPatch(
            Path(pts * s + [x, y], codes),
            facecolor=wash(color, 0.15),
            **_ink(1.5, z),
        ),
    )
    ax.add_patch(
        Ellipse(
            (x - 0.14 * s, y + 0.02 * s),
            0.16 * s,
            0.26 * s,
            angle=22.0,
            facecolor="white",
            alpha=0.42,
            edgecolor="none",
            zorder=z + 1,
        ),
    )


def plate(
    ax: Axes,
    x: float,
    y: float,
    s: float,
    weights: np.ndarray,
    ring: int | None = None,
    ncols: int = 5,
    nrows: int = 4,
    z: int = 10,
) -> None:
    """Microplate seen from above; each well tinted by its dye mixture."""
    w, h = 1.55 * s, 1.20 * s
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0.01,rounding_size={0.10 * s}",
            facecolor="white",
            **_ink(1.5, z),
        ),
    )
    r = 0.115 * s
    xs = np.linspace(x - w / 2 + 0.22 * s, x + w / 2 - 0.22 * s, ncols)
    ys = np.linspace(y + h / 2 - 0.24 * s, y - h / 2 + 0.24 * s, nrows)
    for i in range(nrows * ncols):
        cx, cy = xs[i % ncols], ys[i // ncols]
        wt = weights[i] if i < len(weights) else weights[-1]
        ax.add_patch(
            Circle(
                (cx, cy),
                r,
                facecolor=blend(DYE, wt),
                alpha=0.85,
                edgecolor=RULE,
                lw=0.7,
                zorder=z + 1,
            ),
        )
        if ring is not None and i == ring:
            ax.add_patch(
                Circle(
                    (cx, cy),
                    r * 1.62,
                    facecolor="none",
                    edgecolor=INK,
                    lw=1.6,
                    zorder=z + 3,
                ),
            )
    return None


def sweep_grid(
    ax: Axes,
    x: float,
    y: float,
    s: float,
    ncols: int = 9,
    nrows: int = 7,
    lit: tuple[int, int] = (4, 3),
    z: int = 10,
) -> None:
    """Excitation × emission grid with one pair lit — one reading."""
    w, h = 1.30 * s, 1.05 * s
    x0, y0 = x - w / 2, y - h / 2
    cw, ch = w / ncols, h / nrows
    for i in range(ncols):
        for j in range(nrows):
            is_lit = (i, j) == lit
            ax.add_patch(
                FancyBboxPatch(
                    (x0 + i * cw + 0.012 * s, y0 + j * ch + 0.012 * s),
                    cw - 0.024 * s,
                    ch - 0.024 * s,
                    boxstyle="round,pad=0.002,rounding_size=0.012",
                    facecolor=AMBER if is_lit else wash(ACCENT, 0.92),
                    edgecolor="none",
                    zorder=z + (2 if is_lit else 1),
                ),
            )
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle=f"round,pad=0.01,rounding_size={0.05 * s}",
            facecolor="none",
            **_ink(1.4, z + 3),
        ),
    )


def flatten_icon(ax: Axes, x: float, y: float, s: float, z: int = 10) -> None:
    """Draw a cube squashed into one long row — what unfolding does."""
    dx, dy = 0.20 * s, 0.12 * s
    w, h = 0.52 * s, 0.52 * s
    bl = np.array([x - 1.02 * s, y - h / 2])
    front = [bl, bl + [w, 0], bl + [w, h], bl + [0, h]]
    ax.add_patch(
        Polygon(front, closed=True, facecolor=wash(ACCENT, 0.86), **_ink(1.4, z + 1)),
    )
    ax.add_patch(
        Polygon(
            [front[3], front[3] + [dx, dy], front[2] + [dx, dy], front[2]],
            closed=True,
            facecolor=wash(ACCENT, 0.72),
            **_ink(1.4, z),
        ),
    )
    ax.add_patch(
        Polygon(
            [front[1], front[1] + [dx, dy], front[2] + [dx, dy], front[2]],
            closed=True,
            facecolor=wash(ACCENT, 0.78),
            **_ink(1.4, z),
        ),
    )
    arrow(ax, (x - 0.16 * s, y), (x + 0.16 * s, y), color=MUTED, lw=1.4, z=z + 2)
    n = 14
    sw = 1.02 * s / n
    for i in range(n):
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.26 * s + i * sw, y - 0.09 * s),
                sw * 0.86,
                0.18 * s,
                boxstyle="round,pad=0.002,rounding_size=0.01",
                facecolor=wash(ACCENT, 0.55 + 0.03 * (i % 3)),
                edgecolor="none",
                zorder=z + 1,
            ),
        )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.26 * s, y - 0.09 * s),
            1.02 * s,
            0.18 * s,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            facecolor="none",
            **_ink(1.3, z + 2),
        ),
    )


def rank_one_chip(
    ax: Axes,
    x: float,
    y: float,
    s: float,
    color: Color,
    z: int = 10,
) -> None:
    """Column ∘ row = map — one rank-1 tensor, drawn."""
    ax.add_patch(
        FancyBboxPatch(
            (x - 1.00 * s, y - 0.26 * s),
            0.15 * s,
            0.52 * s,
            boxstyle="round,pad=0.004,rounding_size=0.03",
            facecolor=wash(color, 0.45),
            **_ink(1.2, z),
        ),
    )
    ax.text(
        x - 0.70 * s,
        y,
        "∘",
        ha="center",
        va="center",
        fontsize=15 * s,
        color=MUTED,
        zorder=z + 1,
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.56 * s, y - 0.075 * s),
            0.52 * s,
            0.15 * s,
            boxstyle="round,pad=0.004,rounding_size=0.03",
            facecolor=wash(color, 0.62),
            **_ink(1.2, z),
        ),
    )
    ax.text(
        x + 0.10 * s,
        y,
        "=",
        ha="center",
        va="center",
        fontsize=15 * s,
        color=MUTED,
        zorder=z + 1,
    )
    n = 6
    step = 0.52 * s / n
    for i in range(n):
        for j in range(n):
            ax.add_patch(
                Polygon(
                    [
                        (x + 0.32 * s + i * step, y - 0.26 * s + j * step),
                        (x + 0.32 * s + (i + 1) * step, y - 0.26 * s + j * step),
                        (x + 0.32 * s + (i + 1) * step, y - 0.26 * s + (j + 1) * step),
                        (x + 0.32 * s + i * step, y - 0.26 * s + (j + 1) * step),
                    ],
                    closed=True,
                    facecolor=wash(color, 0.30 + 0.62 * (1 - (i / n) * (j / n))),
                    edgecolor="none",
                    zorder=z,
                ),
            )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.32 * s, y - 0.26 * s),
            0.52 * s,
            0.52 * s,
            boxstyle="round,pad=0.004,rounding_size=0.03",
            facecolor="none",
            **_ink(1.2, z + 1),
        ),
    )


def tick(
    ax: Axes,
    x: float,
    y: float,
    s: float,
    color: str = GREEN,
    z: int = 12,
) -> None:
    ax.add_patch(Circle((x, y), 0.20 * s, facecolor=color, edgecolor="none", zorder=z))
    ax.add_patch(
        PathPatch(
            Path(
                np.array([[-0.10, 0.01], [-0.03, -0.08], [0.11, 0.09]]) * s + [x, y],
                [Path.MOVETO, Path.LINETO, Path.LINETO],
            ),
            facecolor="none",
            edgecolor="white",
            lw=2.0,
            capstyle="round",
            joinstyle="round",
            zorder=z + 1,
        ),
    )


def cross(
    ax: Axes,
    x: float,
    y: float,
    s: float,
    color: str = RED,
    z: int = 12,
) -> None:
    ax.add_patch(Circle((x, y), 0.20 * s, facecolor=color, edgecolor="none", zorder=z))
    for a, b in (((-0.08, -0.08), (0.08, 0.08)), ((-0.08, 0.08), (0.08, -0.08))):
        ax.add_patch(
            PathPatch(
                Path(np.array([a, b]) * s + [x, y]),
                facecolor="none",
                edgecolor="white",
                lw=2.0,
                capstyle="round",
                zorder=z + 1,
            ),
        )


def spectrum_bar(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    colors: Sequence[Color] = SPECTRUM,
    z: int = 10,
    lit: int | None = None,
) -> None:
    """Horizontal swatch of the swept colours; ``lit`` ringed as the pick."""
    n = len(colors)
    step = w / n
    for i, color in enumerate(colors):
        ax.add_patch(
            Polygon(
                [
                    (x + i * step, y),
                    (x + (i + 1) * step, y),
                    (x + (i + 1) * step, y + h),
                    (x + i * step, y + h),
                ],
                closed=True,
                facecolor=color,
                edgecolor="none",
                zorder=z,
            ),
        )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={min(h / 2, 0.05)}",
            facecolor="none",
            **_ink(1.2, z + 1),
        ),
    )
    if lit is not None:
        ax.add_patch(
            FancyBboxPatch(
                (x + lit * step, y - 0.03),
                step,
                h + 0.06,
                boxstyle="round,pad=0.004,rounding_size=0.02",
                facecolor="none",
                **_ink(1.8, z + 2),
            ),
        )


def _rot(pts, deg: float, about):
    """Rotate points about a centre, for glyph parts drawn off-axis."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    about = np.asarray(about, dtype=float)
    return (np.asarray(pts, dtype=float) - about) @ np.array([[c, s], [-s, c]]) + about


SKIN = "#E3B18B"
HAIR = "#2F2A33"
COAT_SHADE = "#EDF1F7"


def _path(ax: Axes, pts, codes, z: int, **kw) -> None:
    ax.add_patch(PathPatch(Path(np.asarray(pts, dtype=float), codes), zorder=z, **kw))


def _limb(ax: Axes, pts, width: float, z: int, fill: str = "white") -> None:
    """Draw a sleeve: an ink stroke with a lighter one laid over it."""
    path = Path(
        np.asarray(pts, dtype=float), [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=INK,
            lw=width * 1.30,
            capstyle="round",
            joinstyle="round",
            zorder=z,
        ),
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=fill,
            lw=width,
            capstyle="round",
            joinstyle="round",
            zorder=z + 1,
        ),
    )


def scientist(
    ax: Axes,
    x: float,
    y: float,
    s: float = 1.0,
    color: Color = TEAL,
    z: int = 10,
) -> None:
    """Draw someone at the bench: lab coat, goggles, pipette in a gloved hand.

    Stylised rather than a portrait — the poster needs a person doing the work,
    not a particular one.
    """
    o = np.array([x, y], dtype=float)

    def P(pts):
        return np.asarray(pts, dtype=float) * s + o

    scrub = wash(color, 0.42)

    # far arm, behind the coat
    _limb(ax, P([[-0.20, 0.26], [-0.42, 0.02], [-0.45, -0.22]]), 0.115 * s * 72, z - 2)
    ax.add_patch(
        Circle(
            tuple(P([[-0.455, -0.27]])[0]),
            0.075 * s,
            facecolor=scrub,
            **_ink(1.3, z - 1),
        ),
    )

    _limb(ax, P([[0.22, 0.28], [0.46, 0.10], [0.60, -0.10]]), 0.125 * s * 72, z - 2)

    # coat
    coat = P(
        [
            [-0.44, -0.76],
            [-0.37, 0.14],
            [-0.35, 0.34],
            [-0.22, 0.44],
            [-0.11, 0.49],
            [0.17, 0.49],
            [0.31, 0.44],
            [0.40, 0.30],
            [0.42, 0.14],
            [0.48, -0.76],
            [-0.44, -0.76],
        ]
    )
    codes = (
        [Path.MOVETO, Path.LINETO]
        + [Path.CURVE4] * 3
        + [Path.CURVE4] * 3
        + [Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    )
    _path(
        ax, coat, codes, z, facecolor="white", edgecolor=INK, lw=1.6, joinstyle="round"
    )
    ax.add_patch(
        Polygon(
            P(
                [
                    [-0.44, -0.76],
                    [-0.37, 0.14],
                    [-0.26, 0.36],
                    [-0.20, 0.34],
                    [-0.30, 0.12],
                    [-0.36, -0.76],
                ]
            ),
            closed=True,
            facecolor=COAT_SHADE,
            edgecolor="none",
            zorder=z + 1,
        ),
    )

    # neck, then the collar V with the scrubs showing through
    ax.add_patch(
        Polygon(
            P([[-0.08, 0.36], [0.10, 0.36], [0.10, 0.58], [-0.08, 0.58]]),
            closed=True,
            facecolor=wash(SKIN, 0.18),
            **_ink(1.3, z + 1),
        ),
    )
    ax.add_patch(
        Polygon(
            P([[-0.13, 0.49], [0.19, 0.49], [0.03, 0.16]]),
            closed=True,
            facecolor=scrub,
            **_ink(1.4, z + 3),
        ),
    )
    for a, b in (((-0.13, 0.49), (0.03, 0.16)), ((0.19, 0.49), (0.03, 0.16))):
        _path(
            ax,
            P([a, b]),
            [Path.MOVETO, Path.LINETO],
            z + 4,
            facecolor="none",
            edgecolor=INK,
            lw=1.4,
            capstyle="round",
        )
    _path(
        ax,
        P([[0.05, 0.14], [0.07, -0.58]]),
        [Path.MOVETO, Path.LINETO],
        z + 3,
        facecolor="none",
        edgecolor=RULE,
        lw=1.3,
    )
    for by in (-0.08, -0.32):
        ax.add_patch(
            Circle(
                tuple(P([[0.06, by]])[0]),
                0.022 * s,
                facecolor=RULE,
                edgecolor="none",
                zorder=z + 4,
            ),
        )
    ax.add_patch(
        Polygon(
            P([[0.18, -0.44], [0.36, -0.44], [0.36, -0.24], [0.18, -0.24]]),
            closed=True,
            facecolor="none",
            edgecolor=RULE,
            lw=1.3,
            zorder=z + 3,
        ),
    )
    _path(
        ax,
        P([[0.30, -0.36], [0.30, -0.16]]),
        [Path.MOVETO, Path.LINETO],
        z + 3,
        facecolor="none",
        edgecolor=CORAL,
        lw=2.0,
        capstyle="round",
    )

    # head
    ax.add_patch(
        Ellipse(
            tuple(P([[0.03, 0.74]])[0]),
            0.40 * s,
            0.46 * s,
            facecolor=SKIN,
            **_ink(1.5, z + 4),
        ),
    )
    ax.add_patch(
        Circle(
            tuple(P([[-0.175, 0.68]])[0]), 0.058 * s, facecolor=SKIN, **_ink(1.2, z + 4)
        ),
    )
    hair = P(
        [
            [-0.19, 0.72],
            [-0.24, 0.99],
            [-0.05, 1.06],
            [0.07, 1.02],
            [0.19, 0.98],
            [0.24, 0.90],
            [0.23, 0.78],
            [0.17, 0.86],
            [0.06, 0.89],
            [-0.03, 0.86],
            [-0.10, 0.84],
            [-0.15, 0.80],
            [-0.19, 0.72],
        ]
    )
    _path(
        ax,
        hair,
        [Path.MOVETO] + [Path.CURVE4] * 12,
        z + 5,
        facecolor=HAIR,
        edgecolor=INK,
        lw=1.3,
        joinstyle="round",
    )
    for ex in (-0.03, 0.13):
        ax.add_patch(
            Ellipse(
                tuple(P([[ex, 0.755]])[0]),
                0.045 * s,
                0.055 * s,
                facecolor=INK,
                edgecolor="none",
                zorder=z + 5,
            ),
        )
    goggles = FancyBboxPatch(
        tuple(P([[-0.17, 0.70]])[0]),
        0.38 * s,
        0.105 * s,
        boxstyle=f"round,pad=0.002,rounding_size={0.045 * s}",
        facecolor=wash(GLASS, 0.05),
        alpha=0.62,
        **_ink(1.4, z + 6),
    )
    ax.add_patch(goggles)
    _path(
        ax,
        P([[0.04, 0.70], [0.04, 0.805]]),
        [Path.MOVETO, Path.LINETO],
        z + 7,
        facecolor="none",
        edgecolor=INK,
        lw=1.0,
    )
    _path(
        ax,
        P([[-0.19, 0.755], [-0.23, 0.73]]),
        [Path.MOVETO, Path.LINETO],
        z + 5,
        facecolor="none",
        edgecolor=INK,
        lw=1.2,
        capstyle="round",
    )
    # a small smile, drawn as an arc so the face is not a blank
    smile = P([[0.00, 0.615], [0.05, 0.585], [0.11, 0.605]])
    _path(
        ax,
        smile,
        [Path.MOVETO, Path.CURVE3, Path.CURVE3],
        z + 7,
        facecolor="none",
        edgecolor=wash(INK, 0.25),
        lw=1.4,
        capstyle="round",
    )

    # near arm, over the coat, with the pipette
    _path(
        ax,
        P([[0.545, -0.02], [0.655, -0.06]]),
        [Path.MOVETO, Path.LINETO],
        z + 8,
        facecolor="none",
        edgecolor=INK,
        lw=1.2,
    )
    hand = P([[0.62, -0.16]])[0]
    ax.add_patch(Circle(tuple(hand), 0.078 * s, facecolor=scrub, **_ink(1.3, z + 9)))
    body = (
        np.array(
            [
                [-0.055, 0.20],
                [0.055, 0.20],
                [0.055, -0.16],
                [0.0, -0.30],
                [-0.055, -0.16],
            ]
        )
        * s
        + hand
    )
    ax.add_patch(
        Polygon(
            _rot(body, 34.0, hand),
            closed=True,
            facecolor=wash(STEEL, 0.30),
            **_ink(1.4, z + 8),
        ),
    )
    ax.add_patch(
        Polygon(
            _rot(
                np.array(
                    [
                        [-0.055, 0.20],
                        [0.055, 0.20],
                        [0.055, 0.10],
                        [-0.055, 0.10],
                    ]
                )
                * s
                + hand,
                34.0,
                hand,
            ),
            closed=True,
            facecolor=color,
            edgecolor="none",
            zorder=z + 9,
        ),
    )
    tip = _rot(np.array([[0.0, -0.40]]) * s + hand, 34.0, hand)[0]
    ax.add_patch(
        Circle(tuple(tip), 0.05 * s, facecolor=TEAL, edgecolor="none", zorder=z + 8)
    )


def settings_panel(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    rows: Sequence[tuple[str, str, object]],
    header: str = "",
    fs: float = 12.0,
    z: int = 6,
) -> None:
    """Draw the console: what the operator sets before a reading is taken."""
    card(ax, x, y, w, h, fc="#F4F6FC", ec=RULE, z=z, rs=0.10)
    top = y + h
    if header:
        ax.text(
            x + 0.22,
            top - 0.30,
            header,
            fontsize=fs * 0.86,
            color=MUTED,
            fontweight="bold",
            va="center",
            zorder=z + 2,
        )
        top -= 0.52
    step = (top - y - 0.16) / max(len(rows), 1)
    for i, (label, value, chip) in enumerate(rows):
        cy = top - (i + 0.5) * step
        if i:
            ax.add_patch(
                Polygon(
                    [(x + 0.18, cy + step / 2), (x + w - 0.18, cy + step / 2)],
                    closed=False,
                    facecolor="none",
                    edgecolor=RULE,
                    lw=1.0,
                    zorder=z + 1,
                ),
            )
        cx = x + 0.24
        if chip is not None:
            ax.add_patch(
                FancyBboxPatch(
                    (cx, cy - 0.10),
                    0.20,
                    0.20,
                    boxstyle="round,pad=0.005,rounding_size=0.04",
                    facecolor=chip,
                    **_ink(1.0, z + 2),
                ),
            )
            cx += 0.34
        ax.text(cx, cy, label, fontsize=fs, va="center", zorder=z + 2)
        ax.text(
            x + w - 0.24,
            cy,
            value,
            fontsize=fs,
            va="center",
            ha="right",
            fontweight="bold",
            color=ACCENT,
            zorder=z + 2,
        )
