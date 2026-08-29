r"""Drawn poster for the unmixing section: assay, tensor, and the two splits.

Everything but the heat maps and line plots is drawn — lamp, monochromators,
well, detector — from the glyph kit in ``draw.py``. Every number on the poster
comes from ``tensors.mixing_scene``, so it cannot disagree with the prose.

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
from matplotlib.patches import Polygon
from matplotlib.patheffects import withStroke
from matplotlib.transforms import Affine2D
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw as D
import tensors as T

POST = Path(__file__).resolve().parent.parent
OUT = POST / "media" / "mixing-assay.png"

W_FIG, H_FIG = 24.0, 15.0
DPI = 200

# Type scale, in points on a 24-inch canvas.
TITLE, SUB = 40.0, 22.0
CARD_T, LEAD = 22.0, 18.0
BODY, SMALL, TINY = 15.5, 14.0, 12.5

WELL_SHOWN = 10  # the well drawn in the instrument and mapped in the cube


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": D.FONT,
            "text.color": D.INK,
            "axes.labelcolor": D.INK,
            "xtick.color": D.MUTED,
            "ytick.color": D.MUTED,
            "font.size": BODY,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def _cmap(hex_color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "dye",
        ["#FFFFFF", hex_color + "22", hex_color],
    )


DIVERGING = LinearSegmentedColormap.from_list(
    "signed",
    [D.CORAL, "#FFFFFF", D.ACCENT],
)


def _to_rgb(z: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    z = np.abs(np.asarray(z, dtype=float))
    z = z / (float(z.max()) + 1e-12)
    rgba = plt.colormaps[cmap](z)
    return (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)


def _hi(z: np.ndarray, factor: float = 10.0) -> np.ndarray:
    return zoom(np.asarray(z, dtype=float), factor, order=3)


def _ax(fig: plt.Figure, x: float, y: float, w: float, h: float):
    """Axes placed in poster inches rather than figure fractions."""
    return fig.add_axes([x / W_FIG, y / H_FIG, w / W_FIG, h / H_FIG])


def _map_axes(
    ax,
    z: np.ndarray,
    cmap,
    title: str = "",
    title_color: str = D.INK,
    xlabel: str = "",
    ylabel: str = "",
    signed: bool = False,
) -> None:
    hi = _hi(z, 8)
    kw: dict = {}
    if signed:
        peak = float(np.max(np.abs(hi))) or 1.0
        kw = dict(vmin=-peak, vmax=peak)
    ax.imshow(
        np.flipud(hi),
        origin="upper",
        aspect="auto",
        cmap=cmap,
        interpolation="bicubic",
        **kw,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=SMALL, color=title_color, pad=4, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=TINY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=TINY)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(D.RULE)


def _amount_axes(ax, title: str) -> None:
    ax.set_xlim(-0.4, 19.4)
    ax.set_xlabel("sample (well)", fontsize=SMALL)
    ax.set_ylabel("amount", fontsize=SMALL)
    ax.yaxis.grid(True, color="#EEF0F4", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, labelsize=TINY)
    ax.tick_params(axis="y", labelsize=TINY)
    ax.spines["left"].set_color(D.RULE)
    ax.spines["bottom"].set_color(D.RULE)
    ax.set_facecolor("white")
    if title:
        ax.set_title(title, fontsize=SMALL, pad=5)


# --- the isometric cube ----------------------------------------------------


def iso_xy(x, y, z, origin, scale: float = 1.0, depth: float = 0.40):
    return np.array(
        [
            origin[0] + scale * (x + depth * z),
            origin[1] + scale * (y + 0.48 * depth * z),
        ],
    )


def imshow_quad(ax, image, bl, br, tr, tl, zorder: int = 5, lw: float = 0.7):
    image = np.clip(np.asarray(image), 0, 255).astype(np.uint8)
    bl, br, tr, tl = map(np.asarray, (bl, br, tr, tl))
    vx, vy = br - bl, tl - bl
    trans = (
        Affine2D.from_values(vx[0], vx[1], vy[0], vy[1], bl[0], bl[1]) + ax.transData
    )
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
            edgecolor=D.INK,
            lw=lw,
            zorder=zorder + 1,
        ),
    )
    return im


def _paint_cube(ax, cube: np.ndarray, origin, scale: float) -> None:
    n_s, n_em, n_ex = cube.shape
    i_em, i_ex = int(0.50 * (n_em - 1)), int(0.52 * (n_ex - 1))
    front = np.flipud(_hi(cube[WELL_SHOWN]))
    right = np.flipud(_hi(cube[:, :, i_ex].T))
    top = _hi(cube[:, i_em, :])
    W, H, Dp = 1.55, 1.70, 1.45

    def p(x, y, z):
        return iso_xy(x, y, z, np.asarray(origin), scale=scale)

    ax.add_patch(
        Polygon(
            [
                p(0.14, -0.10, 0.14),
                p(W + 0.14, -0.10, 0.14),
                p(W + 0.14, -0.10, Dp + 0.14),
                p(0.14, -0.10, Dp + 0.14),
            ],
            closed=True,
            facecolor=D.SHADOW,
            edgecolor="none",
            zorder=1,
            alpha=0.95,
        ),
    )
    tl, tr, trz, tlz = p(0, H, 0), p(W, H, 0), p(W, H, Dp), p(0, H, Dp)
    br, brz, bl = p(W, 0, 0), p(W, 0, Dp), p(0, 0, 0)
    imshow_quad(ax, _to_rgb(top), tl, tr, trz, tlz, zorder=2, lw=0.9)
    imshow_quad(ax, _to_rgb(right), br, brz, trz, tr, zorder=3, lw=0.9)
    imshow_quad(ax, _to_rgb(front), bl, br, tr, tl, zorder=6, lw=1.1)

    D.arrow(ax, bl + [0.0, -0.30], br + [0.0, -0.30], color=D.INK, lw=1.4, head=8)
    D.arrow(ax, br + [0.0, -0.30], bl + [0.0, -0.30], color=D.INK, lw=1.4, head=8)
    ax.text(
        *((bl + br) / 2 + np.array([0.0, -0.58])),
        f"excitation colour  {n_ex}",
        ha="center",
        fontsize=SMALL,
        fontweight="bold",
        zorder=15,
    )
    D.arrow(ax, bl + [-0.34, 0.0], tl + [-0.34, 0.0], color=D.INK, lw=1.4, head=8)
    D.arrow(ax, tl + [-0.34, 0.0], bl + [-0.34, 0.0], color=D.INK, lw=1.4, head=8)
    ax.text(
        *((bl + tl) / 2 + np.array([-0.52, 0.0])),
        f"emission colour  {n_em}",
        ha="center",
        va="center",
        fontsize=SMALL,
        fontweight="bold",
        rotation=90,
        zorder=15,
    )
    p0, p1 = br + np.array([0.30, -0.10]), brz + np.array([0.42, 0.12])
    D.arrow(ax, p0, p1, color=D.INK, lw=1.4, head=8)
    D.arrow(ax, p1, p0, color=D.INK, lw=1.4, head=8)
    ax.text(
        *((p0 + p1) / 2 + np.array([0.52, 0.04])),
        f"sample (well)  {n_s}",
        ha="center",
        fontsize=TINY,
        fontweight="bold",
        rotation=26,
        zorder=15,
    )
    ax.text(
        *((bl + tr) / 2 + np.array([0.0, 0.06])),
        "front = one well",
        fontsize=TINY,
        color="white",
        ha="center",
        zorder=12,
        path_effects=[withStroke(linewidth=3.0, foreground=D.INK)],
    )


# --- panels ----------------------------------------------------------------


def _header(ax) -> None:
    ax.text(
        0.55,
        14.40,
        "You want dye amounts. You measure intensity.",
        fontsize=TITLE,
        fontweight="bold",
        va="center",
        zorder=5,
    )
    ax.text(
        0.55,
        13.82,
        "Twenty wells × emission colour × excitation colour is a 3-way tensor. "
        "Dye is a factor, not an axis.",
        fontsize=SUB,
        color=D.MUTED,
        va="center",
        zorder=5,
    )
    D.pill(
        ax,
        21.75,
        14.28,
        "synthetic · seed 7 · no wet-lab data",
        fs=SMALL,
        h=0.56,
    )


def _row_reading(fig, ax, scene) -> None:
    """Row 1: samples and settings in, one observation out, stacked to a tensor."""
    x0, y0, w, h = 0.35, 9.05, 23.30, 4.50
    D.card(ax, x0, y0, w, h)
    n_s, n_em, n_ex = scene["shape"]
    i_em, i_ex = 12, 9  # the pair of colours the drawn reading is taken at
    pick_ex, pick_em = D.SPECTRUM[1], D.SPECTRUM[5]

    ax.text(
        x0 + 0.30,
        y0 + h - 0.40,
        "One reading at a time: settings in, one observation out",
        fontsize=CARD_T,
        fontweight="bold",
        va="center",
        zorder=6,
    )
    for label, text, bx in (
        ("1", "The samples", 0.75),
        ("2", "The settings", 4.90),
        ("3", "The instrument", 8.40),
        ("4", "The observation", 14.60),
        ("5", "The tensor", 17.55),
    ):
        D.badge(ax, bx, 12.80, label, r=0.20, fs=BODY)
        ax.text(
            bx + 0.30,
            12.80,
            text,
            fontsize=LEAD,
            fontweight="bold",
            va="center",
            zorder=6,
        )

    # 1 — someone makes up twenty wells of mixed dye
    for i, color in enumerate(D.DYE):
        D.droplet(ax, 2.86 + i * 0.46, 11.92, 0.36, color)
    # Plate first: the pipette then reaches over it rather than behind it.
    D.plate(ax, 3.02, 10.52, 0.70, scene["true_s"], ring=WELL_SHOWN)
    D.scientist(ax, 1.62, 11.08, 1.30, z=12)
    ax.text(
        2.60,
        9.58,
        "three dyes, unknown amounts, in 20 wells",
        ha="center",
        fontsize=SMALL,
        color=D.MUTED,
        zorder=6,
    )

    # 2 — the console: everything the operator chooses
    D.settings_panel(
        ax,
        4.85,
        10.15,
        3.20,
        2.25,
        [
            ("well", f"{WELL_SHOWN + 1} of {n_s}", None),
            ("excitation", f"{i_ex + 1} of {n_ex}", pick_ex),
            ("emission", f"{i_em + 1} of {n_em}", pick_em),
        ],
        header="inputs you set",
        fs=SMALL,
    )
    D.arrow(ax, (8.12, 11.28), (8.42, 11.28), color=D.MUTED, lw=1.8)

    # 3 — the optics, one well at a time
    top, bot = 11.85, 10.30
    D.lamp(ax, 8.85, top, 0.70)
    D.beam(
        ax,
        (9.22, top),
        (9.90, top + 0.02),
        0.09,
        0.18,
        D.wash(D.AMBER, 0.25),
        alpha=0.65,
    )
    D.prism(ax, 10.10, top, 0.70)
    D.fan(ax, (10.28, top + 0.02), 11.22, 0.29)
    D.beam(ax, (11.18, top), (12.14, top), 0.09, 0.12, pick_ex, alpha=0.9, z=8)
    D.slit(ax, 11.28, top, 0.78, gap=0.12)
    D.well(ax, 12.42, top, 0.85, weights=scene["true_s"][WELL_SHOWN])

    D.beam(ax, (12.42, top - 0.38), (12.42, bot), 0.10, 0.09, pick_em, alpha=0.85)
    D.beam(ax, (12.38, bot), (12.86, bot), 0.09, 0.12, pick_em, alpha=0.85)
    D.prism(ax, 13.15, bot, 0.66)
    D.fan(ax, (13.32, bot + 0.02), 14.06, 0.25)
    D.beam(ax, (14.02, bot), (14.58, bot), 0.08, 0.10, pick_em, alpha=0.9, z=8)
    D.slit(ax, 14.12, bot, 0.72, gap=0.12)
    D.detector(ax, 14.92, bot, 0.72)

    for x, text in ((8.85, "lamp"), (10.75, "excitation colour"), (12.42, "one well")):
        ax.text(
            x,
            top + 0.55,
            text,
            ha="center",
            fontsize=BODY,
            fontweight="bold",
            zorder=8,
        )
    ax.text(
        13.45,
        bot - 0.58,
        "emission colour",
        ha="center",
        fontsize=BODY,
        fontweight="bold",
        zorder=8,
    )
    ax.text(
        12.62,
        11.02,
        "read at 90°",
        ha="left",
        fontsize=SMALL,
        color=D.MUTED,
        zorder=8,
    )

    # 4 — what comes back is one number
    reading = float(scene["cube"][WELL_SHOWN, i_em, i_ex])
    D.arrow(ax, (15.34, bot), (15.54, bot), color=D.MUTED, lw=1.8)
    D.card(ax, 15.58, 9.92, 1.70, 0.98, fc="#F4F6FC", z=6, rs=0.10)
    ax.text(
        16.43,
        10.60,
        f"{reading:.2f}",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=D.ACCENT,
        zorder=8,
    )
    ax.text(
        16.43,
        10.20,
        "intensity",
        ha="center",
        va="center",
        fontsize=SMALL,
        color=D.MUTED,
        zorder=8,
    )
    ax.text(
        16.43,
        11.14,
        "one observation",
        ha="center",
        fontsize=BODY,
        fontweight="bold",
        zorder=8,
    )

    # 5 — sweep the colours for one map, stack the wells for the tensor
    axm = _ax(fig, 17.72, 10.20, 1.25, 1.25)
    _map_axes(
        axm,
        scene["cube"][WELL_SHOWN],
        "inferno",
        title="one map",
        xlabel="all colour pairs",
    )
    D.arrow(ax, (19.14, 10.82), (19.44, 10.82), color=D.MUTED, lw=1.8)
    ax.text(
        19.29,
        11.06,
        f"×{n_s}",
        ha="center",
        fontsize=SMALL,
        color=D.MUTED,
        zorder=8,
    )
    _paint_cube(ax, scene["cube"], origin=(20.12, 10.20), scale=0.92)
    ax.text(
        21.80,
        12.10,
        rf"$\mathcal{{X}}\in\mathbb{{R}}^{{{n_s}\times {n_em}\times {n_ex}}}$",
        ha="center",
        fontsize=LEAD,
        color=D.ACCENT,
        fontweight="bold",
        zorder=8,
    )

    ax.text(
        12.60,
        9.30,
        f"Sweep both colour settings over one well and the observations make a map. "
        f"Stack the {n_s} wells and they make the tensor: "
        f"{n_s} × {n_em} × {n_ex} = {n_s * n_em * n_ex:,} readings, one number each.",
        ha="center",
        fontsize=BODY,
        color=D.MUTED,
        zorder=6,
    )


def _row_makeup(fig, ax, scene) -> None:
    x0, y0, w, h = 0.35, 5.35, 23.30, 3.50
    D.card(ax, x0, y0, w, h, fc="#FBFBFE")
    ax.text(
        x0 + 0.30,
        y0 + h - 0.42,
        "What the cube is made of",
        fontsize=CARD_T,
        fontweight="bold",
        va="center",
        zorder=6,
    )
    ax.text(
        x0 + 0.30,
        y0 + h - 0.98,
        "observed cube  =  dye A  +  dye B  +  dye C  +  noise",
        fontsize=LEAD,
        va="center",
        zorder=6,
    )
    ax.text(
        x0 + 0.30,
        y0 + h - 1.44,
        "one dye  =  amount\n∘  emission spectrum\n∘  excitation spectrum",
        fontsize=SMALL,
        color=D.MUTED,
        va="top",
        linespacing=1.5,
        zorder=6,
    )
    D.rank_one_chip(ax, 1.95, 6.30, 1.05, D.ACCENT)
    ax.text(
        1.95,
        5.72,
        "a rank-1 tensor",
        ha="center",
        fontsize=SMALL,
        color=D.MUTED,
        zorder=6,
    )

    dye_eems = scene["dye_eems"]
    em = T.unit_peak(scene["true"][1])
    ex = T.unit_peak(scene["true"][2])
    for r, color in enumerate(D.DYE):
        left = 4.35 + r * 3.30
        axm = _ax(fig, left, 5.95, 1.35, 1.55)
        _map_axes(
            axm,
            dye_eems[r],
            _cmap(color),
            title=D.NAMES[r],
            title_color=color,
            xlabel="excitation →",
            ylabel="emission →" if r == 0 else "",
        )
        axs = _ax(fig, left + 1.55, 5.95, 1.35, 1.55)
        axs.plot(
            np.linspace(0, 1, ex.shape[0]),
            ex[:, r],
            color=color,
            lw=2.2,
            label="excitation",
        )
        axs.plot(
            np.linspace(0, 1, em.shape[0]),
            em[:, r],
            color=color,
            lw=1.6,
            ls="--",
            alpha=0.9,
            label="emission",
        )
        axs.fill_between(
            np.linspace(0, 1, ex.shape[0]),
            ex[:, r],
            color=color,
            alpha=0.12,
        )
        axs.set_xlim(0, 1)
        axs.set_ylim(0, 1.30)
        axs.set_yticks([])
        axs.set_xticks([])
        axs.set_title("its two spectra", fontsize=SMALL, pad=4)
        axs.set_xlabel("colour →", fontsize=TINY)
        axs.spines["left"].set_visible(False)
        axs.spines["bottom"].set_color(D.RULE)
        if r == 0:
            axs.legend(
                frameon=False,
                fontsize=TINY,
                loc="upper left",
                handlelength=1.2,
                borderpad=0.1,
                labelspacing=0.2,
            )

    axa = _ax(fig, 15.55, 5.95, 7.55, 1.55)
    x = np.arange(scene["true_s"].shape[0])
    for r, color in enumerate(D.DYE):
        axa.plot(
            x,
            scene["true_s"][:, r],
            color=color,
            lw=2.4,
            label=D.NAMES[r],
            zorder=3,
        )
        axa.fill_between(x, scene["true_s"][:, r], color=color, alpha=0.08, zorder=2)
    axa.set_ylim(-0.04, 1.42)
    _amount_axes(axa, "the amounts overlap — most wells are mixes")
    axa.legend(frameon=False, ncol=3, fontsize=TINY, loc="upper right")


def _card_cp(fig, ax, scene) -> None:
    x0, y0, w, h = 0.35, 0.35, 11.45, 4.40
    D.card(ax, x0, y0, w, h)
    D.badge(ax, x0 + 0.42, y0 + h - 0.44, "A", color=D.GREEN, r=0.25, fs=LEAD)
    ax.text(
        x0 + 0.80,
        y0 + h - 0.44,
        "CP keeps the three axes — the dyes come back",
        fontsize=CARD_T,
        fontweight="bold",
        va="center",
        zorder=6,
    )
    ax.text(
        x0 + 0.80,
        y0 + h - 0.88,
        "three outer products, one per dye: amount ∘ emission ∘ excitation",
        fontsize=SMALL,
        color=D.MUTED,
        va="center",
        zorder=6,
    )

    for r, color in enumerate(D.DYE):
        axm = _ax(fig, 0.80 + r * 1.60, 1.95, 1.35, 1.35)
        _map_axes(
            axm,
            scene["cp_eems"][r],
            _cmap(color),
            title=f"CP {D.NAMES[r]}",
            title_color=color,
            ylabel="emission →" if r == 0 else "",
            xlabel="excitation →",
        )
    axa = _ax(fig, 6.30, 1.95, 5.05, 1.35)
    x = np.arange(scene["true_s"].shape[0])
    for r, color in enumerate(D.DYE):
        axa.plot(x, scene["true_s"][:, r], color=color, lw=2.3, zorder=3)
        axa.plot(x, scene["cp_s"][:, r], color=color, lw=1.5, ls="--", zorder=4)
    axa.set_ylim(-0.10, 1.30)
    _amount_axes(axa, "amounts: true (solid) vs CP (dashed)")

    D.tick(ax, x0 + 0.55, 1.10, 1.0)
    ax.text(
        x0 + 0.90,
        1.10,
        f"amount correlation {scene['mean_cp_corr']:.2f}   ·   "
        f"leftover error {scene['cp_rel_error']:.3f}, which is the noise that was added",
        fontsize=BODY,
        va="center",
        zorder=6,
    )
    ax.text(
        x0 + 0.90,
        0.66,
        "Kruskal at rank 3: $k_A+k_B+k_C\\geq 2R+2$, so the split is unique "
        "up to relabelling and scale.",
        fontsize=SMALL,
        color=D.MUTED,
        va="center",
        zorder=6,
    )


def _card_svd(fig, ax, scene) -> None:
    x0, y0, w, h = 12.20, 0.35, 11.45, 4.40
    n_s, n_em, n_ex = scene["shape"]
    D.card(ax, x0, y0, w, h)
    D.badge(ax, x0 + 0.42, y0 + h - 0.44, "B", color=D.RED, r=0.25, fs=LEAD)
    ax.text(
        x0 + 0.80,
        y0 + h - 0.44,
        "Flatten first, then SVD — mixed dyes",
        fontsize=CARD_T,
        fontweight="bold",
        va="center",
        zorder=6,
    )
    ax.text(
        x0 + 0.80,
        y0 + h - 0.88,
        f"each map stacked into one long row: a {n_s} × {n_em * n_ex} matrix",
        fontsize=SMALL,
        color=D.MUTED,
        va="center",
        zorder=6,
    )

    for r in range(3):
        axm = _ax(fig, 12.65 + r * 1.60, 1.95, 1.35, 1.35)
        _map_axes(
            axm,
            scene["svd_eems"][r],
            DIVERGING,
            title=f"SVD map {r + 1}",
            title_color=D.MUTED,
            signed=True,
            ylabel="emission →" if r == 0 else "",
            xlabel="excitation →",
        )
    axa = _ax(fig, 18.15, 1.95, 5.05, 1.35)
    x = np.arange(scene["true_s"].shape[0])
    for r, color in enumerate(D.DYE):
        axa.plot(x, scene["true_s"][:, r], color=color, lw=2.3, zorder=3)
        axa.plot(x, scene["svd_s"][:, r], color=color, lw=1.5, ls="--", zorder=4)
    axa.axhline(0.0, color=D.INK, lw=1.0, zorder=2)
    axa.set_ylim(-1.15, 1.30)
    _amount_axes(axa, "amounts: true (solid) vs SVD (dashed)")

    D.flatten_icon(ax, 13.35, 1.10, 0.72)
    D.cross(ax, 14.70, 1.10, 1.0)
    ax.text(
        15.05,
        1.10,
        f"amount correlation {scene['mean_svd_corr']:.3f}   ·   amounts go negative",
        fontsize=BODY,
        va="center",
        zorder=6,
    )
    ax.text(
        x0 + 0.90,
        0.66,
        "The unfolding reconstructs the matrix well and still misses the sources: "
        "these maps are mixes.",
        fontsize=SMALL,
        color=D.MUTED,
        va="center",
        zorder=6,
    )


def _connectors(ax) -> None:
    D.arrow(ax, (6.08, 5.30), (6.08, 4.82), color=D.GREEN, lw=1.8)
    D.arrow(ax, (17.93, 5.30), (17.93, 4.82), color=D.RED, lw=1.8)
    ax.text(
        6.23,
        5.06,
        "factor the cube",
        fontsize=SMALL,
        color=D.GREEN,
        va="center",
        zorder=14,
    )
    ax.text(
        18.08,
        5.06,
        "flatten the cube",
        fontsize=SMALL,
        color=D.RED,
        va="center",
        zorder=14,
    )


def draw_poster(fig: plt.Figure | None = None) -> plt.Figure:
    _style()
    scene = T.mixing_scene()
    if fig is None:
        fig = plt.figure(figsize=(W_FIG, H_FIG), facecolor=D.PAGE, dpi=DPI)
    else:
        fig.set_size_inches(W_FIG, H_FIG)
        fig.set_facecolor(D.PAGE)
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_FIG)
    ax.set_ylim(0, H_FIG)
    ax.axis("off")
    ax.set_facecolor(D.PAGE)

    _header(ax)
    _row_reading(fig, ax, scene)
    _row_makeup(fig, ax, scene)
    _card_cp(fig, ax, scene)
    _card_svd(fig, ax, scene)
    _connectors(ax)
    return fig


def write_png(path: Path | None = None) -> Path:
    dest = Path(path) if path is not None else OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig = draw_poster()
    fig.savefig(dest, dpi=DPI, facecolor=D.PAGE, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return dest


def main() -> None:
    path = write_png(OUT)
    from PIL import Image

    print(
        f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)  {Image.open(path).size}",
    )


if __name__ == "__main__":
    main()
