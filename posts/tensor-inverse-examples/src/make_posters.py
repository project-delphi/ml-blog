"""Poster-style figures: tensor shape, then what the inverse does.

Furniture matches posts/uses-of-tensor-factorizations/src/draw.py so the
three posters sit in the same visual family as the dye-assay figure.

    /Users/ravikalia/Code/github.com/ml-blog/.venv-tensor-factorizations/bin/python \\
        posts/tensor-inverse-examples/src/make_posters.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
    Wedge,
)

INK = "#1B2433"
MUTED = "#5C6570"
RULE = "#DCE0E8"
PAGE = "#F7F8FB"
CARD = "#FFFFFF"
SHADOW = "#E1E5EC"
ACCENT = "#4A3AA7"
TEAL = "#2A9D8F"
CORAL = "#E07A5F"
GOLD = "#E8A33D"
FONT = ["DejaVu Sans"]

W_FIG, H_FIG = 16.4, 8.6
DPI = 160
TITLE, SUB = 22.0, 12.5
BODY, SMALL, TINY = 11.0, 9.5, 8.2

OUT_DIR = Path(__file__).resolve().parent.parent / "media"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT,
            "text.color": INK,
            "font.size": BODY,
        }
    )


def card(ax, x, y, w, h, fc=CARD, ec=RULE, z=1, rs=0.14) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.05, y - 0.06),
            w,
            h,
            boxstyle=f"round,pad=0.01,rounding_size={rs}",
            facecolor=SHADOW,
            edgecolor="none",
            zorder=z - 1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.01,rounding_size={rs}",
            facecolor=fc,
            edgecolor=ec,
            lw=1.1,
            zorder=z,
        )
    )


def badge(ax, x, y, label, color=ACCENT, r=0.20, fs=11.0) -> None:
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=12))
    ax.text(
        x,
        y - 0.01,
        label,
        ha="center",
        va="center",
        fontsize=fs,
        color="white",
        fontweight="bold",
        zorder=13,
    )


def pill(ax, x, y, text, fc="#EDEAF8", tc=ACCENT, fs=8.5, h=0.36) -> None:
    w = 0.22 + fs / 72.0 * 0.58 * len(text)
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0.01,rounding_size={h / 2}",
            facecolor=fc,
            edgecolor="none",
            zorder=12,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold", zorder=13)


def arrow(ax, p0, p1, color=INK, lw=1.4, head=8.0, z=14) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=f"-|>,head_length={head * 0.55},head_width={head * 0.30}",
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def iso(x, y, z, origin, s=1.0, depth=0.42):
    o = np.asarray(origin, dtype=float)
    return o + s * np.array([x + depth * z, y + 0.48 * depth * z])


def cube(ax, origin, labels, sizes, face=ACCENT, s=1.15, z=4) -> None:
    """Isometric 3-mode box. ``labels`` / ``sizes`` are (width, height, depth)."""
    W, H, D = 1.55, 1.55, 1.35

    def p(x, y, z_):
        return iso(x, y, z_, origin, scale := s, depth=0.40)

    # shadow
    ax.add_patch(
        Polygon(
            [p(0.12, -0.08, 0.10), p(W + 0.12, -0.08, 0.10), p(W + 0.12, -0.08, D + 0.10), p(0.12, -0.08, D + 0.10)],
            closed=True,
            facecolor=SHADOW,
            edgecolor="none",
            zorder=z,
        )
    )
    top = [p(0, H, 0), p(W, H, 0), p(W, H, D), p(0, H, D)]
    right = [p(W, 0, 0), p(W, 0, D), p(W, H, D), p(W, H, 0)]
    front = [p(0, 0, 0), p(W, 0, 0), p(W, H, 0), p(0, H, 0)]
    from matplotlib.colors import to_rgb

    rgb = np.array(to_rgb(face))
    ax.add_patch(Polygon(top, closed=True, facecolor=tuple(rgb * 0.55 + 0.45), edgecolor=INK, lw=1.0, zorder=z + 1))
    ax.add_patch(Polygon(right, closed=True, facecolor=tuple(rgb * 0.72 + 0.20), edgecolor=INK, lw=1.0, zorder=z + 2))
    ax.add_patch(Polygon(front, closed=True, facecolor=tuple(rgb * 0.88 + 0.08), edgecolor=INK, lw=1.15, zorder=z + 3))

    bl, br, tl = p(0, 0, 0), p(W, 0, 0), p(0, H, 0)
    brz = p(W, 0, D)
    mid_w = (bl + br) / 2 + np.array([0.0, -0.38])
    ax.text(*mid_w, f"{labels[0]}  {sizes[0]}", ha="center", fontsize=TINY, fontweight="bold", zorder=z + 6)
    mid_h = (bl + tl) / 2 + np.array([-0.40, 0.0])
    ax.text(*mid_h, f"{labels[1]}  {sizes[1]}", ha="center", va="center", fontsize=TINY, fontweight="bold", rotation=90, zorder=z + 6)
    mid_d = (br + brz) / 2 + np.array([0.46, 0.02])
    ax.text(*mid_d, f"{labels[2]}  {sizes[2]}", ha="center", fontsize=TINY, fontweight="bold", rotation=24, zorder=z + 6)


def _new(title: str, subtitle: str):
    _style()
    fig = plt.figure(figsize=(W_FIG, H_FIG), dpi=DPI, facecolor=PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_FIG)
    ax.set_ylim(0, H_FIG)
    ax.axis("off")
    ax.set_facecolor(PAGE)
    ax.add_patch(Rectangle((0, 0), W_FIG, H_FIG, facecolor=PAGE, edgecolor="none", zorder=0))
    ax.text(0.45, 8.18, title, fontsize=TITLE, fontweight="bold", va="center", zorder=5)
    ax.text(0.45, 7.72, subtitle, fontsize=SUB, color=MUTED, va="center", zorder=5)
    return fig, ax


def _stage(ax, n, x, y, w, h, title, color=ACCENT) -> None:
    card(ax, x, y, w, h)
    badge(ax, x + 0.32, y + h - 0.34, str(n), color=color)
    ax.text(x + 0.58, y + h - 0.34, title, fontsize=SMALL, fontweight="bold", va="center", zorder=6)


def _wave(ax, x, y, w, h, color, seed) -> None:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 80)
    env = np.exp(-((t - 0.45) ** 2) / 0.08) + 0.25 * np.exp(-((t - 0.75) ** 2) / 0.03)
    sig = env * np.sin(2 * np.pi * (4 + seed) * t + rng.normal(0, 0.2, t.size))
    xs = x + t * w
    ys = y + h / 2 + 0.42 * h * sig
    ax.plot(xs, ys, color=color, lw=1.15, zorder=8, solid_capstyle="round")


# --- speech ----------------------------------------------------------------


def poster_speech(path: Path) -> Path:
    fig, ax = _new(
        "Three voices. One mix. A 3 × 3 × 3 × 3 cumulant.",
        "The inverse undoes the mix. A signed permutation undoes it too — and swaps the names.",
    )
    colors = (TEAL, GOLD, CORAL)
    names = ("English", "Swedish", "Foochow")

    _stage(ax, 1, 0.35, 3.55, 3.55, 3.85, "Sources", TEAL)
    for i, (c, n) in enumerate(zip(colors, names)):
        yy = 6.55 - i * 0.95
        ax.add_patch(Circle((0.85, yy), 0.18, facecolor=c, edgecolor="none", zorder=8))
        ax.text(1.15, yy, n, fontsize=TINY, va="center", fontweight="bold", zorder=8)
        _wave(ax, 1.15, yy - 0.42, 2.45, 0.38, c, seed=3 + i)

    _stage(ax, 2, 4.20, 3.55, 3.70, 3.85, "Mix  A ∈ ℝ³ˣ³", CORAL)
    ax.text(6.05, 6.85, "3 mics × time", ha="center", fontsize=TINY, color=MUTED, zorder=8)
    for i, c in enumerate(colors):
        _wave(ax, 4.50, 6.25 - i * 0.85, 3.10, 0.62, c, seed=20 + i)
    pill(ax, 6.05, 3.90, "constructed mix, seed 7")

    _stage(ax, 3, 8.20, 3.55, 3.85, 3.85, "Cumulant  C", ACCENT)
    cube(ax, (8.85, 4.15), ("j", "i", "k"), ("3", "3", "3"), face=ACCENT, s=1.05)
    pill(ax, 10.10, 6.85, "order 4   3 × 3 × 3 × 3")
    ax.text(10.10, 3.82, "independent sources → diagonal C", ha="center", fontsize=TINY, color=MUTED, zorder=8)

    _stage(ax, 4, 12.30, 3.55, 3.75, 3.85, "Two inverses", GOLD)
    card(ax, 12.50, 5.55, 3.35, 1.40, fc="#F4F1FC")
    ax.text(14.18, 6.60, "A⁻¹", ha="center", fontsize=SMALL, fontweight="bold", color=ACCENT, zorder=8)
    ax.text(14.18, 6.18, "English stays English.", ha="center", fontsize=TINY, zorder=8)
    ax.text(14.18, 5.82, "Transcript keeps the names.", ha="center", fontsize=TINY, color=MUTED, zorder=8)
    card(ax, 12.50, 3.80, 3.35, 1.55, fc="#FDF3EF")
    ax.text(14.18, 5.00, "P A⁻¹", ha="center", fontsize=SMALL, fontweight="bold", color=CORAL, zorder=8)
    ax.text(14.18, 4.58, "Same words. Swapped speakers.", ha="center", fontsize=TINY, zorder=8)
    ax.text(14.18, 4.20, "Leftover: permutation and scale.", ha="center", fontsize=TINY, color=MUTED, zorder=8)

    arrow(ax, (3.95, 5.50), (4.15, 5.50), color=MUTED, lw=1.2)
    arrow(ax, (7.95, 5.50), (8.15, 5.50), color=MUTED, lw=1.2)
    arrow(ax, (12.10, 5.50), (12.25, 5.50), color=MUTED, lw=1.2)

    card(ax, 0.35, 0.35, 15.70, 2.90)
    ax.text(0.60, 2.80, "How the inverse helps", fontsize=SMALL, fontweight="bold", zorder=8)
    ax.text(
        0.60,
        2.28,
        "A meeting recorder hears three talkers on three channels. The product is a transcript that follows one voice.",
        fontsize=BODY,
        zorder=8,
    )
    ax.text(
        0.60,
        1.72,
        "Joint-diagonalising C recovers a mixing-matrix inverse. That inverse puts each voice on its own channel, so a quote can be attributed.",
        fontsize=BODY,
        zorder=8,
    )
    ax.text(
        0.60,
        1.16,
        "It is not unique. Any signed permutation of A⁻¹ also undoes the mix. The leftover is the speaker label, not the words.",
        fontsize=BODY,
        zorder=8,
    )
    ax.text(
        0.60,
        0.60,
        "A wrong leftover names the person who did not say it.",
        fontsize=BODY,
        fontweight="bold",
        zorder=8,
    )
    fig.savefig(path, dpi=DPI, facecolor=PAGE)
    plt.close(fig)
    return path


# --- ratings ---------------------------------------------------------------


def _sparse_face(ax, origin, s, rng, n=9, color=ACCENT) -> None:
    """Tiny observed-cell grid painted on the front of a cube."""
    W, H = 1.55 * s, 1.55 * s
    o = np.asarray(origin, dtype=float)
    for i in range(n):
        for j in range(n):
            if rng.random() > 0.22:
                continue
            x0, y0 = o[0] + j * W / n, o[1] + i * H / n
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    W / n * 0.82,
                    H / n * 0.82,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.55 + 0.4 * rng.random(),
                    zorder=9,
                )
            )


def poster_ratings(path: Path) -> Path:
    fig, ax = _new(
        "User × movie × month is 80 × 80 × 8. Most cells are empty.",
        "A rank picks one filling of those holes. Two ranks ship two catalogue pages.",
    )
    rng = np.random.default_rng(7)

    _stage(ax, 1, 0.35, 3.55, 3.55, 3.85, "Observed ratings", TEAL)
    cube(ax, (0.85, 4.05), ("movie", "user", "month"), ("80", "80", "8"), face=TEAL, s=1.00)
    pill(ax, 2.12, 3.82, "4,797 of 51,200 cells")

    _stage(ax, 2, 4.20, 3.55, 3.70, 3.85, "Sampling operator  PΩ", CORAL)
    cube(ax, (4.75, 4.05), ("movie", "user", "month"), ("80", "80", "8"), face="#C3CDDB", s=1.00)
    # holes: white squares on the front
    o = iso(0, 0, 0, (4.75, 4.05), s=1.00)
    _sparse_face(ax, (4.78, 4.08), 1.00, rng, n=8, color=CORAL)
    ax.text(6.05, 3.82, "null space = any fill of the holes", ha="center", fontsize=TINY, color=MUTED, zorder=8)

    _stage(ax, 3, 8.20, 3.55, 3.85, 3.85, "Two CP inverses", ACCENT)
    card(ax, 8.40, 5.55, 3.45, 1.40, fc="#F4F1FC")
    ax.text(10.12, 6.60, "rank 3", ha="center", fontsize=SMALL, fontweight="bold", color=ACCENT, zorder=8)
    ax.text(10.12, 6.18, "Terminator  →  3.1", ha="center", fontsize=TINY, zorder=8)
    ax.text(10.12, 5.82, "middling — show it", ha="center", fontsize=TINY, color=MUTED, zorder=8)
    card(ax, 8.40, 3.80, 3.45, 1.55, fc="#FDF3EF")
    ax.text(10.12, 5.00, "rank 8", ha="center", fontsize=SMALL, fontweight="bold", color=CORAL, zorder=8)
    ax.text(10.12, 4.58, "Terminator  →  2.0", ha="center", fontsize=TINY, zorder=8)
    ax.text(10.12, 4.20, "park it", ha="center", fontsize=TINY, color=MUTED, zorder=8)

    _stage(ax, 4, 12.30, 3.55, 3.75, 3.85, "Two pages", GOLD)
    card(ax, 12.50, 5.55, 3.35, 1.40, fc="#F4F1FC")
    ax.text(14.18, 6.60, "Page A", ha="center", fontsize=SMALL, fontweight="bold", color=ACCENT, zorder=8)
    ax.text(14.18, 6.18, "The Terminator  ★★★", ha="center", fontsize=TINY, zorder=8)
    ax.text(14.18, 5.82, "slot used", ha="center", fontsize=TINY, color=MUTED, zorder=8)
    card(ax, 12.50, 3.80, 3.35, 1.55, fc="#FDF3EF")
    ax.text(14.18, 5.00, "Page B", ha="center", fontsize=SMALL, fontweight="bold", color=CORAL, zorder=8)
    ax.text(14.18, 4.58, "The Terminator  hidden", ha="center", fontsize=TINY, zorder=8)
    ax.text(14.18, 4.20, "same user, same month", ha="center", fontsize=TINY, color=MUTED, zorder=8)

    arrow(ax, (3.95, 5.50), (4.15, 5.50), color=MUTED, lw=1.2)
    arrow(ax, (7.95, 5.50), (8.15, 5.50), color=MUTED, lw=1.2)
    arrow(ax, (12.10, 5.50), (12.25, 5.50), color=MUTED, lw=1.2)

    card(ax, 0.35, 0.35, 15.70, 2.90)
    ax.text(0.60, 2.80, "How the inverse helps", fontsize=SMALL, fontweight="bold", zorder=8)
    ax.text(
        0.60,
        2.28,
        "A catalogue page cannot show an empty slot. The product is a ranked title for this user this month.",
        fontsize=BODY,
        zorder=8,
    )
    ax.text(
        0.60,
        1.72,
        "PΩ has no inverse. A CP rank picks one array that matches the observed cells. ALS is how that point is found.",
        fontsize=BODY,
        zorder=8,
    )
    ax.text(
        0.60,
        1.16,
        "The leftover is the rank. User 269, The Terminator, April 1998: rank 3 fills 3.1, rank 8 fills 2.0. True rating 3.",
        fontsize=BODY,
        zorder=8,
    )
    ax.text(
        0.60,
        0.60,
        "A wrong leftover hides a film the user rated, or trains the next model on a rating nobody gave.",
        fontsize=BODY,
        fontweight="bold",
        zorder=8,
    )
    fig.savefig(path, dpi=DPI, facecolor=PAGE)
    plt.close(fig)
    return path


# --- dryer -----------------------------------------------------------------


def _tank(ax, x, y, s=1.0) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.55 * s, y - 0.15 * s),
            1.10 * s,
            1.35 * s,
            boxstyle="round,pad=0.01,rounding_size=0.18",
            facecolor="#E8EEF6",
            edgecolor=INK,
            lw=1.2,
            zorder=7,
        )
    )
    ax.add_patch(
        Wedge((x, y + 1.05 * s), 0.55 * s, 0, 180, facecolor="#C3CDDB", edgecolor=INK, lw=1.1, zorder=8)
    )
    ax.add_patch(Circle((x, y + 0.45 * s), 0.16 * s, facecolor=GOLD, edgecolor=INK, lw=0.8, zorder=9))


def poster_dryer(path: Path) -> Path:
    fig, ax = _new(
        "Output × input × lag is 3 × 3 × L. L is a choice.",
        "Inverting H for a moisture target is a second product. Two products, two actuator settings.",
    )

    _stage(ax, 1, 0.35, 3.55, 3.55, 3.85, "The plant", TEAL)
    _tank(ax, 2.12, 5.15, s=0.95)
    ax.text(2.12, 4.55, "industrial dryer", ha="center", fontsize=TINY, fontweight="bold", zorder=8)
    ax.text(2.12, 4.18, "fuel · fan · feed  →", ha="center", fontsize=TINY, color=MUTED, zorder=8)
    ax.text(2.12, 3.86, "temp · temp · moisture", ha="center", fontsize=TINY, color=MUTED, zorder=8)

    _stage(ax, 2, 4.20, 3.55, 3.70, 3.85, "Impulse response  H", CORAL)
    cube(ax, (4.75, 4.05), ("input", "output", "lag"), ("3", "3", "L"), face=CORAL, s=1.00)
    pill(ax, 6.05, 6.85, "H ∈ ℝ³ˣ³ˣᴸ")
    ax.text(6.05, 3.82, "L = 5 or 15 on this series", ha="center", fontsize=TINY, color=MUTED, zorder=8)

    _stage(ax, 3, 8.20, 3.55, 3.85, 3.85, "Two inverses of H", ACCENT)
    card(ax, 8.40, 5.55, 3.45, 1.40, fc="#FDF3EF")
    ax.text(10.12, 6.60, "lag-0  H[:,:,0]† y", ha="center", fontsize=SMALL, fontweight="bold", color=CORAL, zorder=8)
    ax.text(10.12, 6.18, "one-step product", ha="center", fontsize=TINY, zorder=8)
    ax.text(10.12, 5.82, "κ ≈ 10³ on this slice", ha="center", fontsize=TINY, color=MUTED, zorder=8)
    card(ax, 8.40, 3.80, 3.45, 1.55, fc="#EAF6F4")
    ax.text(10.12, 5.00, "stacked  H† y", ha="center", fontsize=SMALL, fontweight="bold", color=TEAL, zorder=8)
    ax.text(10.12, 4.58, "Einstein / Moore–Penrose", ha="center", fontsize=TINY, zorder=8)
    ax.text(10.12, 4.20, "of the 3 × 15 unfolding", ha="center", fontsize=TINY, color=MUTED, zorder=8)

    _stage(ax, 4, 12.30, 3.55, 3.75, 3.85, "Two settings", GOLD)
    card(ax, 12.50, 5.55, 3.35, 1.40, fc="#FDF3EF")
    ax.text(14.18, 6.60, "lag-0 pick", ha="center", fontsize=SMALL, fontweight="bold", color=CORAL, zorder=8)
    ax.text(14.18, 6.18, "feed ≈ +1,200", ha="center", fontsize=TINY, zorder=8)
    ax.text(14.18, 5.82, "dump raw material", ha="center", fontsize=TINY, color=MUTED, zorder=8)
    card(ax, 12.50, 3.80, 3.35, 1.55, fc="#EAF6F4")
    ax.text(14.18, 5.00, "stacked pick", ha="center", fontsize=SMALL, fontweight="bold", color=TEAL, zorder=8)
    ax.text(14.18, 4.58, "feed ≈ 0", ha="center", fontsize=TINY, zorder=8)
    ax.text(14.18, 4.20, "small correction", ha="center", fontsize=TINY, color=MUTED, zorder=8)

    arrow(ax, (3.95, 5.50), (4.15, 5.50), color=MUTED, lw=1.2)
    arrow(ax, (7.95, 5.50), (8.15, 5.50), color=MUTED, lw=1.2)
    arrow(ax, (12.10, 5.50), (12.25, 5.50), color=MUTED, lw=1.2)

    card(ax, 0.35, 0.35, 15.70, 2.90)
    ax.text(0.60, 2.80, "How the inverse helps", fontsize=SMALL, fontweight="bold", zorder=8)
    ax.text(
        0.60,
        2.28,
        "The dryer has to hit a moisture target without scorching the product or wasting fuel. The product of the inverse is a setting: fuel, fan, feed.",
        fontsize=BODY,
        zorder=8,
    )
    ax.text(
        0.60,
        1.72,
        "Fitting H at a fixed L is ordinary least squares. Inverting H for a desired y is a different product, and two products disagree.",
        fontsize=BODY,
        zorder=8,
    )
    ax.text(
        0.60,
        1.16,
        "The leftover is which inverse you take, and which L you fitted. Same target y. One pick dumps feed. The other barely moves it.",
        fontsize=BODY,
        zorder=8,
    )
    ax.text(
        0.60,
        0.60,
        "A wrong leftover over-dries a batch, or ships it wet.",
        fontsize=BODY,
        fontweight="bold",
        zorder=8,
    )
    fig.savefig(path, dpi=DPI, facecolor=PAGE)
    plt.close(fig)
    return path


def write_all(media: Path | None = None) -> dict[str, Path]:
    media = Path(media) if media is not None else OUT_DIR
    media.mkdir(parents=True, exist_ok=True)
    return {
        "speech": poster_speech(media / "poster-speech.png"),
        "ratings": poster_ratings(media / "poster-ratings.png"),
        "dryer": poster_dryer(media / "poster-dryer.png"),
    }


if __name__ == "__main__":
    written = write_all()
    for key, path in written.items():
        print(f"wrote {path} ({path.stat().st_size:,} bytes)")
