r"""Write cube-ml.gif, cube-film.gif, fig-cp.png, fig-tucker.png, and cover.png.

Schematic only: no MovieLens bytes, no parafac / tucker fit. Rank-1 volumes
are outer products of Gaussian bumps. The Tucker still is a layout, not a
decomposition of those volumes.

Usage (from repo root):

    uv run --no-project --with matplotlib --with pillow python \
        posts/cp-or-tucker/src/make_cubes.py

`--no-project` matters: a plain `uv run` here deletes and re-syncs the repo venv.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

POST = Path(__file__).resolve().parent.parent

INK = "#1F2430"
MUTED = "#5F6672"
PAPER = "#FFFFFF"
ACCENT = "#4A3AA7"
TEAL = "#2A9D8F"
CORAL = "#E07A5F"
GOLD = "#E9C46A"
DPI = 100
N_AZIM = 36
HOLD_MS = 80
SHRINK_FRAMES = 16


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        },
    )


def _fig_to_image(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=PAPER)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _save_gif(frames: list[Image.Image], durations: list[int], dest: Path) -> None:
    w, h = frames[0].size
    picks = frames[:: max(1, len(frames) // 5)][:5]
    if frames[-1] not in picks:
        picks.append(frames[-1])
    strip = Image.new("RGB", (w, h * len(picks)))
    for i, frame in enumerate(picks):
        strip.paste(frame, (0, i * h))
    palette = strip.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    out = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    dest.parent.mkdir(parents=True, exist_ok=True)
    out[0].save(
        dest,
        save_all=True,
        append_images=out[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=True,
    )


def _cube_faces(origin: np.ndarray, size: np.ndarray) -> list[np.ndarray]:
    o = np.asarray(origin, dtype=float)
    s = np.asarray(size, dtype=float)
    x0, y0, z0 = o
    x1, y1, z1 = o + s
    return [
        np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]),
        np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
        np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
        np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]),
        np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]),
        np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]),
    ]


def _face_shades(rgb: np.ndarray) -> list[tuple[float, float, float]]:
    lifts = (0.22, 0.55, 0.08, 0.40, 0.00, 0.28)
    return [tuple(np.clip(rgb * (1.0 - t) + t, 0, 1)) for t in lifts]


def _draw_solid_cube(ax, origin, size, color, alpha: float = 0.92) -> None:
    rgb = np.array(to_rgb(color))
    faces = _cube_faces(origin, size)
    coll = Poly3DCollection(
        faces,
        facecolors=_face_shades(rgb),
        edgecolors=INK,
        linewidths=0.8,
        alpha=alpha,
    )
    ax.add_collection3d(coll)


def _axis_frame(
    ax,
    labels: tuple[str, str, str],
    lim: float = 1.35,
    origin: np.ndarray | None = None,
    size: np.ndarray | None = None,
) -> None:
    """Label the three modes of the box that was just drawn.

    The offsets are measured from that box, not from the axes limits, so a
    small core keeps its labels against its own edges instead of floating a
    core-width away from the thing they name.
    """
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_zlim(0, lim)
    ax.set_box_aspect((1, 1, 1), zoom=0.92)
    ax.set_axis_off()
    lo = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    ext = np.ones(3) if size is None else np.asarray(size, dtype=float)
    hi = lo + ext
    mid = lo + 0.5 * ext
    pad = 0.20 * lim
    # Hang the labels off the corner nearest the camera. Placed on the far side
    # they project onto the faces, and matplotlib draws text over a
    # Poly3DCollection regardless of depth, so "Month" lands on the cube's lid.
    drop = lo[2] - 0.10 * lim
    ax.text(mid[0], hi[1] + pad, drop, labels[0], ha="center", fontsize=11, color=INK)
    ax.text(hi[0] + pad, mid[1], drop, labels[1], ha="center", fontsize=11, color=INK)
    # The vertical label goes beside the right-hand silhouette edge. On the near
    # corner it would sit on the cube: that corner's outward direction points at
    # the camera, so pushing it out moves it nowhere on screen.
    ax.text(
        lo[0] - 0.12 * lim,
        hi[1] + pad,
        mid[2],
        labels[2],
        ha="center",
        fontsize=11,
        color=INK,
    )


def _cube_frame(
    labels: tuple[str, str, str],
    azim: float,
    scale: float | tuple[float, float, float] = 1.0,
    color: str = ACCENT,
    title: str = "",
) -> Image.Image:
    fig = plt.figure(figsize=(5.2, 4.4), dpi=DPI, facecolor=PAPER)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PAPER)
    size = np.broadcast_to(np.asarray(scale, dtype=float), (3,)).copy()
    origin = (1.0 - size) * 0.5
    _draw_solid_cube(ax, origin, size, color)
    _axis_frame(ax, labels, origin=origin, size=size)
    ax.view_init(elev=18, azim=azim)
    if title:
        ax.set_title(title, fontsize=12, pad=4, color=INK)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
    return _fig_to_image(fig)


def _azim(i: int) -> float:
    """Rock the camera instead of spinning it.

    A full turn carries the near corner — and the three labels hung off it —
    round to the back, where they project onto the faces. A rock keeps the same
    corner facing the reader and still shows three dimensions.
    """
    return 35.0 + 18.0 * float(np.sin(2.0 * np.pi * i / N_AZIM))


def write_ml_gif(path: Path) -> Path:
    labels = ("User", "Movie", "Month")
    frames = [
        _cube_frame(labels, azim=_azim(i), title="user × movie × month")
        for i in range(N_AZIM)
    ]
    _save_gif(frames, [HOLD_MS] * N_AZIM, path)
    return path


def write_film_gif(path: Path) -> Path:
    labels = ("Height", "Width", "Colour")
    # The core's axes are ranks, not the original modes, so the shrink frames
    # relabel. Colour has three slices and a film keeps all three, so R_3 stays
    # full length while the two picture modes give ground.
    core_labels = (r"$R_1$", r"$R_2$", r"$R_3$")
    final = np.array([0.35, 0.35, 1.0])
    frames = [
        _cube_frame(labels, azim=_azim(i), title="height × width × colour")
        for i in range(N_AZIM)
    ]
    durs = [HOLD_MS] * N_AZIM
    for k in range(SHRINK_FRAMES):
        t = (k + 1) / SHRINK_FRAMES
        scale = 1.0 - (1.0 - final) * t
        last = k == SHRINK_FRAMES - 1
        title = r"core $\mathcal{G}$" if last else "each mode has its own budget"
        frames.append(
            _cube_frame(
                core_labels,
                azim=35,
                scale=tuple(scale),
                color=TEAL,
                title=title,
            ),
        )
        durs.append(120 if not last else 1400)
    _save_gif(frames, durs, path)
    return path


def _bump(n: int, center: float, width: float) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, n)
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def _rank1(n: int, centres: tuple[float, float, float], width: float) -> np.ndarray:
    a = _bump(n, centres[0], width)
    b = _bump(n, centres[1], width)
    c = _bump(n, centres[2], width)
    return a[:, None, None] * b[None, :, None] * c[None, None, :]


def _occupied_bounds(vol: np.ndarray, thresh: float = 0.18, pad: int = 1) -> tuple:
    """Smallest cube-shaped window holding every lit voxel.

    The bumps fill about half of the 0..12 grid, so framing on the grid leaves
    the panels mostly white and the blobs tiny once the figure is scaled into
    the text column. One window is shared by every panel, or the terms would
    move between the sum and their own panels.
    """
    mag = np.abs(vol)
    idx = np.argwhere(mag > thresh * (mag.max() + 1e-12))
    lo = idx.min(axis=0) - pad
    hi = idx.max(axis=0) + 1 + pad
    side = float(np.max(hi - lo))
    centre = 0.5 * (lo + hi)
    return centre - 0.5 * side, centre + 0.5 * side


def _draw_voxels(
    ax,
    vol: np.ndarray,
    color: str,
    thresh: float = 0.18,
    bounds: tuple | None = None,
) -> None:
    mag = np.abs(vol)
    peak = mag.max() + 1e-12
    filled = mag > thresh * peak
    rgb = to_rgb(color)
    fc = np.zeros(vol.shape + (4,))
    fc[..., 0], fc[..., 1], fc[..., 2] = rgb
    fc[..., 3] = np.where(filled, 0.28 + 0.65 * mag / peak, 0.0)
    ax.voxels(filled, facecolors=fc, edgecolor="none")
    lo, hi = bounds if bounds is not None else (np.zeros(3), np.array(vol.shape))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    # The three terms sit in different corners, so their shared window is most
    # of the grid; the remaining whitespace is the 3D axes margin, and zoom is
    # what removes it.
    ax.set_box_aspect((1, 1, 1), zoom=1.28)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=40)


CP_SPECS = [
    ((-0.55, -0.45, 0.40), TEAL),
    ((0.15, 0.55, -0.35), CORAL),
    ((0.60, -0.20, 0.55), GOLD),
]


def write_cp_png(path: Path) -> Path:
    n = 12
    specs = CP_SPECS
    terms = [_rank1(n, c, 0.28) for c, _ in specs]
    total = sum(terms)
    bounds = _occupied_bounds(total)
    panels = [(total, ACCENT, r"sum of concepts")] + [
        (vol, color, rf"$r={i}$")
        for i, (vol, (_, color)) in enumerate(zip(terms, specs), start=1)
    ]
    # 2×2, not 1×4: a four-panel strip scaled into the text column leaves each
    # blob about 130 px across.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.0, 6.4),
        dpi=140,
        subplot_kw={"projection": "3d"},
    )
    fig.patch.set_facecolor(PAPER)
    for ax, (vol, color, title) in zip(axes.ravel(), panels):
        ax.set_facecolor(PAPER)
        _draw_voxels(ax, vol, color, bounds=bounds)
        ax.set_title(title, fontsize=11, pad=6, color=INK)
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.94,
        bottom=0.01,
        wspace=0.02,
        hspace=0.06,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=PAPER)
    plt.close(fig)
    return path


def write_tucker_png(path: Path) -> Path:
    rng = np.random.default_rng(7)
    n, r1, r2, r3 = 12, 4, 4, 3
    factors = [
        rng.normal(size=(n, r1)),
        rng.normal(size=(n, r2)),
        rng.normal(size=(n, r3)),
    ]
    # Layout only: a full-size cube next to a small core, plus three factor maps.
    # Not a fit of the CP volumes.
    fig = plt.figure(figsize=(12.0, 4.6), dpi=140, facecolor=PAPER)
    # Column 4 is an empty gutter: the core panel hangs its $R_3$ label to the
    # right, and without the gap it lands on the $V$ heatmap.
    gs = fig.add_gridspec(
        3,
        8,
        hspace=0.38,
        wspace=0.28,
        left=0.02,
        right=0.97,
        top=0.88,
        bottom=0.08,
    )
    ax_x = fig.add_subplot(gs[:, :2], projection="3d")
    ax_g = fig.add_subplot(gs[:, 2:4], projection="3d")
    ax_x.set_facecolor(PAPER)
    ax_g.set_facecolor(PAPER)
    _draw_solid_cube(ax_x, np.zeros(3), np.ones(3), ACCENT)
    _axis_frame(ax_x, ("$I$", "$J$", "$K$"), lim=1.15, size=np.ones(3))
    ax_x.view_init(elev=18, azim=40)
    ax_x.set_title(r"$\mathcal{X}$", fontsize=12, pad=6)
    g_origin = np.zeros(3)
    # True scale against the unit cube beside it: a 4x4x3 core of a 12-cube is
    # 1/36 of the volume, and inflating it to fit the eye would understate what
    # Tucker saves — the one number this figure exists to show.
    g_size = np.array([r1 / n, r2 / n, r3 / n])
    _draw_solid_cube(ax_g, g_origin, g_size, TEAL)
    _axis_frame(
        ax_g,
        (r"$R_1$", r"$R_2$", r"$R_3$"),
        lim=1.15,
        origin=g_origin,
        size=g_size,
    )
    ax_g.view_init(elev=18, azim=40)
    ax_g.set_title(r"core $\mathcal{G}$", fontsize=12, pad=6)
    names = (r"$U$", r"$V$", r"$W$")
    for i, (factor, name) in enumerate(zip(factors, names)):
        ax = fig.add_subplot(gs[i, 5:])
        im = ax.imshow(factor, aspect="auto", cmap="magma")
        ax.set_ylabel(name, rotation=0, labelpad=16, va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=PAPER)
    plt.close(fig)
    return path


def write_cover_png(path: Path) -> Path:
    """1200×630 from the post's own two schematics, side by side.

    The claim is the contrast, so neither fig-cp nor fig-tucker carries it
    alone: one panel holds CP's separately coloured parts, the other holds the
    full array beside the core it compresses to.
    """
    n = 12
    fig = plt.figure(figsize=(12.0, 6.3), dpi=100, facecolor=PAPER)
    ax_cp = fig.add_subplot(121, projection="3d")
    ax_tk = fig.add_subplot(122, projection="3d")

    ax_cp.set_facecolor(PAPER)
    bounds = _occupied_bounds(sum(_rank1(n, c, 0.28) for c, _ in CP_SPECS))
    for centres, color in CP_SPECS:
        _draw_voxels(ax_cp, _rank1(n, centres, 0.28), color, bounds=bounds)
    ax_cp.set_box_aspect((1, 1, 1), zoom=1.15)
    ax_cp.set_title(
        r"CP: $\sum_r a_r \circ b_r \circ c_r$",
        fontsize=17,
        pad=2,
        color=INK,
    )

    ax_tk.set_facecolor(PAPER)
    _draw_solid_cube(ax_tk, np.zeros(3), np.ones(3), ACCENT)
    # +y projects to screen right, so the core sits after the array, not before it.
    _draw_solid_cube(
        ax_tk,
        np.array([0.05, 1.30, 0.05]),
        np.array([0.42, 0.42, 0.30]),
        TEAL,
    )
    ax_tk.set_xlim(0, 1.85)
    ax_tk.set_ylim(0, 1.85)
    ax_tk.set_zlim(0, 1.85)
    ax_tk.set_box_aspect((1, 1, 1), zoom=1.30)
    ax_tk.set_axis_off()
    ax_tk.view_init(elev=18, azim=35)
    ax_tk.set_title(
        r"Tucker: $\mathcal{G} \times_1 U \times_2 V \times_3 W$",
        fontsize=17,
        pad=2,
        color=INK,
    )

    fig.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.0, wspace=0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=PAPER)
    plt.close(fig)
    return path


def main() -> None:
    _style()
    plt.switch_backend("Agg")
    paths = (
        write_ml_gif(POST / "cube-ml.gif"),
        write_film_gif(POST / "cube-film.gif"),
        write_cp_png(POST / "fig-cp.png"),
        write_tucker_png(POST / "fig-tucker.png"),
        write_cover_png(POST / "cover.png"),
    )
    for p in paths:
        kb = p.stat().st_size / 1024
        print(f"wrote {p} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
