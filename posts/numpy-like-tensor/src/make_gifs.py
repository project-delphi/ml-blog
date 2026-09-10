r"""Write reshape.gif, transpose.gif, and fig-order.png for the post.

One 2x3x4 tensor, 24 colours (C-order indices 0..23). Reshape keeps storage
order; transpose reorders the flat list. Both GIFs share the palette.

Usage (from repo root):

    uv run --with matplotlib --with pillow python \
        posts/numpy-like-tensor/src/make_gifs.py
"""

from __future__ import annotations

import colorsys
import io
from dataclasses import dataclass
from math import prod
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
from matplotlib.patches import (  # type: ignore[import-not-found]
    FancyBboxPatch,
    Rectangle,
)
from PIL import Image  # type: ignore[import-not-found]

POST = Path(__file__).resolve().parent.parent
SHAPE0 = (2, 3, 4)
N = prod(SHAPE0)
INK = "#1b1d21"
MUTED = "#5c6370"
PAPER = "#ffffff"
W_FIG, H_FIG = 8.2, 4.7
DPI = 88
N_MORPH = 8
HOLD_MS = 3000
MORPH_MS = 160

# Layout region and storage strip in axes data coordinates.
LAYOUT = (0.25, 1.85, 11.55, 6.15)
STRIP = (0.25, 0.28, 11.55, 1.22)
AX_LIM = (0.0, 11.8, 0.0, 6.85)


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def _palette() -> list[tuple[float, float, float]]:
    """24 hues, one per C-order index, so storage order reads as a rainbow."""
    return [colorsys.hsv_to_rgb(i / N, 0.78, 0.92) for i in range(N)]


PALETTE = _palette()


def _ink_for(rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > 0.55 else "#ffffff"


def _unravel(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    for size in reversed(shape):
        out.append(index % size)
        index //= size
    return tuple(reversed(out))


def _ravel(multi: tuple[int, ...], shape: tuple[int, ...]) -> int:
    index = 0
    for idx, size in zip(multi, shape, strict=True):
        index = index * size + idx
    return index


def _transpose_map(
    old_shape: tuple[int, ...] = SHAPE0,
    axes: tuple[int, ...] = (2, 1, 0),
) -> tuple[tuple[int, ...], list[int]]:
    """Return new shape and, for each colour i, its new C-order slot."""
    new_shape = tuple(old_shape[a] for a in axes)
    new_of_color = [0] * N
    for i in range(N):
        old_multi = _unravel(i, old_shape)
        new_multi = tuple(old_multi[a] for a in axes)
        new_of_color[i] = _ravel(new_multi, new_shape)
    return new_shape, new_of_color


def _grid_boxes(
    nrows: int,
    ncols: int,
    x0: float,
    y0: float,
    cell_w: float,
    cell_h: float,
) -> list[Box]:
    """Row 0 at the top. C-order: row-major."""
    pad = min(cell_w, cell_h) * 0.06
    boxes = []
    for r in range(nrows):
        for c in range(ncols):
            boxes.append(
                Box(
                    x0 + c * cell_w + pad,
                    y0 + (nrows - 1 - r) * cell_h + pad,
                    cell_w - 2 * pad,
                    cell_h - 2 * pad,
                ),
            )
    return boxes


def _layout_boxes(
    shape: tuple[int, ...], region: tuple[float, ...] = LAYOUT
) -> list[Box]:
    """One box per C-order slot, packed as axis-0 slabs (3-D) or one grid (2-D)."""
    x0, y0, x1, y1 = region
    width, height = x1 - x0, y1 - y0
    if len(shape) == 1:
        shape = (shape[0], 1)
    if len(shape) == 2:
        nrows, ncols = shape
        cell_h = height / nrows
        cell_w = width / ncols
        if ncols == 1 and nrows >= 8:
            cell_w = min(width * 0.22, cell_h * 6)
        elif nrows == 1 and ncols >= 8:
            cell_h = min(height * 0.55, cell_w)
        cell_w = min(cell_w, width / ncols)
        cell_h = min(cell_h, height / nrows)
        grid_w, grid_h = cell_w * ncols, cell_h * nrows
        ox = x0 + (width - grid_w) / 2
        oy = y0 + (height - grid_h) / 2
        return _grid_boxes(nrows, ncols, ox, oy, cell_w, cell_h)
    n_slabs, nrows, ncols = shape
    hgap = min(0.28, width * 0.03)
    label_h = 0.38
    usable_h = height - label_h
    usable_w = width - hgap * (n_slabs - 1)
    slab_w = usable_w / n_slabs
    cell = min(slab_w / ncols, usable_h / nrows)
    boxes: list[Box] = []
    for s in range(n_slabs):
        grid_w, grid_h = cell * ncols, cell * nrows
        ox = x0 + s * (slab_w + hgap) + (slab_w - grid_w) / 2
        oy = y0 + (usable_h - grid_h) / 2
        boxes.extend(_grid_boxes(nrows, ncols, ox, oy, cell, cell))
    return boxes


def _slab_labels(
    shape: tuple[int, ...], region: tuple[float, ...] = LAYOUT
) -> list[tuple[float, float, str]]:
    if len(shape) != 3:
        return []
    x0, y0, x1, y1 = region
    width = x1 - x0
    n_slabs = shape[0]
    hgap = min(0.28, width * 0.03)
    slab_w = (width - hgap * (n_slabs - 1)) / n_slabs
    labels = []
    for s in range(n_slabs):
        cx = x0 + s * (slab_w + hgap) + slab_w / 2
        labels.append((cx, y1 - 0.12, f"axis 0 = {s}"))
    return labels


def _strip_boxes(region: tuple[float, ...] = STRIP) -> list[Box]:
    x0, y0, x1, y1 = region
    cell = (x1 - x0) / N
    h = y1 - y0
    pad = min(cell, h) * 0.06
    return [
        Box(x0 + i * cell + pad, y0 + pad, cell - 2 * pad, h - 2 * pad)
        for i in range(N)
    ]


def _lerp_box(a: Box, b: Box, t: float) -> Box:
    u = t * t * (3.0 - 2.0 * t)
    return Box(
        a.x + (b.x - a.x) * u,
        a.y + (b.y - a.y) * u,
        a.w + (b.w - a.w) * u,
        a.h + (b.h - a.h) * u,
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "text.color": INK,
            "font.size": 10,
            "text.parse_math": False,
        },
    )


def _draw_cell(ax, box: Box, color_i: int, fontsize: float) -> None:
    rgb = PALETTE[color_i]
    ax.add_patch(
        FancyBboxPatch(
            (box.x, box.y),
            box.w,
            box.h,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=0.6,
            edgecolor="white",
            facecolor=rgb,
            mutation_aspect=0.4,
        ),
    )
    ax.text(
        box.cx,
        box.cy,
        str(color_i),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=_ink_for(rgb),
        fontweight="bold",
    )


def _draw_frame(
    layout_of_color: list[Box],
    strip_of_color: list[Box],
    title: str,
    note: str,
    shape: tuple[int, ...],
    *,
    layout_fs: float | None = None,
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(W_FIG, H_FIG), facecolor=PAPER)
    ax.set_xlim(AX_LIM[0], AX_LIM[1])
    ax.set_ylim(AX_LIM[2], AX_LIM[3])
    ax.set_axis_off()
    ax.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)

    ax.text(0.25, 6.58, title, fontsize=13, fontweight="bold", color=INK, va="bottom")
    ax.text(11.55, 6.58, note, fontsize=11, color=MUTED, ha="right", va="bottom")

    ax.text(0.25, LAYOUT[3] + 0.02, "layout", fontsize=9, color=MUTED, va="bottom")
    ax.text(
        0.25, STRIP[3] + 0.04, "storage order", fontsize=9, color=MUTED, va="bottom"
    )

    # Light wells behind each panel.
    ax.add_patch(
        Rectangle(
            (LAYOUT[0] - 0.08, LAYOUT[1] - 0.12),
            LAYOUT[2] - LAYOUT[0] + 0.16,
            LAYOUT[3] - LAYOUT[1] + 0.28,
            linewidth=0,
            facecolor="#f4f5f7",
            zorder=0,
        ),
    )
    ax.add_patch(
        Rectangle(
            (STRIP[0] - 0.08, STRIP[1] - 0.10),
            STRIP[2] - STRIP[0] + 0.16,
            STRIP[3] - STRIP[1] + 0.20,
            linewidth=0,
            facecolor="#f4f5f7",
            zorder=0,
        ),
    )

    for cx, cy, lab in _slab_labels(shape):
        ax.text(cx, cy, lab, ha="center", va="top", fontsize=8.5, color=MUTED)

    mean_w = sum(b.w for b in layout_of_color) / N
    fs = layout_fs if layout_fs is not None else max(5.0, min(9.5, mean_w * 7.2))
    strip_fs = 7.0
    for i in range(N):
        _draw_cell(ax, layout_of_color[i], i, fs)
        _draw_cell(ax, strip_of_color[i], i, strip_fs)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=PAPER)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _identity_slots() -> list[int]:
    return list(range(N))


def _boxes_for_colors(slot_boxes: list[Box], new_of_color: list[int]) -> list[Box]:
    return [slot_boxes[new_of_color[i]] for i in range(N)]


def _hold(
    shape: tuple[int, ...],
    new_of_color: list[int],
    title: str,
    note: str,
) -> Image.Image:
    layout = _boxes_for_colors(_layout_boxes(shape), new_of_color)
    strip = _boxes_for_colors(_strip_boxes(), new_of_color)
    return _draw_frame(layout, strip, title, note, shape)


def _morph_frames(
    shape_a: tuple[int, ...],
    map_a: list[int],
    shape_b: tuple[int, ...],
    map_b: list[int],
    title: str,
    note: str,
) -> list[Image.Image]:
    lay_a = _boxes_for_colors(_layout_boxes(shape_a), map_a)
    lay_b = _boxes_for_colors(_layout_boxes(shape_b), map_b)
    str_a = _boxes_for_colors(_strip_boxes(), map_a)
    str_b = _boxes_for_colors(_strip_boxes(), map_b)
    frames = []
    for k in range(1, N_MORPH + 1):
        t = k / (N_MORPH + 1)
        layout = [_lerp_box(lay_a[i], lay_b[i], t) for i in range(N)]
        strip = [_lerp_box(str_a[i], str_b[i], t) for i in range(N)]
        shape = shape_b if t >= 0.5 else shape_a
        frames.append(_draw_frame(layout, strip, title, note, shape))
    return frames


def _save_gif(frames: list[Image.Image], durations: list[int], dest: Path) -> None:
    w, h = frames[0].size
    # Shared palette from a contact strip of a few frames; no dither.
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


def write_reshape_gif(path: Path) -> Path:
    ident = _identity_slots()
    stages = [
        (SHAPE0, "original", "shape = (2, 3, 4)"),
        ((24, 1), "reshape(24, 1)", "column vector"),
        ((1, 24), "reshape(1, 24)", "row vector"),
        ((4, 3, 2), "reshape(4, 3, 2)", "a different 3-tensor"),
    ]
    frames: list[Image.Image] = []
    durs: list[int] = []
    for i, (shape, title, note) in enumerate(stages):
        if i:
            prev_shape = stages[i - 1][0]
            frames.extend(_morph_frames(prev_shape, ident, shape, ident, title, note))
            durs.extend([MORPH_MS] * N_MORPH)
        frames.append(_hold(shape, ident, title, note))
        durs.append(HOLD_MS)
    _save_gif(frames, durs, path)
    return path


def write_transpose_gif(path: Path) -> Path:
    ident = _identity_slots()
    new_shape, new_of_color = _transpose_map()
    frames = [_hold(SHAPE0, ident, "original", "shape = (2, 3, 4)")]
    durs = [HOLD_MS]
    frames.extend(
        _morph_frames(
            SHAPE0,
            ident,
            new_shape,
            new_of_color,
            "transpose()",
            "axes (2, 1, 0) → (4, 3, 2)",
        ),
    )
    durs.extend([MORPH_MS] * N_MORPH)
    frames.append(
        _hold(new_shape, new_of_color, "transpose()", "shape = (4, 3, 2)"),
    )
    durs.append(HOLD_MS + 400)
    _save_gif(frames, durs, path)
    return path


def write_order_still(path: Path) -> Path:
    """Side-by-side (4, 3, 2): reshape keeps order, transpose does not."""
    ident = _identity_slots()
    new_shape, new_of_color = _transpose_map()
    left = _hold(new_shape, ident, "reshape(4, 3, 2)", "storage order unchanged")
    right = _hold(new_shape, new_of_color, "transpose()", "storage order changed")
    w, h = left.size
    gap = 12
    header = 36
    canvas = Image.new("RGB", (w * 2 + gap, h + header), PAPER)
    canvas.paste(left, (0, header))
    canvas.paste(right, (w + gap, header))
    # Title drawn as a thin matplotlib strip so type matches the GIFs.
    fig, ax = plt.subplots(figsize=((w * 2 + gap) / DPI, header / DPI), facecolor=PAPER)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.0,
        0.35,
        "Same shape. Different storage order.",
        fontsize=13,
        color=INK,
        fontweight="bold",
    )
    buf = io.BytesIO()
    fig.subplots_adjust(left=0.02, right=0.98, top=1, bottom=0)
    fig.savefig(buf, format="png", dpi=DPI, facecolor=PAPER)
    plt.close(fig)
    buf.seek(0)
    banner = Image.open(buf).convert("RGB").resize((w * 2 + gap, header))
    canvas.paste(banner, (0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    return path


def main() -> None:
    _style()
    plt.switch_backend("Agg")
    reshape_path = write_reshape_gif(POST / "reshape.gif")
    transpose_path = write_transpose_gif(POST / "transpose.gif")
    still_path = write_order_still(POST / "fig-order.png")
    for p in (reshape_path, transpose_path, still_path):
        kb = p.stat().st_size / 1024
        print(f"wrote {p} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
