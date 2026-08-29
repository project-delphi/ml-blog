r"""Write media/mixing.gif: the instrument, what the cube is, then CP vs flatten-SVD.

Shares the palette, glyphs, and fitted scene with ``make_mixing_poster.py`` so
the still and the animation cannot drift apart.

Usage (from repo root or this directory):

    .venv-tensor-factorizations/bin/python \\
        posts/uses-of-tensor-factorizations/src/make_mixing_gif.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draw as D
import tensors as T

INK, MUTED = D.INK, D.MUTED
DYE, NAMES = list(D.DYE), D.NAMES
POST = Path(__file__).resolve().parent.parent
OUT = POST / "media" / "mixing.gif"

W_FIG, H_FIG = 9.6, 4.35
WELL_SHOWN = 10


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": D.FONT,
            "axes.edgecolor": "0.85",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def _cmap(hex_color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("dye", ["#FFFFFF", hex_color])


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _new_fig():
    fig = plt.figure(figsize=(W_FIG, H_FIG), facecolor="white")
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.08, 1.12],
        wspace=0.32,
        left=0.06,
        right=0.98,
        top=0.80,
        bottom=0.18,
    )
    return fig, gs


def _heatmap(ax, z: np.ndarray, cmap, title: str) -> None:
    z = np.asarray(z, dtype=float)
    vmax = float(np.max(np.abs(z))) or 1.0
    ax.imshow(
        np.abs(z),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("excitation colour →")
    ax.set_ylabel("emission colour →")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)


def _three_eems(fig, gs, maps: list[np.ndarray], cmaps, titles: list[str]) -> None:
    inner = gs[0].subgridspec(3, 1, hspace=0.38)
    for i, (z, cmap, title) in enumerate(zip(maps, cmaps, titles)):
        ax = fig.add_subplot(inner[i])
        _heatmap(ax, z, cmap, title)
        if i < 2:
            ax.set_xlabel("")
        if i != 1:
            ax.set_ylabel("")


def _amounts(
    ax,
    truth: np.ndarray,
    needle: int | None = None,
    recovered: np.ndarray | None = None,
    recovered_label: str = "",
    n_show: int | None = None,
) -> None:
    n = truth.shape[0]
    x = np.arange(n)
    n_show = truth.shape[1] if n_show is None else n_show
    for r in range(n_show):
        ax.plot(x, truth[:, r], color=DYE[r], lw=2.2, label=NAMES[r], zorder=3)
        if recovered is not None:
            ax.plot(x, recovered[:, r], color=DYE[r], lw=1.6, ls="--", zorder=4)
    if needle is not None:
        ax.axvline(needle, color=INK, lw=1.15, zorder=2)
        ax.scatter(
            [needle] * n_show,
            truth[needle, :n_show],
            c=DYE[:n_show],
            s=28,
            zorder=5,
            edgecolors="white",
            linewidths=0.6,
        )
    ax.set_xlim(-0.5, n - 0.5)
    ymin, ymax = -0.08, 1.18
    if recovered is not None:
        ymin = min(ymin, float(np.min(recovered)) - 0.12)
        ymax = max(ymax, float(np.max(recovered)) + 0.12)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("sample")
    ax.set_ylabel("amount")
    ax.yaxis.grid(True, color="#EEEFF2", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    if recovered is None:
        ax.set_title("How much of each dye is in each sample", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    else:
        ax.set_title(f"Solid = true. Dashed = {recovered_label}.", fontsize=10)


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.045, text, ha="center", va="center", fontsize=11, color=INK)


def _frame_instrument(scene) -> Image.Image:
    """Where the numbers come from — the poster's spine, flat for GIF colours."""
    fig = plt.figure(figsize=(W_FIG, H_FIG), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_FIG)
    ax.set_ylim(0, H_FIG)
    ax.axis("off")

    top, bot = 2.90, 1.35
    pick_ex, pick_em = D.SPECTRUM[1], D.SPECTRUM[5]
    weights = scene["true_s"][WELL_SHOWN]

    D.lamp(ax, 1.05, top, 0.62, soft=False)
    D.beam(
        ax,
        (1.42, top),
        (2.32, top + 0.02),
        0.08,
        0.17,
        D.wash(D.AMBER, 0.25),
        alpha=0.65,
    )
    D.prism(ax, 2.50, top, 0.62)
    D.fan(ax, (2.67, top + 0.02), 3.52, 0.26)
    D.beam(ax, (3.48, top), (4.55, top), 0.09, 0.11, pick_ex, alpha=0.9, z=8)
    D.slit(ax, 3.58, top, 0.70, gap=0.12)
    D.well(ax, 4.85, top, 0.76, weights=weights, soft=False)

    D.beam(ax, (4.85, top - 0.36), (4.85, bot), 0.10, 0.09, pick_em, alpha=0.85)
    D.beam(ax, (4.81, bot), (5.30, bot), 0.09, 0.12, pick_em, alpha=0.85)
    D.prism(ax, 5.58, bot, 0.60)
    D.fan(ax, (5.74, bot + 0.02), 6.50, 0.23)
    D.beam(ax, (6.46, bot), (6.94, bot), 0.08, 0.10, pick_em, alpha=0.9, z=8)
    D.slit(ax, 6.55, bot, 0.66, gap=0.12)
    reading = float(scene["cube"][WELL_SHOWN, 12, 9])
    D.detector(ax, 7.22, bot, 0.62, reading=f"{reading:.2f}")

    for x, text in ((1.05, "lamp"), (2.95, "excitation colour"), (4.85, "one well")):
        ax.text(x, top + 0.58, text, ha="center", fontsize=10, fontweight="bold")
    ax.text(
        5.85,
        bot - 0.52,
        "emission colour",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        7.22,
        bot + 0.55,
        "one observation",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=D.ACCENT,
    )
    ax.text(5.05, 2.15, "read at 90°", ha="left", fontsize=9.5, color=MUTED)
    n_s, n_em, n_ex = scene["shape"]
    D.settings_panel(
        ax,
        0.45,
        0.62,
        2.55,
        1.55,
        [
            ("well", f"{WELL_SHOWN + 1} of {n_s}", None),
            ("excitation", f"10 of {n_ex}", pick_ex),
            ("emission", f"13 of {n_em}", pick_em),
        ],
        header="inputs you set",
        fs=9.5,
    )

    fig.suptitle("Where the numbers come from", fontsize=13, color=INK, y=0.97)
    _caption(
        fig,
        "One excitation colour, one emission colour, one number. "
        "Sweep both and a well becomes a map.",
    )
    return _fig_to_pil(fig)


def _frame_build(scene, k: int) -> Image.Image:
    """First k dyes, frozen at the middle sample."""
    fig, gs = _new_fig()
    mid = scene["true_s"].shape[0] // 2
    z = sum(scene["dye_eems"][:k])
    ax_h = fig.add_subplot(gs[0])
    if k == 1:
        _heatmap(ax_h, z, _cmap(DYE[0]), "One dye, one map")
    else:
        _heatmap(ax_h, z, "magma", f"{k} dyes added together")
    ax_a = fig.add_subplot(gs[1])
    _amounts(ax_a, scene["true_s"], needle=mid, n_show=k)
    dyes = ", ".join(NAMES[:k])
    _caption(fig, f"Each dye is a map (left) times an amount (right). Now: {dyes}.")
    fig.suptitle("What the cube is made of", fontsize=13, color=INK, y=0.97)
    return _fig_to_pil(fig)


def _frame_sample(scene, i: int) -> Image.Image:
    fig, gs = _new_fig()
    ax_h = fig.add_subplot(gs[0])
    _heatmap(ax_h, scene["cube"][i], "magma", f"Sample {i + 1} of 20 — the recording")
    ax_a = fig.add_subplot(gs[1])
    _amounts(ax_a, scene["true_s"], needle=i)
    mix = scene["true_s"][i]
    parts = " + ".join(f"{mix[r]:.2f} {NAMES[r]}" for r in range(3))
    _caption(fig, f"This map is {parts}. The methods see only the maps, not the split.")
    fig.suptitle("What you actually measure", fontsize=13, color=INK, y=0.97)
    return _fig_to_pil(fig)


def _frame_recover(scene, kind: str) -> Image.Image:
    fig, gs = _new_fig()
    if kind == "cp":
        maps, rec, corr, label = (
            scene["cp_eems"],
            scene["cp_s"],
            scene["mean_cp_corr"],
            "CP",
        )
        title = "CP names the dyes"
        cap = (
            f"CP writes the cube as three outer products. "
            f"Amount correlation {corr:.2f}. "
            f"Leftover error {scene['cp_rel_error']:.3f} (the noise)."
        )
    else:
        maps, rec, corr, label = (
            scene["svd_eems"],
            scene["svd_s"],
            scene["mean_svd_corr"],
            "flatten-then-SVD",
        )
        title = "Flatten, then SVD — mixed dyes"
        cap = (
            f"Smashed to a 20×432 matrix, then SVD. "
            f"Amount correlation {corr:.2f}. Maps are mixes, not dyes."
        )
    titles = (
        [f"CP: {n}" for n in NAMES]
        if kind == "cp"
        else ["SVD map 1", "SVD map 2", "SVD map 3"]
    )
    _three_eems(fig, gs, maps, [_cmap(c) for c in DYE], titles)
    ax_a = fig.add_subplot(gs[1])
    _amounts(ax_a, scene["true_s"], recovered=rec, recovered_label=label)
    _caption(fig, cap)
    fig.suptitle(title, fontsize=13, color=INK, y=0.97)
    return _fig_to_pil(fig)


def write_gif(path: Path | None = None) -> Path:
    _style()
    scene = T.mixing_scene()
    frames: list[Image.Image] = []
    durs: list[int] = []

    frames.append(_frame_instrument(scene))
    durs.append(3000)

    for k in (1, 2, 3):
        frames.append(_frame_build(scene, k))
        durs.append(1600 if k < 3 else 1800)

    sweep = (2, 5, 7, 9, 10, 11, 13, 16, 18)
    for i in sweep:
        frames.append(_frame_sample(scene, i))
        durs.append(380)
    durs[-1] = 900

    frames.append(_frame_recover(scene, "cp"))
    durs.append(2800)
    frames.append(_frame_recover(scene, "svd"))
    durs.append(3200)

    dest = Path(path) if path is not None else OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    w, h = frames[0].size
    picks = [frames[0], frames[3], frames[8], frames[-2], frames[-1]]
    strip = Image.new("RGB", (w, h * len(picks)))
    for i, f in enumerate(picks):
        strip.paste(f, (0, i * h))
    palette = strip.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    # No dithering: flat illustration fills speckle badly in a 128-colour GIF.
    out = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out[0].save(
        dest,
        save_all=True,
        append_images=out[1:],
        duration=durs,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return dest


def main() -> None:
    path = write_gif(OUT)
    kb = path.stat().st_size / 1024
    print(f"wrote {path} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
