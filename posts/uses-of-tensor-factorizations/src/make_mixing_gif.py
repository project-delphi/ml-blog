"""Write media/mixing.gif: what the cube is, then CP vs flatten-SVD.

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
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensors as T

INK = "#1F2430"
MUTED = "#5F6672"
GREY = "#999999"
ACCENT = "#4A3AA7"
TEAL = "#2A9D8F"
CORAL = "#E07A5F"
DYE = [ACCENT, TEAL, CORAL]
NAMES = ("dye A", "dye B", "dye C")
POST = Path(__file__).resolve().parent.parent
OUT = POST / "media" / "mixing.gif"


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.edgecolor": "0.85",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _cmap(hex_color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("dye", ["#FFFFFF", hex_color])


def _maxabs(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    peak = np.max(np.abs(a), axis=0, keepdims=True)
    peak[peak == 0] = 1.0
    return a / peak


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _new_fig():
    fig = plt.figure(figsize=(9.6, 4.35), facecolor="white")
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
            ax.plot(
                x,
                recovered[:, r],
                color=DYE[r],
                lw=1.6,
                ls="--",
                zorder=4,
            )
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
        ax.set_title(
            f"Solid = true. Dashed = {recovered_label}.",
            fontsize=10,
        )


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.045, text, ha="center", va="center", fontsize=11, color=INK)


def _load():
    rng = np.random.default_rng(T.SEED)
    cube, true = T.make_mixing_cube(rng)
    n_s, n_em, n_ex = cube.shape
    fit = T.mixing_fit(cube, true)
    from tensorly.decomposition import parafac

    _weights, factors = parafac(
        cube, rank=T.MIX_RANK, n_iter_max=200, init="svd", random_state=T.SEED
    )
    cp_ex, _ = T.align_factors(true[2], factors[2])
    unfolding = cube.reshape(n_s, -1)
    u, _, vt = np.linalg.svd(unfolding, full_matrices=False)
    order, signs = _match_parts(true[0], u[:, : T.MIX_RANK])
    u_al = u[:, : T.MIX_RANK][:, order] * signs
    vt_al = vt[: T.MIX_RANK][order] * signs[:, None]
    cp_e = np.asarray(fit["cp_emission"])
    return {
        "cube": cube,
        "true_s": _maxabs(true[0]),
        "cp_s": _maxabs(np.asarray(fit["cp_sample"])),
        "svd_s": _maxabs(u_al),
        "dye_eems": [np.outer(true[1][:, r], true[2][:, r]) for r in range(3)],
        "cp_eems": [np.outer(cp_e[:, r], cp_ex[:, r]) for r in range(3)],
        "svd_eems": [vt_al[r].reshape(n_em, n_ex) for r in range(T.MIX_RANK)],
        "mean_cp": fit["mean_cp_corr"],
        "mean_svd": fit["mean_svd_corr"],
        "cp_err": fit["cp_rel_error"],
    }


def _match_order(true: np.ndarray, est: np.ndarray) -> np.ndarray:
    t = true / np.linalg.norm(true, axis=0, keepdims=True)
    e = est / np.linalg.norm(est, axis=0, keepdims=True)
    corr = t.T @ e
    row, col = linear_sum_assignment(-np.abs(corr))
    order = np.empty(est.shape[1], dtype=int)
    order[row] = col
    return order


def _match_parts(true: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = _match_order(true, est)
    aligned = est[:, order]
    signs = np.sign(np.sum(true * aligned, axis=0))
    signs[signs == 0] = 1.0
    return order, signs


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
            scene["mean_cp"],
            "CP",
        )
        title = "CP names the dyes"
        cap = (
            f"CP writes the cube as three outer products. "
            f"Amount correlation {corr:.2f}. Leftover error {scene['cp_err']:.3f} (the noise)."
        )
    else:
        maps, rec, corr, label = (
            scene["svd_eems"],
            scene["svd_s"],
            scene["mean_svd"],
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
    _three_eems(
        fig,
        gs,
        maps,
        [_cmap(c) for c in DYE],
        titles,
    )
    ax_a = fig.add_subplot(gs[1])
    _amounts(ax_a, scene["true_s"], recovered=rec, recovered_label=label)
    _caption(fig, cap)
    fig.suptitle(title, fontsize=13, color=INK, y=0.97)
    return _fig_to_pil(fig)


def write_gif(path: Path | None = None) -> Path:
    _style()
    scene = _load()
    frames: list[Image.Image] = []
    durs: list[int] = []

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
    picks = [frames[2], frames[7], frames[-2], frames[-1]]
    strip = Image.new("RGB", (w, h * len(picks)))
    for i, f in enumerate(picks):
        strip.paste(f, (0, i * h))
    palette = strip.quantize(colors=72, method=Image.Quantize.MEDIANCUT)
    out = [f.quantize(palette=palette) for f in frames]
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
