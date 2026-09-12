"""Draw the post's figures from the real cache, and commit the PNGs.

The rendered post executes nothing: every ``{python}`` block in ``index.qmd`` is
``#| eval: false`` and quotes the module it came from. That keeps a clone's
render instant and offline, and it means the figures have to be built here, by
hand, and committed under ``figures/``.

Run from the post directory, after ``src/fetch_data.py`` has written the cache::

    PYTHONPATH=src ../../.venv-fingerprint-algorithms/bin/python src/figures.py
    PYTHONPATH=src ../../.venv-fingerprint-algorithms/bin/python src/figures.py --only cmc

The two figures that report scores -- ``cmc.png`` and ``embedding.png`` -- read
``bench/ladder.json`` and ``bench/rungs.npz``, which ``src/ladder.py --out
--dump`` writes. Everything else is computed from one print on the spot.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import data
import enhance
import matplotlib.pyplot as plt
import minutiae
import minutiae_follow as follow
import numpy as np
import ridges

FIGURES = Path("figures")
BENCH = Path("bench")

# How many fingers to consider when choosing the print the figures are drawn from.
CANDIDATES = 40

# Memoised result of `pick`. Four figures want the same print, and each call
# analyses up to 80 images -- 320 analyses for a run that needs 80.
_PICKED: list = []

INK, PURPLE, TEAL, CORAL, GOLD = "#1F2430", "#4A3AA7", "#2A9D8F", "#E07A5F", "#E8A33D"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    },
)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}  {path.stat().st_size / 1e3:.0f} kB")


def bare(ax, title: str = "") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title)


def pick(prints):
    """One print, its mate, and an unrelated finger, with ridge geometry.

    Chosen by how much of the frame the ridge mask covers, over the first
    `CANDIDATES` fingers, and deterministic given the cache.

    Neither of the two obvious selections works. The front of the cache is a
    fair-grade roll with a punch hole through the pattern. And NIST's own quality
    grade is not a proxy either -- it correlates about 0.56 with mask coverage
    here, and the top-graded finger in this cache is a faint roll on which
    `ridges.analyse` returns an empty mask, so every panel downstream of it comes
    out blank. Coverage is the property the figures actually need, so it is the
    property selected on.
    """
    if _PICKED:
        return _PICKED[0]

    candidates = [
        np.where(prints.finger == f)[0]
        for f in np.unique(prints.finger)[:CANDIDATES]
        if len(np.where(prints.finger == f)[0]) == 2
    ]
    scored = []
    for idx in candidates:
        pair = [
            (prints.images[int(i)], ridges.analyse(prints.images[int(i)])) for i in idx
        ]
        # Rank on the worse impression, so the finger chosen is good in *both* --
        # the minutiae figure shows the pair side by side and one blank panel
        # would ruin it.
        scored.append((min(a.mask.mean() for _, a in pair), int(idx[0]), pair))
    _, first, pair = max(scored, key=lambda row: row[0])

    other = int(np.where(prints.finger != prints.finger[first])[0][0])
    # Report the coverage of the print the panels actually show, not the pair's
    # minimum: the caption in the post quotes this line, and quoting the mate's
    # coverage next to the first impression's mask panel is a mismatch a reader
    # can see.
    shown = pair[0][1].mask.mean()
    print(
        f"  {prints.name[first]}, NIST grade {int(prints.quality[first])},"
        f" ridge mask over {shown:.0%} of the frame"
        f" (its mate: {pair[1][1].mask.mean():.0%})",
    )
    _PICKED.append(
        [*pair, (prints.images[other], ridges.analyse(prints.images[other]))],
    )
    return _PICKED[0]


# ---------------------------------------------------------------------------


def fig_ridge_geometry(prints) -> None:
    (img, an), _, _ = pick(prints)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))

    bare(axes[0], "the print, standardised")
    axes[0].imshow(img, cmap="gray")

    bare(axes[1], "orientation field")
    axes[1].imshow(img, cmap="gray", alpha=0.55)
    b, step = ridges.BLOCK, 2
    rows, cols = np.mgrid[0 : an.theta.shape[0] : step, 0 : an.theta.shape[1] : step]
    t = an.theta[::step, ::step]
    keep = an.mask[::step, ::step]
    length = b * 1.1
    axes[1].quiver(
        cols[keep] * b + b / 2,
        rows[keep] * b + b / 2,
        np.cos(t[keep]) * length,
        np.sin(t[keep]) * length,
        color=PURPLE,
        headwidth=1,
        headlength=0,
        headaxislength=0,
        pivot="middle",
        scale_units="xy",
        scale=1,
        width=0.004,
    )

    # Ridge-ness, not raw coherence. Coherence alone is speckled at block
    # resolution and reads as noise; this is the smoothed product the mask is
    # actually thresholded from, so the panel explains the panel beside it.
    bare(axes[2], "ridge strength")
    axes[2].imshow(an.ridgeness, cmap="magma")

    bare(axes[3], f"ridge mask, period {an.period:.1f} px")
    axes[3].imshow(an.mask, cmap="Greys_r")

    save(fig, "ridge-geometry.png")


def fig_enhance_steps(prints) -> None:
    (img, an), _, _ = pick(prints)
    _, extras = follow.extract(img, an, return_extras=True)

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    bare(axes[0], "raw 500 dpi scan")
    axes[0].imshow(img, cmap="gray")

    bare(axes[1], "Gabor-enhanced")
    axes[1].imshow(extras["enhanced"], cmap="gray")

    bare(axes[2], "ridge crests")
    axes[2].imshow(extras["crest"], cmap="Greys_r")

    bare(axes[3], "traced centrelines, gaps bridged")
    axes[3].imshow(extras["skeleton"], cmap="Greys_r")

    save(fig, "enhance-steps.png")


def _overlay(ax, img, found, colour, title) -> None:
    bare(ax, title)
    ax.imshow(img, cmap="gray")
    if len(found) == 0:
        return
    ends = found.kind == 1
    for sel, marker in ((ends, "o"), (~ends, "s")):
        if sel.any():
            ax.scatter(
                found.xy[sel, 0],
                found.xy[sel, 1],
                s=18,
                facecolors="none",
                edgecolors=colour,
                linewidths=0.9,
                marker=marker,
            )
    ax.quiver(
        found.xy[:, 0],
        found.xy[:, 1],
        np.cos(found.theta) * 9,
        np.sin(found.theta) * 9,
        color=colour,
        scale_units="xy",
        scale=1,
        width=0.005,
        headwidth=3,
    )


def fig_minutiae(prints) -> None:
    (a_img, a_an), (b_img, b_an), _ = pick(prints)
    old_a = minutiae.extract(a_img, a_an)
    new_a = follow.extract(a_img, a_an)
    new_b = follow.extract(b_img, b_an)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    _overlay(
        axes[0],
        a_img,
        old_a,
        CORAL,
        f"crossing number: {len(old_a)} landmarks",
    )
    _overlay(
        axes[1],
        a_img,
        new_a,
        PURPLE,
        f"ridge following: {len(new_a)} landmarks",
    )
    _overlay(
        axes[2],
        b_img,
        new_b,
        TEAL,
        f"the other impression: {len(new_b)}",
    )
    save(fig, "minutiae.png")


def fig_fingercode(prints) -> None:
    """Tessellation, descriptor, and the rotation the descriptor recovers.

    The third panel is the claim being demonstrated rather than asserted.
    Rolling the descriptor of a print that has been turned 90 degrees by four
    sectors scores 1.000 against the original, where comparing them unrolled
    scores 0.61 -- so "turning the finger is an array roll" is something the
    reader can see peak, not a sentence to take on trust.
    """
    (img, an), _, _ = pick(prints)
    code = enhance.fingercode(img, an)
    cell = enhance.tessellation(img.shape)
    turned = np.rot90(img)
    turned_code = enhance.fingercode(turned, ridges.analyse(turned))
    scores = np.array(
        [
            float(code.ravel() @ rolled.ravel())
            for rolled in enhance.rotations(
                turned_code,
                turned_code.shape[0],
                turned_code.shape[2],
            )
        ],
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    bare(axes[0], "4 bands x 16 sectors")
    axes[0].imshow(img, cmap="gray")
    axes[0].contour(
        np.where(cell < 0, np.nan, cell),
        levels=np.arange(-0.5, 64, 1),
        colors=GOLD,
        linewidths=0.4,
    )

    bare(axes[1], "FingerCode: 8 orientations x 64 cells")
    axes[1].imshow(code.reshape(code.shape[0], -1), cmap="magma", aspect="auto")
    axes[1].set_ylabel("orientation")

    ax = axes[2]
    degrees = np.arange(16) * 360 / 16
    ax.plot(degrees, scores, color=PURPLE, marker="o", markersize=3)
    peak = int(scores.argmax())
    ax.scatter(
        [degrees[peak]],
        [scores[peak]],
        s=70,
        facecolors="none",
        edgecolors=CORAL,
        zorder=3,
    )
    ax.annotate(
        f"{degrees[peak]:.0f}°, score {scores[peak]:.3f}",
        (degrees[peak], scores[peak]),
        textcoords="offset points",
        xytext=(6, -12),
        color=CORAL,
        fontsize=8,
    )
    ax.set_xlabel("rotation assumed when comparing (degrees)")
    ax.set_ylabel("match score")
    ax.set_title("the print, turned 90°, against the original")
    ax.grid(True, color="#D8DBE2", alpha=0.7)

    save(fig, "fingercode.png")


def fig_embedding() -> None:
    """Show the learned rung's two facts: it collapsed, and it still separates a little.

    An earlier version of this figure plotted raw cosine similarity, and
    matplotlib quietly moved the shared leading digits into an offset label --
    so a collapsed embedding, every pair within about 1e-5 of every other,
    looked like an ordinary histogram. Both panels here exist to stop that: the
    loss pinned at the margin is the collapse, and the x-axis is an explicit
    offset in millionths rather than an absolute scale that hides its own range.
    """
    dump, table = BENCH / "rungs.npz", BENCH / "ladder.json"
    if not dump.exists() or not table.exists():
        print(f"  skipped embedding.png: {dump} or {table} not written yet")
        return
    d = np.load(dump, allow_pickle=False)
    payload = json.loads(table.read_text())
    names = [str(n) for n in d["names"]]
    wanted = [i for i, n in enumerate(names) if n.startswith("3")]
    if not wanted:
        print("  skipped embedding.png: no learned rung in the dump")
        return
    i = wanted[0]
    genuine, impostor = d[f"genuine_{i}"], d[f"impostor_{i}"]
    row = payload["rows"][i]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))

    history = payload.get("loss_history", [])
    margin = payload.get("margin")
    ax = axes[0]
    if history:
        ax.plot(np.arange(1, len(history) + 1), history, color=PURPLE)
    if margin is not None:
        ax.axhline(margin, color=CORAL, linestyle="--", linewidth=1)
        ax.annotate(
            f"margin {margin:g}: every distance equal",
            (len(history) * 0.45, margin),
            textcoords="offset points",
            xytext=(0, 7),
            color=CORAL,
            fontsize=8,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("batch-hard triplet loss")
    ax.set_title("training settles at the margin, which is collapse")
    ax.grid(True, color="#D8DBE2", alpha=0.7)

    ax = axes[1]
    centre = float(np.median(impostor))
    g = (genuine - centre) * 1e6
    im = (impostor - centre) * 1e6
    bins = np.linspace(min(g.min(), im.min()), max(g.max(), im.max()), 60)
    ax.hist(
        im,
        bins=bins,
        color=INK,
        alpha=0.55,
        density=True,
        label="different fingers",
    )
    ax.hist(g, bins=bins, color=PURPLE, alpha=0.75, density=True, label="same finger")
    ax.set_xlabel(
        "cosine similarity, offset from the impostor median" r" ($\times 10^{-6}$)",
    )
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8)
    separation = row["d'"]
    ax.set_title(f"d' = {separation:.2f} in the residual directions")

    save(fig, "embedding.png")


def fig_cmc() -> None:
    table = BENCH / "ladder.json"
    if not table.exists():
        print(f"  skipped cmc.png: {table} not written yet")
        return
    payload = json.loads(table.read_text())
    curves = payload.get("cmc", {})
    if not curves:
        print("  skipped cmc.png: no cmc block in the table")
        return

    colours = [INK, "#7A8290", TEAL, CORAL, GOLD, PURPLE]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for (name, values), colour in zip(curves.items(), colours):
        k = np.arange(1, len(values) + 1)
        ax.plot(
            k,
            np.array(values) * 100,
            color=colour,
            marker="o",
            markersize=2.5,
            label=name,
        )
    ax.set_xlabel("rank k")
    ax.set_ylabel("true mate within the top k (%)")
    ax.set_title(
        f"gallery of {payload.get('gallery', '?')} fingers,"
        f" {payload.get('gallery', '?')} probes"
        f" (of {payload['fingers']} in the cache; the rest trained the network)",
    )
    ax.grid(True, color="#D8DBE2", alpha=0.7)
    # Under the axes, in two columns. Six rising curves leave no empty corner:
    # upper-left sits on the leader and lower-right sits on the four that cluster
    # at the bottom, which is where the interesting comparison actually is.
    ax.legend(
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    save(fig, "cmc.png")


FIGURE_SET: dict[str, Callable] = {
    "ridge-geometry": fig_ridge_geometry,
    "enhance-steps": fig_enhance_steps,
    "minutiae": fig_minutiae,
    "fingercode": fig_fingercode,
    "embedding": fig_embedding,
    "cmc": fig_cmc,
}

NEEDS_PRINTS = {"ridge-geometry", "enhance-steps", "minutiae", "fingercode"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/prints.npz"))
    ap.add_argument("--only", choices=sorted(FIGURE_SET), action="append")
    args = ap.parse_args()

    wanted = args.only or list(FIGURE_SET)
    prints = data.load(args.data) if set(wanted) & NEEDS_PRINTS else None

    for name in wanted:
        print(f"{name}:")
        fn = FIGURE_SET[name]
        fn(prints) if name in NEEDS_PRINTS else fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
