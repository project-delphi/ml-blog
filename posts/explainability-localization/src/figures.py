"""Every figure in the post, as a function returning a Matplotlib figure.

The post calls these functions inside code cells; ``make_figures.py`` calls the
same functions to write ``figures/*.png``. There is one implementation, so the
committed PNGs and the rendered page cannot disagree.
"""

from __future__ import annotations

import attribution as A
import house as H
import iris_stats as I
import mnist_model as M
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Part I — MNIST
# --------------------------------------------------------------------------


def fig_examples(t: M.Trained, idx: dict[str, int]) -> plt.Figure:
    """The two running examples and the model's probability over the classes.

    Args:
        t: The trained bundle.
        idx: Mapping from ``"wrong"``/``"right"`` to test-set indices.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.6), gridspec_kw={"width_ratios": [1, 2.3]})
    y = t.yte.numpy()
    for row, key in enumerate(["wrong", "right"]):
        i = idx[key]
        H.digit_axis(axes[row, 0], t.xte[i, 0].numpy())
        axes[row, 0].set_title(f"true {y[i]}, predicted {t.pred[i]}", fontsize=10)

        ax = axes[row, 1]
        probs = np.clip(t.prob[i], 1e-6, 1.0)
        colours = [H.PLUM if c == t.pred[i] else H.GRID for c in range(10)]
        colours[y[i]] = H.BLUE if t.pred[i] != y[i] else H.PLUM
        ax.bar(range(10), probs, color=colours)
        # Log scale: on a linear axis both rows are one bar at 1.0 and nine
        # invisible ones, which hides how far the runner-up actually is.
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 3.0)
        ax.set_xticks(range(10))
        ax.set_ylabel("probability")
        runner = np.argsort(probs)[-2]
        ax.set_title(
            f"p({t.pred[i]}) = {t.prob[i].max():.3f},   "
            f"p({runner}) = {t.prob[i][runner]:.1e}",
            fontsize=10,
        )
    axes[0, 1].legend(
        handles=[
            Line2D([], [], color=H.PLUM, lw=6, label="predicted"),
            Line2D([], [], color=H.BLUE, lw=6, label="true class"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    fig.suptitle("Two test digits: one confident mistake, one confident success", fontsize=11)
    fig.tight_layout()
    return fig


def fig_method(
    t: M.Trained, idx: dict[str, int], compute, title: str, signed: bool, note: str
) -> plt.Figure:
    """One attribution method, both running examples side by side.

    Args:
        t: The trained bundle.
        idx: Running-example indices.
        compute: Callable ``(model, image, target) -> 28x28 map``.
        title: Method name.
        signed: Whether the map has meaningful sign.
        note: Caption line under the panels.

    Returns:
        The figure.
    """
    y = t.yte.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.7))
    for col, key in enumerate(["wrong", "right"]):
        i = idx[key]
        heat = compute(t.model, t.xte[i], int(t.pred[i]))
        label = (
            f"the mistake: a {y[i]} read as a {t.pred[i]}"
            if key == "wrong"
            else f"the control: a {y[i]} read as a {t.pred[i]}"
        )
        H.overlay_axis(axes[col], t.xte[i, 0].numpy(), heat, label, signed=signed)
    fig.suptitle(title, fontsize=11, y=0.99)
    fig.text(0.5, 0.035, note, ha="center", fontsize=9, color=H.MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    return fig


def fig_sanity(t: M.Trained, idx: dict[str, int], seed: int) -> plt.Figure:
    """Model-randomisation check: the same maps from an untrained network.

    Args:
        t: The trained bundle.
        idx: Running-example indices.
        seed: Seed for the re-initialised network.

    Returns:
        The figure.
    """
    i = idx["wrong"]
    img = t.xte[i, 0].numpy()
    target = int(t.pred[i])
    fresh = A.randomised_copy(t.model, seed)

    panels = [
        ("saliency", lambda m: np.abs(A.saliency(m, t.xte[i], target)), False),
        ("Grad-CAM", lambda m: A.grad_cam(m, t.xte[i], target), False),
        (
            "integrated gradients",
            lambda m: A.integrated_gradients(m, t.xte[i], target),
            True,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.8, 5.4))
    for col, (name, compute, signed) in enumerate(panels):
        trained_map = compute(t.model)
        random_map = compute(fresh)
        # Both correlations are reported because they say different things. The
        # magnitude correlation asks whether the two methods highlight the same
        # pixels; the signed one asks whether they agree on which side those
        # pixels argue for. A method can pass one and fail the other.
        rho_mag = A.rank_correlation(np.abs(trained_map), np.abs(random_map))
        rho_signed = A.rank_correlation(trained_map, random_map)
        # For a map that is non-negative by construction the two coincide, so
        # printing both would only imply a distinction that does not exist here.
        caption = (
            f"randomised weights\n|rho| {rho_mag:+.2f}   signed rho {rho_signed:+.2f}"
            if signed
            else f"randomised weights\nrho {rho_mag:+.2f}"
        )
        H.overlay_axis(axes[0, col], img, trained_map, f"{name}\ntrained", signed=signed)
        H.overlay_axis(axes[1, col], img, random_map, caption, signed=signed)
        axes[1, col].title.set_fontsize(9)
    fig.suptitle(
        "The same three methods on a network that has learned nothing", fontsize=11
    )
    fig.tight_layout()
    return fig


def fig_agreement(t: M.Trained, idx: dict[str, int]) -> plt.Figure:
    """All four maps for the misclassified digit, plus their rank correlations.

    Args:
        t: The trained bundle.
        idx: Running-example indices.

    Returns:
        The figure.
    """
    i = idx["wrong"]
    img = t.xte[i, 0].numpy()
    target = int(t.pred[i])
    maps = {
        "saliency": np.abs(A.saliency(t.model, t.xte[i], target)),
        "occlusion": A.occlusion(t.model, t.xte[i], target),
        "Grad-CAM": A.grad_cam(t.model, t.xte[i], target),
        "int. gradients": A.integrated_gradients(t.model, t.xte[i], target),
    }
    names = list(maps)
    # Ranked on magnitude, so a method that only reports magnitudes (saliency,
    # Grad-CAM) is compared on the same footing as one that reports a sign.
    rho = np.array(
        [
            [A.rank_correlation(np.abs(maps[a]), np.abs(maps[b])) for b in names]
            for a in names
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), gridspec_kw={"width_ratios": [2.1, 1]})
    axes[0].axis("off")
    axes[1].axis("off")

    inner = axes[0].get_subplotspec().subgridspec(2, 2, wspace=0.08, hspace=0.30)
    for k, name in enumerate(names):
        ax = fig.add_subplot(inner[k // 2, k % 2])
        H.overlay_axis(ax, img, maps[name], name, signed=name in ("occlusion", "int. gradients"))
        ax.title.set_fontsize(10)

    ax = fig.add_subplot(axes[1].get_subplotspec())
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    short = [n.split()[0].replace("int.", "IG") for n in names]
    ax.set_xticks(range(4), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(4), short, fontsize=8)
    for a in range(4):
        for b in range(4):
            ax.text(
                b,
                a,
                f"{rho[a, b]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(rho[a, b]) > 0.55 else H.INK,
            )
    ax.set_title("rank correlation\nbetween magnitudes", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
    fig.suptitle(
        f"Four localisations of the same wrong answer (predicted {target})", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------
# Part II — Iris
# --------------------------------------------------------------------------


def fig_coefficients(fit: I.LogitFit) -> plt.Figure:
    """Coefficient estimates with 95% Wald intervals.

    Args:
        fit: A fitted binary logistic regression.

    Returns:
        The figure.
    """
    table = fit.table().iloc[1:]  # drop the intercept
    lo, hi = table["lo"].to_numpy(), table["hi"].to_numpy()
    y = np.arange(len(table))[::-1]

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.axvline(0, color=H.MUTED, lw=1, ls="--", zorder=0)
    crosses_zero = (lo < 0) & (hi > 0)
    for k in range(len(table)):
        colour = H.MUTED if crosses_zero[k] else H.SLATE
        ax.plot([lo[k], hi[k]], [y[k], y[k]], color=colour, lw=2.4, solid_capstyle="round")
        ax.plot(table["coef"].iloc[k], y[k], "o", color=colour, ms=7)
    ax.set_yticks(y, table["term"])
    ax.set_xlabel("log-odds of virginica per standard deviation\n(grey: 95% interval contains zero)")
    ax.set_ylim(-0.7, len(table) - 0.3)
    ax.set_title("Which measurement separates versicolor from virginica?")
    fig.tight_layout()
    return fig


def fig_separation(path: pd.DataFrame) -> plt.Figure:
    """Coefficient norm and largest standard error against the Newton budget.

    Args:
        path: Output of :func:`iris_stats.separation_path`.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(
        path["newton steps"],
        path["coefficient norm"],
        "o-",
        color=H.SLATE,
        label="coefficient norm",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Newton steps allowed")
    ax.set_ylabel("coefficient norm", color=H.SLATE)
    ax.tick_params(axis="y", colors=H.SLATE)

    ax2 = ax.twinx()
    ax2.plot(
        path["newton steps"],
        path["largest std err"],
        "s--",
        color=H.PLUM,
        label="largest standard error",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("largest standard error", color=H.PLUM)
    ax2.tick_params(axis="y", colors=H.PLUM)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(H.PLUM)

    ax.set_title("Setosa is separable, so the estimate never settles")
    ax.legend(
        handles=[
            Line2D([], [], color=H.SLATE, marker="o", label="coefficient norm"),
            Line2D([], [], color=H.PLUM, marker="s", ls="--", label="largest standard error"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    fig.tight_layout()
    return fig


def fig_anova(df: pd.DataFrame, table: pd.DataFrame) -> plt.Figure:
    """Per-feature distributions by species alongside their variance shares.

    Args:
        df: The Iris frame.
        table: Output of :func:`iris_stats.anova_table`.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    for row, feature in enumerate(I.FEATURES):
        for species in I.SPECIES:
            values = df.loc[df.species == species, feature]
            ax.scatter(
                values,
                np.full(len(values), row) + np.linspace(-0.16, 0.16, len(values)),
                s=7,
                alpha=0.65,
                color=H.SPECIES_COLORS[species],
                linewidths=0,
            )
    ax.set_yticks(range(len(I.FEATURES)), I.FEATURES)
    ax.invert_yaxis()
    ax.set_xlabel("cm")
    ax.set_title("Where the species separate")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=c, label=s)
            for s, c in H.SPECIES_COLORS.items()
        ],
        frameon=False,
        fontsize=8,
        loc="lower right",
    )

    ax = axes[1]
    order = np.arange(len(table))[::-1]
    ax.barh(order, table["eta^2"], color=H.SLATE, height=0.6)
    for k, value in enumerate(table["eta^2"]):
        ax.text(value - 0.02, order[k], f"{value:.2f}", va="center", ha="right",
                color="white", fontsize=9)
    ax.set_yticks(order, table["feature"])
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$\eta^2$: share of variance between species")
    ax.set_title(rf"Shares sum to {table['eta^2'].sum():.2f}, not 1")
    fig.tight_layout()
    return fig


def fig_pca(X: np.ndarray, target: np.ndarray) -> plt.Figure:
    """Scree plot for both scalings, and the standardised biplot.

    Args:
        X: The four measurements in cm, shape (n, 4).
        target: Integer species codes.

    Returns:
        The figure.
    """
    std = I.pca_fit(X, standardised=True)
    raw = I.pca_fit(X, standardised=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    w = 0.38
    pcs = np.arange(4)
    ax.bar(pcs - w / 2, std.ratio, w, color=H.SLATE, label="standardised")
    ax.bar(pcs + w / 2, raw.ratio, w, color=H.OCHRE, label="raw cm")
    ax.set_xticks(pcs, [f"PC{k + 1}" for k in pcs])
    ax.set_ylabel("share of total variance")
    ax.set_title("Scaling changes the answer")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for code, species in enumerate(I.SPECIES):
        sel = target == code
        ax.scatter(
            std.scores[sel, 0],
            std.scores[sel, 1],
            s=16,
            alpha=0.75,
            color=H.SPECIES_COLORS[species],
            linewidths=0,
            label=species,
        )
    scale = 2.6
    # Petal length and petal width load almost identically (they correlate at
    # 0.96), so their arrows are near-collinear and their labels collide. Push
    # each label out far enough to clear the one before it, in angle order.
    angles = np.arctan2(std.loadings[:, 1], std.loadings[:, 0])
    order = np.argsort(angles)
    perp = np.zeros(len(I.FEATURES))
    sign = 1.0
    for prev, curr in zip(order, order[1:]):
        if abs(angles[curr] - angles[prev]) < np.deg2rad(25):
            perp[curr] = perp[prev] + sign * 0.42
            sign = -sign
    for k, name in enumerate(I.FEATURES):
        vx, vy = std.loadings[k] * scale
        ax.annotate(
            "",
            xy=(vx, vy),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=H.INK, lw=1.2),
        )
        # Offset the label along the arrow and then perpendicular to it, so a
        # pair of near-parallel loadings does not stack two labels in one spot.
        nx, ny = -np.sin(angles[k]), np.cos(angles[k])
        ax.text(
            vx * 1.13 + nx * perp[k],
            vy * 1.13 + ny * perp[k],
            name,
            fontsize=8,
            ha="center",
            va="center",
            color=H.INK,
            bbox=dict(facecolor=H.SURFACE, edgecolor="none", pad=1.0, alpha=0.85),
        )
    ax.set_xlabel(f"PC1 ({std.ratio[0]:.0%})")
    ax.set_ylabel(f"PC2 ({std.ratio[1]:.0%})")
    ax.set_title("Biplot: loadings over scores, standardised")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def fig_permutation(perm: pd.DataFrame, anova: pd.DataFrame, coef: pd.DataFrame) -> plt.Figure:
    """Permutation importance with error bars, against the two classical scores.

    Args:
        perm: Output of :func:`iris_stats.permutation_importance`.
        anova: Output of :func:`iris_stats.anova_table`.
        coef: Coefficient table from a fitted logistic regression.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    order = np.arange(len(perm))[::-1]
    ax.barh(order, perm["mean drop"], xerr=1.96 * perm["se"], color=H.PLUM, height=0.6,
            error_kw=dict(ecolor=H.INK, lw=1.1, capsize=3))
    ax.set_yticks(order, perm["feature"])
    ax.set_xlabel("held-out accuracy lost when the column is shuffled")
    ax.set_title("Permutation importance, 200 shuffles")

    ax = axes[1]
    scores = pd.DataFrame(
        {
            "feature": perm["feature"],
            "permutation": perm["mean drop"] / perm["mean drop"].max(),
            r"ANOVA $\eta^2$": anova["eta^2"].to_numpy() / anova["eta^2"].max(),
            "|coefficient|": (
                coef["coef"].abs().to_numpy() / coef["coef"].abs().max()
            ),
        }
    )
    w = 0.26
    pos = np.arange(len(scores))
    for k, (name, colour) in enumerate(
        [("permutation", H.PLUM), (r"ANOVA $\eta^2$", H.SLATE), ("|coefficient|", H.OCHRE)]
    ):
        ax.bar(pos + (k - 1) * w, scores[name], w, color=colour, label=name)
    ax.set_xticks(pos, [f.replace(" ", "\n") for f in scores["feature"]], fontsize=8)
    ax.set_ylabel("score, scaled to its own maximum")
    ax.set_title("Three localisations, one dataset")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
