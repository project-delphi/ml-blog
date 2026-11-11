"""Make the post's static figure (and cover): running Monte Carlo means of the
Gaussian and Langevin Stein operators applied to f(x) = x^3, for four laws
standardised to mean 0 and variance 1.

This is an independent numpy implementation of the widget's model, so the figure
doubles as a cross-check of model.js. Run from the repo root:

    uv run --no-project --with numpy --with matplotlib python posts/steins-lemma/src/make_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures" / "stein-paths.png"
INK, MUTED, RULE = "#1F2430", "#5F6672", "#D8DBE2"
PURPLE, TEAL, AMBER, PINK = "#4A3AA7", "#1D6E6E", "#9A5B00", "#A8437A"

N, NU = 100_000, 10.0
SQ3, B = np.sqrt(3.0), 1 / np.sqrt(2.0)
rng = np.random.default_rng(20261111)

laws = {
    "Gaussian": (rng.standard_normal(N), lambda x: -x),
    "Laplace": (rng.laplace(0.0, B, N), lambda x: -np.sign(x) / B),
    f"Student-t, ν = {NU:g}": (
        rng.standard_t(NU, N) * np.sqrt((NU - 2) / NU),
        lambda x: -(NU + 1) * x / (NU - 2 + x**2),
    ),
    "Uniform": (rng.uniform(-SQ3, SQ3, N), lambda x: np.zeros_like(x)),
}
colours = [PURPLE, TEAL, AMBER, PINK]
# exact values: E[3X^2 - X^4] = 3 - E X^4 under the Gaussian operator;
# 0 under the Langevin operator except the uniform's boundary term (f(√3) - f(-√3)) / (2√3) = 3
exact_gauss = [0.0, -3.0, 3 - (3 + 6 / (NU - 4)), 3 - 9 / 5]
exact_lang = [0.0, 0.0, 0.0, 3.0]

n = np.arange(1, N + 1)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), dpi=100, sharey=True)
fig.patch.set_facecolor("white")
for ax, title, op, exacts in [
    (axes[0], "Gaussian operator  f′(x) − x f(x)", "gauss", exact_gauss),
    (axes[1], "Langevin operator  f′(x) + sₚ(x) f(x)", "langevin", exact_lang),
]:
    for (name, (x, score)), c, e in zip(laws.items(), colours, exacts):
        v = 3 * x**2 + (-x if op == "gauss" else score(x)) * x**3
        run = np.cumsum(v) / n
        ax.plot(n[9:], run[9:], color=c, lw=2, label=name)
        ax.axhline(e, color=c, lw=1, ls=(0, (5, 4)), alpha=0.8)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(10, N)
    ax.set_ylim(-4.5, 4.5)
    ax.set_title(title, color=INK, fontsize=14, pad=10)
    ax.set_xlabel("sample size n", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color=RULE, lw=0.8)
axes[0].set_ylabel("running mean, f(x) = x³", color=INK)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4, fontsize=12)
fig.tight_layout(pad=1.6, rect=(0, 0.07, 1, 1))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=100, facecolor="white")
print(f"wrote {OUT}")
for (name, (x, score)), eg, el in zip(laws.items(), exact_gauss, exact_lang):
    vg = 3 * x**2 - x**4
    vl = 3 * x**2 + score(x) * x**3
    print(
        f"{name:22s} gauss {vg.mean():+.3f} (exact {eg:+.3f})   langevin {vl.mean():+.3f} (exact {el:+.3f})"
    )
