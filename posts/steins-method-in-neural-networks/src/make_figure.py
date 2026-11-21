"""Make the post's static figure (and cover): the exact Wasserstein distance between a
random one-hidden-layer ReLU network's output and N(0, 1) against width, with Part 2's
Theorem 4 bound and this post's Corollary 2 bound; and n d_W / kappa for four
activations, which all approach (e^{-3/2} + 1/4) / sqrt(2 pi).

Reads the exact distances from widget-data/data.js (written by src/make_data.py) and
recomputes the bound constants by quadrature here. Run from the repo root:

    uv run --no-project --with numpy --with scipy --with matplotlib \
        python posts/steins-method-in-neural-networks/src/make_figure.py
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import quad

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "figures" / "width-gaussianity.png"
INK, MUTED, RULE = "#1F2430", "#5F6672", "#D8DBE2"
PURPLE, TEAL, AMBER, PINK = "#4A3AA7", "#1D6E6E", "#9A5B00", "#A8437A"

src = (HERE / "widget-data" / "data.js").read_text()
D = json.loads(re.search(r"netGaussianity = (\{.*\});", src).group(1))
n = np.array(D["n"], dtype=float)
ACTS = {
    "linear": lambda g: g,
    "relu": lambda g: np.maximum(g, 0),
    "gelu": lambda g: g * stats.norm.cdf(g),
    "tanh": np.tanh,
}
SQ2PI = np.sqrt(2 / np.pi)


def gmom(f):
    pdf = stats.norm.pdf
    return (
        quad(lambda g: f(g) * pdf(g), -12, 0, limit=400)[0]
        + quad(lambda g: f(g) * pdf(g), 0, 12, limit=400)[0]
    )


def consts(a):
    s = ACTS[a]
    c2, c3, c4 = (
        gmom(lambda g: s(g) ** 2),
        gmom(lambda g: np.abs(s(g)) ** 3),
        gmom(lambda g: s(g) ** 4),
    )
    kappa = c4 / c2**2 - 1
    thm4 = 2 * SQ2PI * c3 / c2**1.5 + SQ2PI * np.sqrt(3 * c4 / c2**2)
    return kappa, thm4


LIMIT = (np.exp(-1.5) + 0.25) / np.sqrt(2 * np.pi)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), dpi=100)
ax = axes[0]
kappa, thm4 = consts("relu")
dW = np.array(D["acts"]["relu"]["dW"])
ax.plot(
    n,
    thm4 / np.sqrt(n),
    color=AMBER,
    lw=2.2,
    ls=(0, (6, 3)),
    label=r"Part 2, Theorem 4: $6.99/\sqrt{n}$",
)
mix = np.minimum(2 * SQ2PI * kappa / n, SQ2PI * np.sqrt(kappa / n))
ax.plot(
    n,
    mix,
    color=TEAL,
    lw=2.2,
    ls=(0, (6, 3)),
    label=r"Corollary 2: $\min(7.98/n,\ 1.78/\sqrt{n})$",
)
ax.plot(n, dW, color=PURPLE, lw=2.6, marker="o", ms=4, label="exact distance")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("width n (log scale)", color=INK)
ax.set_ylabel(r"$d_W$(network output, N(0, 1))", color=INK)
ax.set_title("ReLU network at one input", color=INK, fontsize=14)
ax.legend(frameon=False, fontsize=11, loc="lower left")
ax = axes[1]
for a, col in zip(ACTS, [MUTED, PURPLE, PINK, TEAL]):
    k, _ = consts(a)
    ax.plot(
        n,
        n * np.array(D["acts"][a]["dW"]) / k,
        color=col,
        lw=2.4,
        label=f"{a} (κ = {k:.2f})",
    )
ax.axhline(
    LIMIT,
    color=INK,
    lw=1.4,
    ls=(0, (2, 3)),
    label=r"$(e^{-3/2}+1/4)/\sqrt{2\pi} = 0.189$",
)
ax.set_xscale("log")
ax.set_xlabel("width n (log scale)", color=INK)
ax.set_ylabel(r"$n\, d_W / \kappa$", color=INK)
ax.set_title(
    r"Every activation: $d_W \approx 0.189\,\kappa / n$", color=INK, fontsize=14
)
ax.legend(frameon=False, fontsize=11, loc="lower right")
for ax in axes:
    ax.grid(color=RULE, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK)
fig.tight_layout(pad=1.6)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=100, facecolor="white")
print(
    f"wrote {OUT}; ReLU kappa {kappa:.4f}, Theorem 4 constant {thm4:.4f}; limit {LIMIT:.5f}"
)
for a in ACTS:
    k, _ = consts(a)
    print(
        f"  {a}: n dW / kappa at n = 1, 10, 100, 1000:",
        [
            round(float(v), 4)
            for v in (n * np.array(D["acts"][a]["dW"]) / k)[[0, 7, 17, 27]]
        ],
    )
