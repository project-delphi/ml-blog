"""Make the post's static figure (and cover): SURE against the true loss for
soft-threshold denoising of a synthetic sparse signal, for one noise draw and
averaged over 500 draws.

An independent numpy implementation of model.js (the signal itself is drawn
separately, with the same recipe). Run from the repo root:

    uv run --no-project --with numpy --with matplotlib \
        python posts/stein-and-unnormalised-models/src/make_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures" / "sure-vs-loss.png"
INK, MUTED, RULE = "#1F2430", "#5F6672", "#D8DBE2"
PURPLE, AMBER = "#4A3AA7", "#9A5B00"

rng = np.random.default_rng(20261111)
d, k, sigma = 256, 25, 1.0
mu = np.zeros(d)
idx = rng.choice(d, k, replace=False)
mu[idx] = rng.choice([-1, 1], k) * rng.uniform(2, 6, k)
lams = np.linspace(0, 4 * sigma, 241)


def soft(y, lam):
    return np.sign(y) * np.maximum(np.abs(y) - lam, 0)


def sure(y, lam):
    return (
        d * sigma**2
        - 2 * sigma**2 * np.sum(np.abs(y) <= lam)
        + np.sum(np.minimum(y**2, lam**2))
    )


def loss(y, lam):
    return np.sum((soft(y, lam) - mu) ** 2)


y = mu + sigma * rng.standard_normal(d)
S1 = np.array([sure(y, lam) for lam in lams])
L1 = np.array([loss(y, lam) for lam in lams])
draws = mu + sigma * rng.standard_normal((500, d))
S = np.array([[sure(yy, lam) for lam in lams] for yy in draws]).mean(0)
L = np.array([[loss(yy, lam) for lam in lams] for yy in draws]).mean(0)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), dpi=100, sharey=True)
for ax, s, loss_c, title in [
    (axes[0], S1, L1, "One noisy observation"),
    (axes[1], S, L, "Averaged over 500 noise draws"),
]:
    ax.plot(
        lams,
        loss_c,
        color=AMBER,
        lw=2.4,
        ls=(0, (6, 3)),
        label=r"true loss $\|\hat\mu_\lambda(y)-\mu\|^2$ (needs $\mu$)",
    )
    ax.plot(lams, s, color=PURPLE, lw=2.4, label=r"SURE$(\lambda)$ (from $y$ alone)")
    ax.plot(lams[np.argmin(s)], s.min(), "o", mfc="none", mec=PURPLE, ms=11, mew=2)
    ax.plot(
        lams[np.argmin(loss_c)], loss_c.min(), "o", mfc="none", mec=AMBER, ms=11, mew=2
    )
    ax.set_title(title, color=INK, fontsize=14)
    ax.set_xlabel(r"soft threshold $\lambda$ (noise $\sigma = 1$)", color=INK)
    ax.grid(color=RULE, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK)
axes[0].set_ylabel("squared error over 256 coordinates", color=INK)
axes[0].legend(frameon=False, loc="upper center", fontsize=12)
fig.tight_layout(pad=1.6)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=100, facecolor="white")
print(
    f"wrote {OUT}; one draw: lam_SURE {lams[np.argmin(S1)]:.2f}, lam_oracle {lams[np.argmin(L1)]:.2f}; "
    f"averaged: max |mean SURE - mean loss| = {np.max(np.abs(S - L)):.2f} on a scale of {L.max():.0f}"
)
