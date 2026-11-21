"""Make the post's static figure (and cover): SVGD and unadjusted Langevin (ULA) on
the banana target after 500 iterations with n = 100, and their IMQ kernel Stein
discrepancies against score evaluations.

An independent numpy implementation of model.js, so the figure doubles as a
cross-check of the widget. Run from the repo root:

    uv run --no-project --with numpy --with matplotlib \
        python posts/kernel-stein-discrepancy-and-svgd/src/make_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures" / "svgd-vs-langevin.png"
INK, MUTED, RULE = "#1F2430", "#5F6672", "#D8DBE2"
PURPLE, TEAL, AMBER = "#4A3AA7", "#1D6E6E", "#9A5B00"
S1, S2, B, C = 1.5, 0.6, 0.6, 1.5  # banana: x ~ N(0, S1^2), y | x ~ N(B x^2 - C, S2^2)


def logp(x, y):
    u = y - (B * x**2 - C)
    return -0.5 * x**2 / S1**2 - 0.5 * u**2 / S2**2


def score(P):
    x, y = P[:, 0], P[:, 1]
    u = y - (B * x**2 - C)
    return np.stack([-x / S1**2 + (u / S2**2) * 2 * B * x, -u / S2**2], axis=1)


def median_ell(P):
    n = len(P)
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))[np.triu_indices(n, 1)]
    return max(1e-3, np.sort(d)[len(d) // 2] / np.sqrt(2 * np.log(n)))


def svgd_step(P, eps):
    n, ell = len(P), median_ell(P)
    diff = P[:, None, :] - P[None, :, :]  # diff[i, j] = x_i - x_j
    K = np.exp(-0.5 * (diff**2).sum(-1) / ell**2)
    phi = (K @ score(P) + (K[:, :, None] * diff).sum(1) / ell**2) / n
    return P + eps * phi


def ksd_imq(P, c=1.0, beta=-0.5):
    S = score(P)
    r = P[:, None, :] - P[None, :, :]
    r2 = (r**2).sum(-1)
    q = c**2 + r2
    d = P.shape[1]
    kp = (
        (S @ S.T) * q**beta
        - 2 * beta * q ** (beta - 1) * np.einsum("id,ijd->ij", S, r)
        + 2 * beta * q ** (beta - 1) * np.einsum("jd,ijd->ij", S, r)
        - 4 * beta * (beta - 1) * q ** (beta - 2) * r2
        - 2 * beta * d * q ** (beta - 1)
    )
    return np.sqrt(max(kp.mean(), 0.0))


rng = np.random.default_rng(20261111)
n, eps, steps = 100, 0.05, 500
A = rng.standard_normal((n, 2))
Bp = A.copy()
evals, ka, kb = [0], [ksd_imq(A)], [ksd_imq(Bp)]
for t in range(1, steps + 1):
    A = svgd_step(A, eps)
    Bp = Bp + eps * score(Bp) + np.sqrt(2 * eps) * rng.standard_normal(Bp.shape)
    if t % 5 == 0:
        evals.append(t * n)
        ka.append(ksd_imq(A))
        kb.append(ksd_imq(Bp))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, (ax0, ax1) = plt.subplots(
    1, 2, figsize=(12, 6.3), dpi=100, gridspec_kw={"width_ratios": [1, 1.15]}
)
gx, gy = np.meshgrid(np.linspace(-5, 5, 300), np.linspace(-3, 7, 300))
dens = np.exp(logp(gx, gy) - logp(gx, gy).max())
ax0.contourf(
    gx, gy, dens, levels=[0.05, 0.25, 0.5, 0.75, 1.01], colors=[TEAL], alpha=0.12
)
ax0.contour(
    gx, gy, dens, levels=[0.05, 0.25, 0.5, 0.75], colors=[MUTED], linewidths=0.8
)
ax0.scatter(
    Bp[:, 0],
    Bp[:, 1],
    s=26,
    facecolors="none",
    edgecolors=AMBER,
    linewidths=1.3,
    label="Langevin chains (ULA)",
)
ax0.scatter(A[:, 0], A[:, 1], s=22, color=PURPLE, label="SVGD particles")
ax0.set_xlim(-5, 5)
ax0.set_ylim(-3, 7)
ax0.set_aspect("equal")
ax0.set_title("Banana target, n = 100, after 500 iterations", color=INK, fontsize=14)
ax0.legend(frameon=False, loc="upper center", fontsize=12)
ax1.plot(evals, ka, color=PURPLE, lw=2.4, label="SVGD")
ax1.plot(evals, kb, color=AMBER, lw=2.2, ls=(0, (6, 3)), label="Langevin (ULA)")
ax1.set_yscale("log")
ax1.set_xlabel("score evaluations", color=INK)
ax1.set_ylabel("IMQ kernel Stein discrepancy", color=INK)
ax1.set_title("Equal score evaluations, step size 0.05", color=INK, fontsize=14)
ax1.legend(frameon=False, fontsize=12)
for ax in (ax0, ax1):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK)
ax1.grid(color=RULE, lw=0.8)
fig.tight_layout(pad=1.6)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=100, facecolor="white")
print(f"wrote {OUT}; final KSD: SVGD {ka[-1]:.3f}, ULA {kb[-1]:.3f}; start {ka[0]:.3f}")
