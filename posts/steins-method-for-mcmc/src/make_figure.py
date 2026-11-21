"""Make the post's static figure (and cover): unadjusted Langevin's bias divided by the
step size, converging to Stein's first-order coefficient, and the error of an N-step
average against step size for ULA and MALA at three budgets.

An independent numpy implementation of model.js: the ULA and MALA kernels are
discretised on a 481-point grid (model.js uses 241) and solved with numpy.linalg.
Run from the repo root:

    uv run --no-project --with numpy --with matplotlib \
        python posts/steins-method-for-mcmc/src/make_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures" / "ula-bias-and-budget.png"
INK, MUTED, RULE = "#1F2430", "#5F6672", "#D8DBE2"
PURPLE, TEAL, AMBER = "#4A3AA7", "#1D6E6E", "#9A5B00"

W1, M1, S1, W2, M2, S2 = 0.6, -1.2, 0.6, 0.4, 1.3, 0.9


def mix(x):
    a = W1 / S1 * np.exp(-0.5 * ((x - M1) / S1) ** 2)
    b = W2 / S2 * np.exp(-0.5 * ((x - M2) / S2) ** 2)
    sa, sb = -(x - M1) / S1**2, -(x - M2) / S2**2
    s = (a * sa + b * sb) / (a + b)
    ds = (a * (sa**2 - 1 / S1**2) + b * (sb**2 - 1 / S2**2)) / (a + b) - s**2
    return np.log(a + b), s, ds


TARGETS = {
    "Gaussian": (
        (-7, 7),
        lambda x: (-0.5 * x**2, -x, -np.ones_like(x)),
    ),
    "Heavier tails": (
        (-18, 18),
        lambda x: (
            -np.sqrt(1 + x**2),
            -x / np.sqrt(1 + x**2),
            -((1 + x**2) ** -1.5),
        ),
    ),
    "Two modes": ((-7, 7), mix),
}
NG = 481


def kernels(lohi, fn, eps):
    x = np.linspace(*lohi, NG)
    dx = x[1] - x[0]
    lp, s, ds = fn(x)
    v = 2 * eps
    lq = -((x[None, :] - (x + eps * s)[:, None]) ** 2) / (2 * v)
    q = np.exp(lq) * dx / np.sqrt(2 * np.pi * v)
    ula = q / q.sum(1, keepdims=True)
    la = (lp[None, :] + lq.T) - (lp[:, None] + lq)
    mala = q * np.minimum(1, np.exp(np.minimum(la, 0)))
    np.fill_diagonal(mala, 0)
    mala[np.diag_indices(NG)] = 1 - mala.sum(1)
    w = np.exp(lp - lp.max())
    return x, w / w.sum(), s, ds, ula, mala


def solve(P, hv):
    n = len(hv)
    A = np.eye(n) - P + 1.0 / n
    pi = np.linalg.solve(A.T, np.full(n, 1.0 / n))
    pi = np.clip(pi, 0, None)
    pi /= pi.sum()
    hb = hv - pi @ hv
    u = np.linalg.solve(A, hb)
    return pi @ hv, 2 * pi @ (hb * u) - pi @ hb**2


eps_grid = np.geomspace(0.01, 0.4, 13)
bias_over_eps, coef = {}, {}
for name, (lohi, fn) in TARGETS.items():
    vals = []
    for e in eps_grid:
        x, w, s, ds, ula, _ = kernels(lohi, fn, e)
        hv = x**2
        m, _ = solve(ula, hv)
        vals.append((m - w @ hv) / e)
    psi = ds + 0.5 * s**2
    coef[name] = 0.5 * (w @ (hv * psi) - (w @ hv) * (w @ psi))
    bias_over_eps[name] = np.array(vals)

# RMSE of an N-step average of x^2 under the Gaussian target, ULA and MALA
eps_fine = np.geomspace(0.01, 0.4, 25)
ula_b, ula_v, mala_v = [], [], []
for e in eps_fine:
    x, w, s, ds, ula, mala = kernels(*TARGETS["Gaussian"], e)
    hv = x**2
    m, av = solve(ula, hv)
    ula_b.append(m - w @ hv)
    ula_v.append(av)
    _, avm = solve(mala, hv)
    mala_v.append(avm)
ula_b, ula_v, mala_v = map(np.array, (ula_b, ula_v, mala_v))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), dpi=100)
ax = axes[0]
for name, col in zip(TARGETS, [PURPLE, TEAL, AMBER]):
    ax.plot(
        eps_grid,
        bias_over_eps[name],
        color=col,
        lw=2.4,
        marker="o",
        ms=5,
        label=f"{name}: exact bias / ε",
    )
    ax.axhline(coef[name], color=col, lw=1.6, ls=(0, (2, 3)))
ax.set_xscale("log")
ax.set_xlabel("step size ε (log scale)", color=INK)
ax.set_ylabel(r"ULA bias in $E[X^2]$, divided by ε", color=INK)
ax.set_title("Bias / ε tends to Stein's coefficient (dotted)", color=INK, fontsize=14)
ax.legend(frameon=False, fontsize=11, loc="upper left")
ax = axes[1]
for N, ls in [(1e4, "-"), (1e5, (0, (6, 3))), (1e6, (0, (2, 2)))]:
    r = np.sqrt(ula_b**2 + ula_v / N)
    rm = np.sqrt(mala_v / N)
    tag = f"N = 1e{int(np.log10(N))}"
    ax.plot(eps_fine, r, color=PURPLE, lw=2.4, ls=ls, label=f"ULA, {tag}")
    ax.plot(eps_fine, rm, color=TEAL, lw=2.4, ls=ls, label=f"MALA, {tag}")
    j = np.argmin(r)
    ax.plot(eps_fine[j], r[j], "o", mfc="none", mec=PURPLE, ms=10, mew=2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("step size ε (log scale)", color=INK)
ax.set_ylabel(r"RMSE of the N-step average of $X^2$, Gaussian target", color=INK)
ax.set_title("ULA's best step shrinks with the budget", color=INK, fontsize=14)
ax.legend(frameon=False, fontsize=10, ncol=2, loc="lower left")
for ax in axes:
    ax.grid(color=RULE, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK)
fig.tight_layout(pad=1.6)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=100, facecolor="white")
print(f"wrote {OUT}")
for name in TARGETS:
    print(
        f"  {name}: c = {coef[name]:.5f}; bias/eps at eps = 0.01: {bias_over_eps[name][0]:.5f}, at 0.4: {bias_over_eps[name][-1]:.5f}"
    )
for N in (1e4, 1e5, 1e6):
    r = np.sqrt(ula_b**2 + ula_v / N)
    print(
        f"  N = {N:.0e}: ULA best eps on this grid {eps_fine[np.argmin(r)]:.3f}, rmse {r.min():.4f}; MALA rmse at 0.4 {np.sqrt(mala_v[-1] / N):.4f}"
    )
