"""Exact Wasserstein-1 distances between a standardised sum and N(0, 1), and the
Ross (2011, Thm 3.2) bound, for four summand laws and n = 1..200.

Writes widget-data/data.js (read by the widget) and figures/wasserstein-rate.png
(the post's static figure and cover). Run from the repo root:

    uv run --no-project --with numpy --with scipy --with matplotlib \
        python posts/steins-method-normal-approximation/src/make_data.py

d_W(W, Z) = integral of |F_W - Phi| over the line. Every law is computed without
Monte Carlo, so the curves carry no seed noise:
  - Bernoulli(p): the binomial CDF is a step function; each step is integrated
    against Phi in closed form with Psi(x) = x Phi(x) + phi(x), an antiderivative of Phi.
  - Exponential(1): the sum is Gamma(n, 1); its CDF is the regularised incomplete gamma.
  - Uniform: Irwin-Hall CDF by the recursion F_{n+1}(x) = G_n(x) - G_n(x - 1), with
    G_n the running integral of F_n on a grid of spacing 1/M (checked against the
    closed-form Irwin-Hall CDF for small n).
  - Skewed mixture: a sum of n draws from a two-component Gaussian mixture is a
    binomial mixture of Gaussians, so its CDF is a finite sum of Phi terms.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, special, stats

HERE = Path(__file__).resolve().parents[1]
NMAX = 200
NS = np.arange(1, NMAX + 1)
P_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
SQ2PI = np.sqrt(2 / np.pi)

# Skewed mixture: 85% from N(0, 0.6^2), 15% from N(3, 1) -- synthetic, standardised below.
MIX_W, MIX_M1, MIX_S1, MIX_M2, MIX_S2 = 0.85, 0.0, 0.6, 3.0, 1.0
MIX_MEAN = MIX_W * MIX_M1 + (1 - MIX_W) * MIX_M2
MIX_VAR = (
    MIX_W * (MIX_S1**2 + MIX_M1**2)
    + (1 - MIX_W) * (MIX_S2**2 + MIX_M2**2)
    - MIX_MEAN**2
)
MIX_SD = np.sqrt(MIX_VAR)

Phi, phi = special.ndtr, stats.norm.pdf


def Psi(x):
    """Antiderivative of Phi: integral_{-inf}^x Phi(t) dt."""
    return x * Phi(x) + phi(x)


def int_abs_const_minus_Phi(c, a, b):
    """integral_a^b |c - Phi(x)| dx, exactly (Phi crosses c at most once)."""
    if b <= a:
        return 0.0
    xs = (
        special.ndtri(min(max(c, 1e-300), 1 - 1e-16))
        if 0 < c < 1
        else (np.inf if c >= 1 else -np.inf)
    )
    m = min(max(xs, a), b)
    left = c * (m - a) - (Psi(m) - Psi(a))  # c > Phi on [a, m]
    right = (Psi(b) - Psi(m)) - c * (b - m)  # Phi > c on [m, b]
    return left + right


def dw_bernoulli(n, p):
    q = 1 - p
    k = np.arange(n + 1)
    w = (k - n * p) / np.sqrt(n * p * q)
    cdf = stats.binom.cdf(k, n, p)
    total = Psi(w[0])  # (-inf, w_0): F = 0
    for j in range(n):
        total += int_abs_const_minus_Phi(cdf[j], w[j], w[j + 1])
    total += Psi(-w[n])  # [w_n, inf): F = 1
    return total


def trapz_abs(F, x):
    return float(np.trapezoid(np.abs(F - Phi(x)), x))


def dw_exponential(n):
    lo = -np.sqrt(n)
    x = np.linspace(lo, 40.0, 400_001)
    F = special.gammainc(n, np.maximum(0.0, n + x * np.sqrt(n)))
    return Psi(lo) + trapz_abs(F, x)


def irwin_hall_cdfs(nmax, M=4000):
    """Yield (n, x, F) for the sum of n U(0,1) on the grid x = k/M, n = 1..nmax."""
    x = np.arange(0, M + 1) / M
    F = x.copy()
    yield 1, x, F
    for n in range(2, nmax + 1):
        xg = np.arange(0, n * M + 1) / M
        Fp = np.concatenate(
            [F, np.ones(len(xg) - len(F))]
        )  # F_{n-1} extended to the new range
        G = integrate.cumulative_trapezoid(Fp, xg, initial=0.0)
        Gs = np.concatenate([np.zeros(M), G[:-M]])  # G(x - 1)
        F = G - Gs
        x = xg
        yield n, x, F


def irwin_hall_exact(n, x):
    k = np.arange(n + 1)
    out = []
    for xv in np.atleast_1d(x):
        kk = k[k <= np.floor(xv)]
        out.append(
            np.sum((-1.0) ** kk * special.comb(n, kk) * (xv - kk) ** n)
            / special.factorial(n)
        )
    return np.array(out)


def dw_uniform_all():
    out = {}
    for n, x, F in irwin_hall_cdfs(NMAX):
        if n in (2, 3, 6, 10):  # check the recursion against the closed form
            probe = np.array([0.3, n / 2 - 0.4, n / 2, n / 2 + 0.77])
            err = np.max(np.abs(np.interp(probe, x, F) - irwin_hall_exact(n, probe)))
            assert err < 1e-6, (n, err)
        sd = np.sqrt(n / 12)
        w = (x - n / 2) / sd
        edge = n / 2 / sd  # support of W is [-edge, edge]
        out[n] = Psi(-edge) + trapz_abs(F, w) + Psi(-edge)
    return out


def dw_mixture(n):
    k = np.arange(n + 1)
    b = stats.binom.pmf(k, n, MIX_W)
    means = k * MIX_M1 + (n - k) * MIX_M2
    sds = np.sqrt(k * MIX_S1**2 + (n - k) * MIX_S2**2)
    x = np.linspace(-12, 40, 200_001)
    s = n * MIX_MEAN + x * MIX_SD * np.sqrt(n)
    F = np.zeros_like(x)
    for bk, mk, sk in zip(b, means, sds):
        if bk > 1e-18:
            F += bk * Phi((s - mk) / sk)
    return trapz_abs(F, x)


def moments():
    """E|X|^3 and E X^4 of each standardised law."""
    out = {}
    out["exponential"] = (
        integrate.quad(lambda t: (1 - t) ** 3 * np.exp(-t), 0, 1)[0]
        + integrate.quad(lambda t: (t - 1) ** 3 * np.exp(-t), 1, np.inf)[0],
        9.0,
    )
    out["uniform"] = (3 * np.sqrt(3) / 4, 9 / 5)

    def mix_pdf(y):
        return MIX_W * stats.norm.pdf(y, MIX_M1, MIX_S1) + (1 - MIX_W) * stats.norm.pdf(
            y, MIX_M2, MIX_S2
        )

    def mix_moment(g):
        return integrate.quad(
            lambda y: g((y - MIX_MEAN) / MIX_SD) * mix_pdf(y), -20, 30, limit=400
        )[0]

    assert (
        abs(mix_moment(lambda x: x)) < 1e-10
        and abs(mix_moment(lambda x: x * x) - 1) < 1e-10
    )
    out["mixture"] = (mix_moment(lambda x: abs(x) ** 3), mix_moment(lambda x: x**4))
    return out


def bern_moments(p):
    q = 1 - p
    return (p * p + q * q) / np.sqrt(p * q), (1 - 3 * p * q) / (p * q)


def bound(m3, m4, n):
    return (m3 + SQ2PI * np.sqrt(m4)) / np.sqrt(n)


def main():
    # cross-check the Bernoulli step formula against brute-force integration
    for n, p in [(1, 0.5), (3, 0.2), (10, 0.7)]:
        x = np.linspace(-15, 15, 600_001)
        F = stats.binom.cdf(np.floor(n * p + x * np.sqrt(n * p * (1 - p))), n, p)
        assert abs(dw_bernoulli(n, p) - trapz_abs(F, x)) < 1e-5

    mom = moments()
    data = {"n": NS.tolist(), "p": P_GRID.tolist(), "laws": {}}
    bern = []
    for p in P_GRID:
        m3, m4 = bern_moments(p)
        dw = [dw_bernoulli(int(n), float(p)) for n in NS]
        bern.append({"p": float(p), "m3": m3, "m4": m4, "dw": dw})
    data["laws"]["bernoulli"] = bern

    expo = [dw_exponential(int(n)) for n in NS]
    unif_d = dw_uniform_all()
    unif = [unif_d[int(n)] for n in NS]
    mix = [dw_mixture(int(n)) for n in NS]
    for key, dw in [("exponential", expo), ("uniform", unif), ("mixture", mix)]:
        m3, m4 = mom[key]
        data["laws"][key] = {"m3": m3, "m4": m4, "dw": dw}
    data["mixture"] = {
        "w": MIX_W,
        "m1": MIX_M1,
        "s1": MIX_S1,
        "m2": MIX_M2,
        "s2": MIX_S2,
        "mean": MIX_MEAN,
        "sd": MIX_SD,
    }

    # The theorem: the bound sits above the exact distance at every n, law and grid p.
    worst = np.inf
    for entry in bern:
        r = bound(entry["m3"], entry["m4"], NS) / np.array(entry["dw"])
        worst = min(worst, r.min())
    for key in ("exponential", "uniform", "mixture"):
        m3, m4 = mom[key]
        r = bound(m3, m4, NS) / np.array(data["laws"][key]["dw"])
        worst = min(worst, r.min())
    assert np.isfinite(worst) and worst > 1, worst
    for key in ("exponential", "uniform", "mixture"):
        assert np.all(np.isfinite(data["laws"][key]["dw"])), key

    def rnd(v):
        return float(f"{v:.6g}")

    for entry in bern:
        entry["dw"] = [rnd(v) for v in entry["dw"]]
        entry["m3"], entry["m4"] = rnd(entry["m3"]), rnd(entry["m4"])
    for key in ("exponential", "uniform", "mixture"):
        d = data["laws"][key]
        d["dw"] = [rnd(v) for v in d["dw"]]
        d["m3"], d["m4"] = rnd(d["m3"]), rnd(d["m4"])
    out = HERE / "widget-data" / "data.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "/* Written by src/make_data.py: exact d_W(W_n, Z) for n = 1..200. Do not edit. */\n"
        "window.WK_DATA = window.WK_DATA || {};\nwindow.WK_DATA.steinClt = "
        + json.dumps(data, separators=(",", ":"))
        + ";\nif (typeof module !== 'undefined') module.exports = window.WK_DATA.steinClt;\n"
    )
    print(
        f"wrote {out} ({out.stat().st_size / 1024:.0f} kB); smallest bound / d_W = {worst:.2f}"
    )

    # Summary numbers for the prose.
    def row(name, dw, m3, m4):
        dw = np.array(dw)
        slope = np.polyfit(np.log(NS[49:]), np.log(dw[49:]), 1)[0]
        print(
            f"{name:16s} m3={m3:.4f} m4={m4:.4f} C={m3 + SQ2PI * np.sqrt(m4):.3f} "
            f"dW(1)={dw[0]:.4f} dW(10)={dw[9]:.4f} dW(100)={dw[99]:.5f} dW(200)={dw[199]:.5f} "
            f"bound(200)={bound(m3, m4, 200):.4f} ratio(200)={bound(m3, m4, 200) / dw[199]:.1f} "
            f"slope(50..200)={slope:.3f}"
        )

    for entry in bern:
        if entry["p"] in (0.05, 0.1, 0.3, 0.5):
            row(f"bernoulli p={entry['p']}", entry["dw"], entry["m3"], entry["m4"])
    for key in ("exponential", "uniform", "mixture"):
        d = data["laws"][key]
        row(key, d["dw"], d["m3"], d["m4"])

    figure(data)


def figure(data):
    INK, MUTED, RULE = "#1F2430", "#5F6672", "#D8DBE2"
    cols = {
        "bernoulli": "#4A3AA7",
        "exponential": "#1D6E6E",
        "uniform": "#9A5B00",
        "mixture": "#A8437A",
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(12, 6.3), dpi=100)
    series = [
        (
            "bernoulli",
            "Bernoulli(0.3)",
            next(e for e in data["laws"]["bernoulli"] if e["p"] == 0.3),
        ),
        ("exponential", "Exponential", data["laws"]["exponential"]),
        ("uniform", "Uniform", data["laws"]["uniform"]),
        ("mixture", "Skewed mixture", data["laws"]["mixture"]),
    ]
    for key, label, d in series:
        c = cols[key]
        ax.plot(NS, d["dw"], color=c, lw=2.4, label=f"{label}: exact $d_W$")
        ax.plot(NS, bound(d["m3"], d["m4"], NS), color=c, lw=1.4, ls=(0, (6, 4)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, NMAX)
    ax.set_xlabel("number of summands n", color=INK)
    ax.set_ylabel(r"Wasserstein distance to $N(0,1)$", color=INK)
    ax.set_title(
        "Exact distance (solid) and the leave-one-out Stein bound (dashed)",
        color=INK,
        fontsize=14,
    )
    ax.grid(which="major", color=RULE, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=12)
    fig.tight_layout(pad=1.6)
    out = HERE / "figures" / "wasserstein-rate.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
