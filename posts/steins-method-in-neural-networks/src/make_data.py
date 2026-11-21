"""Exact law of a random one-hidden-layer network's output at one input, for the
post's first chart. Writes widget-data/data.js.

With Gaussian output weights, the output is W = sqrt(V) Z, Z ~ N(0, 1) independent of
V = (1/n) sum_i sigma(g_i)^2 / E sigma(g)^2, g_i i.i.d. N(0, 1). So the law of W is
fixed by the law of V:
  - linear: n V ~ chi-square(n);
  - ReLU:   n V / 2 ~ chi-square(K), K ~ Binomial(n, 1/2) (K = 0 is an atom at 0);
  - sign:   V = 1, so W is exactly N(0, 1);
  - tanh, GELU: the law of n V by FFT convolution of a fine lattice discretisation
    of sigma(g)^2 / E sigma^2 (spacing <= 0.002).
d_W(W, Z) = int |F_W - Phi| is then a quadrature over w and over the law of V.

Run from the repo root:

    uv run --no-project --with numpy --with scipy \
        python posts/steins-method-in-neural-networks/src/make_data.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.integrate import quad

OUT = Path(__file__).resolve().parents[1] / "widget-data" / "data.js"
ACTS = {
    "linear": lambda g: g,
    "relu": lambda g: np.maximum(g, 0),
    "gelu": lambda g: g * stats.norm.cdf(g),
    "tanh": np.tanh,
    "sign": np.sign,
}
NS = [
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    13,
    16,
    20,
    25,
    32,
    40,
    50,
    64,
    80,
    100,
    128,
    160,
    200,
    256,
    320,
    400,
    500,
    640,
    800,
    1000,
]
WD = np.linspace(-3.975, 3.975, 160)  # display grid (midpoints, so w = 0 is avoided)
WQ = np.linspace(-30, 30, 30001)  # quadrature grid for d_W


def gmom(f):
    pdf = stats.norm.pdf
    return (
        quad(lambda g: f(g) * pdf(g), -12, 0, limit=400)[0]
        + quad(lambda g: f(g) * pdf(g), 0, 12, limit=400)[0]
    )


def lattice_law(act, n):
    a = ACTS[act]
    c2 = gmom(lambda g: a(g) ** 2)
    M = 400000
    g = stats.norm.ppf((np.arange(M) + 0.5) / M)
    Y = a(g) ** 2 / c2
    delta = min(0.002, Y.max() / 4000)
    K = int(np.ceil(Y.max() / delta)) + 2
    py = np.bincount(np.round(Y / delta).astype(int), minlength=K) / M
    L = n * (K - 1) + 1
    nfft = 1 << int(np.ceil(np.log2(L)))
    ps = np.fft.irfft(np.fft.rfft(py, nfft) ** n, nfft)[:L]
    ps = np.clip(ps, 0, None)
    ps /= ps.sum()
    v = np.arange(L) * delta / n
    # lattice point 0 holds sums that rounded to 0, not an atom: place it inside its cell
    v[0] = 0.25 * delta / n
    keep = ps > 1e-14
    return v[keep], ps[keep]


def law_V(act, n):
    """Support points and weights of V (a discretisation fine enough for 4-digit d_W)."""
    if act == "sign":
        return np.array([1.0]), np.array([1.0])
    if act == "linear":
        q = (np.arange(4000) + 0.5) / 4000
        return stats.chi2.ppf(q, n) / n, np.full(4000, 1 / 4000)
    if act == "relu":
        vs, ws = [], []
        for k in range(n + 1):
            pk = stats.binom.pmf(k, n, 0.5)
            if pk < 1e-13:
                continue
            if k == 0:
                vs.append([0.0])
                ws.append([pk])
                continue
            # 4000 midpoint quantiles per k: with 400 the tails were short and Var V came out 0.3% low
            q = (np.arange(4000) + 0.5) / 4000
            vs.append(2 * stats.chi2.ppf(q, k) / n)
            ws.append(np.full(4000, pk / 4000))
        return np.concatenate(vs), np.concatenate(ws)
    return lattice_law(act, n)


def compress(v, w, bins=3000):
    if len(v) <= bins:
        return v, w
    edges = np.quantile(v, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, bins - 1)
    wb = np.bincount(idx, weights=w, minlength=bins)
    vb = np.bincount(idx, weights=w * v, minlength=bins) / np.maximum(wb, 1e-300)
    keep = wb > 0
    return vb[keep], wb[keep]


def summaries(act, n):
    # keep the atom at V = 0 (ReLU) out of the binning, which would merge it with small V
    v0, w0 = law_V(act, n)
    zero = v0 == 0
    v, w = compress(v0[~zero], w0[~zero])
    v, w = (
        np.append(v, 0.0) if zero.any() else v,
        np.append(w, w0[zero].sum()) if zero.any() else w,
    )
    F = np.zeros_like(WQ)
    dens = np.zeros_like(WD)
    atom = 0.0
    for vi, wi in zip(v, w):
        if vi == 0:
            F += wi * (WQ >= 0)
            atom += wi
        else:
            r = np.sqrt(vi)
            F += wi * stats.norm.cdf(WQ / r)
            dens += wi * stats.norm.pdf(WD / r) / r
    dW = np.trapezoid(np.abs(F - stats.norm.cdf(WQ)), WQ)
    return dW, dens, atom


def r5(x):
    return float(f"{x:.5g}")


data = {"n": NS, "w": [r5(x) for x in WD], "acts": {}}
for act in ACTS:
    rows = {"dW": [], "dens": [], "atom": []}
    for n in NS:
        dW, dens, atom = summaries(act, n)
        rows["dW"].append(r5(dW))
        rows["dens"].append([r5(x) for x in dens])
        rows["atom"].append(r5(atom))
    data["acts"][act] = rows
    print(
        act,
        "n*dW at n = 1, 10, 100, 1000:",
        [round(n * d, 4) for n, d in zip(NS, rows["dW"]) if n in (1, 10, 100, 1000)],
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    "/* Generated by src/make_data.py: exact Wasserstein distance from N(0, 1) and density of a random\n"
    " * one-hidden-layer network's output at one input, by activation and width. Do not edit. */\n"
    "window.WK_DATA = window.WK_DATA || {};\n"
    "window.WK_DATA.netGaussianity = " + json.dumps(data, separators=(",", ":")) + ";\n"
    'if (typeof module !== "undefined" && module.exports) module.exports = window.WK_DATA.netGaussianity;\n'
)
print("wrote", OUT, OUT.stat().st_size, "bytes")
