"""Weight generators and weighted functionals for the two bootstraps.

Both bootstraps put a random probability vector ``w`` on the observed sample and
push it through a statistic.  They differ only in the law placed on ``w``:

* Efron        ``w = N / n``,  ``N ~ Multinomial(n; 1/n, ..., 1/n)``  — a lattice
* Bayesian     ``w ~ Dirichlet(1, ..., 1)``                          — the whole simplex

Everything in this module is vectorised over a leading replicate axis of size
``B``, and every generator returns an array of shape ``(B, n)`` whose rows sum
to one.

Closed forms are stated in the docstrings and asserted against simulation in the
post itself; see ``index.qmd``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Final

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]

# --------------------------------------------------------------------------
# Seed discipline
# --------------------------------------------------------------------------
# One root seed for the whole post.  Every experiment draws its own independent
# generator from it by name, so adding an experiment never shifts the stream of
# an existing one, and every figure in the post is reproducible from this line.
SEED: Final[int] = 20260726

_ROOT: Final[np.random.SeedSequence] = np.random.SeedSequence(SEED)


def rng_for(name: str) -> np.random.Generator:
    """Return an independent generator keyed by a human-readable label.

    The label is hashed into the spawn key, so ``rng_for("median")`` is stable
    across runs and independent of every other named stream.

    Args:
        name: Identifier for the experiment, e.g. ``"toy-mean"``.

    Returns:
        A fresh ``numpy.random.Generator``.
    """
    key = int.from_bytes(name.encode("utf-8"), "little") % (2**63)
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(key,)))


# --------------------------------------------------------------------------
# Palette (§7) — figures and page are one object
# --------------------------------------------------------------------------
PAPER: Final[str] = "#F1F2EE"
INK: Final[str] = "#171C1B"
MUTED: Final[str] = "#67706E"
RULE: Final[str] = "#D2D6D1"
EFRON: Final[str] = "#1D5C6E"  # petrol — Efron everywhere
BAYES: Final[str] = "#C98A12"  # ochre — Bayes in figures only
BAYES_TEXT: Final[str] = "#8A5E06"  # darker ochre — Bayes in running text
DATA: Final[str] = "#7A3B6B"  # plum — observed data, plug-in estimates


def use_house_style() -> None:
    """Apply the post's matplotlib rcParams.

    Left-aligned bold titles, no top/right spines, hairline grids, the paper
    background, 150 dpi.  Call once at the top of the post.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.edgecolor": PAPER,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 9.0,
            "axes.labelsize": 10.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": RULE,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.5,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9.0,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "figure.constrained_layout.use": True,
        }
    )


# --------------------------------------------------------------------------
# Weight generators
# --------------------------------------------------------------------------
def efron_weights(n: int, B: int, rng: np.random.Generator) -> Floats:
    """Efron's nonparametric bootstrap weights, ``w = N / n``.

    ``N ~ Multinomial(n; 1/n, ..., 1/n)`` is the vector of multiplicities in a
    with-replacement resample of size ``n``.  Dividing by ``n`` turns it into
    the resample's empirical distribution, so resampling *is* a weighted
    bootstrap; this is bookkeeping, not an assumption.

    Closed forms:
        ``E[w_i] = 1/n``
        ``Var(w_i) = (n - 1) / n**3``
        ``Cov(w_i, w_j) = -1 / n**3``
        ``P(w_i = 0) = (1 - 1/n)**n -> exp(-1)``
        number of distinct weight vectors = ``comb(2n - 1, n - 1)``

    Args:
        n: Sample size.
        B: Number of replicates.
        rng: Source of randomness.

    Returns:
        ``(B, n)`` array of weights; each row sums to one and lies on the
        lattice ``{0, 1/n, 2/n, ...}``.
    """
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B) / n


def dirichlet_weights(n: int, B: int, rng: np.random.Generator) -> Floats:
    """Bayesian bootstrap weights, ``w ~ Dirichlet(1, ..., 1)``.

    Uniform on the simplex.  Generated through the exponential (Gamma(1, 1))
    representation rather than a library call, because that construction is
    part of the argument of the post: normalising is free precisely because the
    direction ``w`` is independent of the total ``S``.

    Closed forms:
        ``E[w_i] = 1/n``
        ``Var(w_i) = (n - 1) / (n**2 * (n + 1))``
        ``Cov(w_i, w_j) = -1 / (n**2 * (n + 1))``
        marginal ``w_i ~ Beta(1, n - 1)``
        variance ratio to Efron ``= n / (n + 1)``

    Args:
        n: Sample size.
        B: Number of replicates.
        rng: Source of randomness.

    Returns:
        ``(B, n)`` array of weights; each row sums to one and is strictly
        positive almost surely.
    """
    e = rng.exponential(1.0, size=(B, n))
    return e / e.sum(axis=1, keepdims=True)


def gamma_weights(n: int, B: int, alpha: float, rng: np.random.Generator) -> Floats:
    """Symmetric ``Dirichlet(alpha, ..., alpha)`` weights via normalised gammas.

    ``G_i ~ Gamma(alpha, 1)`` independent, ``w = G / sum(G)``.  The change of
    variables ``g_i = s * w_i`` has Jacobian ``s**(n-1)`` and factors the joint
    density into a Gamma in ``s`` times a Dirichlet in ``w``, which is why this
    works and why ``w`` is independent of ``s``.

    Closed forms:
        ``Var(w_i) = (n - 1) / (n**2 * (n * alpha + 1))``
        ``Cov(w_i, w_j) = -1 / (n**2 * (n * alpha + 1))``

    Setting ``alpha = 1 - 1/n`` reproduces Efron's mean vector *and* full
    covariance matrix exactly; setting ``alpha = 1`` is the Bayesian bootstrap.

    Sampling is done in log space.  A direct ``rng.gamma(alpha)`` underflows to
    exactly zero for small ``alpha`` — every coordinate rounds to 0, the row sum
    is 0, and the normalisation returns ``nan``.  That matters here because the
    edge-seeking regime ``alpha < 1`` is exactly the interesting one (the
    Efron-matching value ``1 - 1/n``, and the ``alpha -> 0`` prior).  Using the
    boost identity ``Gamma(alpha) == Gamma(alpha + 1) * U**(1/alpha)`` gives

        ``log G_i = log g_i + log(U_i) / alpha``,   ``g_i ~ Gamma(alpha + 1, 1)``

    and a softmax over ``log G`` is stable for any positive ``alpha``.

    Args:
        n: Sample size.
        B: Number of replicates.
        alpha: Common concentration parameter, must be positive.
        rng: Source of randomness.

    Returns:
        ``(B, n)`` array of weights summing to one along the last axis.

    Raises:
        ValueError: If ``alpha`` is not positive.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    boosted = rng.gamma(shape=alpha + 1.0, scale=1.0, size=(B, n))
    log_g = np.log(boosted) + np.log(rng.random(size=(B, n))) / alpha
    log_g -= log_g.max(axis=1, keepdims=True)
    w = np.exp(log_g)
    return w / w.sum(axis=1, keepdims=True)


def poisson_weights(n: int, B: int, rng: np.random.Generator) -> Floats:
    """Poisson bootstrap weights, ``W_i ~ Poisson(1)`` i.i.d., **unnormalised**.

    The streaming bootstrap: because the weights are independent, they need no
    knowledge of ``n`` and can be drawn in a single pass with ``O(1)`` memory.

    The connection that matters for the post: conditioning these weights on
    ``sum(W) = n`` gives *exactly* ``Multinomial(n; 1/n, ..., 1/n)``, i.e. Efron.
    Exp(1) and Poisson(1) are the interarrival times and the counts of the same
    unit-rate Poisson process, so Efron counts arrivals and the Bayesian
    bootstrap measures the gaps between them.

    Args:
        n: Sample size.
        B: Number of replicates.
        rng: Source of randomness.

    Returns:
        ``(B, n)`` array of non-negative integer counts as floats.  Rows do
        **not** sum to ``n``; divide by the row sum to obtain a probability
        vector, or condition on the sum to recover Efron.
    """
    return rng.poisson(1.0, size=(B, n)).astype(np.float64)


# --------------------------------------------------------------------------
# Functionals — every one takes (x_sorted, W) and is vectorised over replicates
# --------------------------------------------------------------------------
def weighted_mean(x_sorted: Floats, W: Floats) -> Floats:
    """Mean of ``F_w``, i.e. ``sum_i w_i x_i``.

    Linear in ``w``, so its first two moments follow directly from the moments
    of the weights.  With ``s2 = mean((x - xbar)**2)`` the variance is

        ``s2 / (n + 1)``  under the Bayesian bootstrap,
        ``s2 / n``        under Efron,

    against the textbook ``s**2 / n = s2 / (n - 1)``.

    Args:
        x_sorted: ``(n,)`` data, ascending (sorting is not required here but is
            assumed everywhere else, so it is required for consistency).
        W: ``(B, n)`` weights.

    Returns:
        ``(B,)`` array of weighted means.
    """
    return W @ x_sorted


def _cumulative(W: Floats) -> Floats:
    """Row-wise cumulative sums of the weights."""
    return np.cumsum(W, axis=1)


def weighted_quantile(x_sorted: Floats, W: Floats, q: float) -> Floats:
    """Lower quantile of ``F_w``: ``inf{x : F_w(x) >= q}``.

    Uses the sort-once trick.  The data are sorted a single time by the caller,
    ``O(n log n)``; each replicate is then one cumulative sum and one
    comparison, ``O(n)``.  Thinking of Efron as producing a new *dataset*
    suggests a re-sort per replicate, ``O(B n log n)``; thinking of it as
    weights removes that cost.  The saving applies to both methods equally.

    Args:
        x_sorted: ``(n,)`` data in ascending order.
        W: ``(B, n)`` weights, rows summing to one.
        q: Probability level in ``(0, 1]``.

    Returns:
        ``(B,)`` array of quantiles, each an element of ``x_sorted``.

    Raises:
        ValueError: If ``q`` is outside ``(0, 1]``.
    """
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must lie in (0, 1], got {q}")
    c = _cumulative(W)
    # Index of the first coordinate whose cumulative weight reaches q.
    idx = (c < q).sum(axis=1)
    np.clip(idx, 0, x_sorted.size - 1, out=idx)
    return x_sorted[idx]


def lower_median(x_sorted: Floats, W: Floats) -> Floats:
    """Median of ``F_w``, defined as ``inf{x : F_w(x) >= 1/2}`` for both methods.

    Fixing one definition for both is what makes the comparison meaningful; the
    two methods are otherwise being scored on different statistics.

    Closed form under the Bayesian bootstrap, from uniform spacings:

        ``P(median = x_(j)) = comb(n - 1, j - 1) * 2**(-(n - 1))``

    a ``Binomial(n - 1, 1/2)`` law over the *ranks*, free of the data values.
    At ``n = 59`` its central 95% interval is order statistics 23 and 37, which
    is exactly the classical distribution-free sign-test interval.

    Args:
        x_sorted: ``(n,)`` data in ascending order.
        W: ``(B, n)`` weights.

    Returns:
        ``(B,)`` array of medians.
    """
    return weighted_quantile(x_sorted, W, 0.5)


def weighted_cdf(x_sorted: Floats, W: Floats, t: float) -> Floats:
    """``F_w(t)``, the mass at or below ``t``.

    With ``k = #{x_i <= t}`` this is a sum of ``k`` of the weights, so by the
    aggregation property of the Dirichlet it has an exact law:

        Bayes  ``F_w(t) ~ Beta(k, n - k)``
        Efron  ``F*(t) ~ Binomial(n, k/n) / n``

    Same mean ``k/n``; variances again in ratio ``n / (n + 1)``.  Degenerate at
    0 for ``t < x_(1)`` and at 1 for ``t >= x_(n)``.

    Args:
        x_sorted: ``(n,)`` data in ascending order.
        W: ``(B, n)`` weights.
        t: Threshold.

    Returns:
        ``(B,)`` array of probabilities.
    """
    k = int(np.searchsorted(x_sorted, t, side="right"))
    if k == 0:
        return np.zeros(W.shape[0], dtype=np.float64)
    return W[:, :k].sum(axis=1)


def support_max(x_sorted: Floats, W: Floats) -> Floats:
    """Largest point carrying positive mass under ``F_w``.

    Under ``Dirichlet(1, ..., 1)`` every weight is strictly positive almost
    surely, so this is ``x_(n)`` with probability one: the posterior is a point
    mass.  Under Efron it is non-degenerate, with

        ``P(max* <= x_(j)) = (j / n)**n``

    and a top atom of ``1 - (1 - 1/n)**n -> 1 - exp(-1)``.  Neither is
    trustworthy: both methods assume ``F`` gives zero probability to unseen
    values, and an extremum is entirely a statement about that region.

    Args:
        x_sorted: ``(n,)`` data in ascending order.
        W: ``(B, n)`` weights.

    Returns:
        ``(B,)`` array of support maxima.
    """
    idx = W.shape[1] - 1 - np.argmax(W[:, ::-1] > 0.0, axis=1)
    return x_sorted[idx]


def support_min(x_sorted: Floats, W: Floats) -> Floats:
    """Smallest point carrying positive mass under ``F_w``.

    Mirror image of :func:`support_max`.  Bayes places a point mass at
    ``x_(1)``; Efron has ``P(min* = x_(1)) = 1 - (1 - 1/n)**n``.

    Args:
        x_sorted: ``(n,)`` data in ascending order.
        W: ``(B, n)`` weights.

    Returns:
        ``(B,)`` array of support minima.
    """
    return x_sorted[np.argmax(W > 0.0, axis=1)]


def weighted_kde_mode(
    x_sorted: Floats,
    W: Floats,
    bandwidth: float,
    grid: Floats | None = None,
    n_grid: int = 512,
) -> Floats:
    """Mode of a weighted Gaussian kernel density estimate.

    The mode of an empirical distribution over *distinct* observed values is
    ``argmax_i w_i``, which is meaningless: under the Bayesian bootstrap it is
    uniform over the data points by exchangeability, and under Efron it is
    decided by ties and hence by the tie-breaking rule.  The fix is to repair
    the *estimator*, not the bootstrap — smooth first, then take the argmax of

        ``f_w(t) = sum_i w_i K_h(t - x_i)``

    into which Dirichlet weights slot with no modification at all.

    Args:
        x_sorted: ``(n,)`` data in ascending order.
        W: ``(B, n)`` weights.
        bandwidth: Gaussian kernel bandwidth, must be positive.
        grid: Optional evaluation grid; defaults to ``n_grid`` points spanning
            the data padded by three bandwidths.
        n_grid: Size of the default grid.

    Returns:
        ``(B,)`` array of KDE modes.

    Raises:
        ValueError: If ``bandwidth`` is not positive.
    """
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    if grid is None:
        pad = 3.0 * bandwidth
        grid = np.linspace(x_sorted[0] - pad, x_sorted[-1] + pad, n_grid)
    # (n_grid, n) kernel matrix, shared across replicates.
    z = (grid[:, None] - x_sorted[None, :]) / bandwidth
    kernel = np.exp(-0.5 * z * z)
    dens = W @ kernel.T  # (B, n_grid)
    return grid[np.argmax(dens, axis=1)]


# --------------------------------------------------------------------------
# Chunked driver — peak memory stays bounded
# --------------------------------------------------------------------------
def chunked(B: int, chunk_size: int) -> Iterator[int]:
    """Yield replicate-batch sizes summing to ``B``.

    Args:
        B: Total replicates.
        chunk_size: Maximum replicates held in memory at once.

    Yields:
        Batch sizes.

    Raises:
        ValueError: If ``chunk_size`` is not positive.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    done = 0
    while done < B:
        step = min(chunk_size, B - done)
        yield step
        done += step


def bootstrap_apply(
    x_sorted: Floats,
    weight_fn: Callable[[int, int, np.random.Generator], Floats],
    functional: Callable[[Floats, Floats], Floats],
    B: int,
    rng: np.random.Generator,
    chunk_size: int = 20_000,
) -> Floats:
    """Draw ``B`` replicates in bounded memory and return the functional's values.

    A dense ``(B, n)`` weight array is ``8 * B * n`` bytes: 94 MB at
    ``B = 200_000, n = 59`` and 16 GB at ``n = 10_000``.  Chunking the replicate
    axis keeps peak working memory at ``O(chunk_size * n)`` while leaving the
    result ``O(B)``.

    Args:
        x_sorted: ``(n,)`` data in ascending order.
        weight_fn: One of the generators above, called as ``(n, b, rng)``.
        functional: Called as ``(x_sorted, W)``, returns one value per row.
        B: Total replicates.
        rng: Source of randomness.
        chunk_size: Replicates per batch.

    Returns:
        ``(B,)`` array of the functional evaluated on each replicate.
    """
    n = x_sorted.size
    out = np.empty(B, dtype=np.float64)
    pos = 0
    for step in chunked(B, chunk_size):
        W = weight_fn(n, step, rng)
        out[pos : pos + step] = functional(x_sorted, W)
        pos += step
    return out


# --------------------------------------------------------------------------
# Closed forms, for asserting against simulation
# --------------------------------------------------------------------------
def biased_var(x: Floats) -> float:
    """``s2 = mean((x - xbar)**2)``, the ``1/n`` variance the ladder is built on."""
    return float(np.mean((x - x.mean()) ** 2))


def ladder(x: Floats) -> tuple[float, float, float]:
    """The three denominators for the variance of the mean.

    Returns ``(bayes, efron, classical)`` = ``s2/(n+1)``, ``s2/n``, ``s2/(n-1)``
    — the same quantity with three different fudges, and always in that order.

    Args:
        x: ``(n,)`` data.

    Returns:
        Bayes, Efron and classical variances of the mean.
    """
    n = x.size
    s2 = biased_var(x)
    return s2 / (n + 1), s2 / n, s2 / (n - 1)


def mc_se_of_sd(sd: float, B: int) -> float:
    """Monte-Carlo standard error of a simulated standard deviation.

    The relative standard error of an sd estimated from ``B`` replicates is
    about ``1 / sqrt(2 B)``.  At ``B = 200_000`` that is ~0.16%, so a *ratio* of
    two simulated sds carries roughly +/-0.002 — wider than several of the
    Efron-versus-Bayes gaps quoted in this post, which is why those comparisons
    are reported as ties rather than read as differences.

    Args:
        sd: The estimated standard deviation.
        B: Number of replicates it was estimated from.

    Returns:
        Approximate standard error of ``sd``.
    """
    return sd / np.sqrt(2.0 * B)
