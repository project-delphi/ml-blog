"""Penalised-spline intensity model, its linear baseline, and the scoring.

The two models share a likelihood, an offset and a random-effect structure, so
their expected log predictive densities are comparable. Hold-out is carried by a
weight vector of fixed length rather than by subsetting the data, which lets
nutpie compile each model graph once and reuse it for every fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.interpolate import BSpline
from scipy.stats import nbinom

SMOOTHS = ("load", "gap", "nbr")


# ------------------------------------------------------------------- basis


@dataclass(frozen=True)
class Basis:
    """A cubic B-spline basis, centred so it is identified against the
    intercept. The same knots and the same centring go on to the plotting grid,
    or the estimated curve would not be on the true curve's scale.
    """

    knots: np.ndarray
    degree: int
    centre: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        lo, hi = self.knots[self.degree], self.knots[-self.degree - 1]
        xc = np.clip(np.asarray(x, dtype=float), lo, hi)
        design = BSpline.design_matrix(xc, self.knots, self.degree).toarray()
        return design - self.centre


def make_basis(x: np.ndarray, n_basis: int = 8, degree: int = 3) -> Basis:
    x = np.asarray(x, dtype=float)
    n_interior = n_basis - degree - 1
    interior = np.quantile(x, np.linspace(0, 1, n_interior + 2)[1:-1])
    lo, hi = x.min(), x.max()
    knots = np.concatenate([np.repeat(lo, degree + 1), interior, np.repeat(hi, degree + 1)])
    raw = BSpline.design_matrix(np.clip(x, lo, hi), knots, degree).toarray()
    return Basis(knots=knots, degree=degree, centre=raw.mean(axis=0))


# ------------------------------------------------------------------- models


@dataclass
class Design:
    y: np.ndarray
    log_offset: np.ndarray
    donor: np.ndarray
    well: np.ndarray
    bases: dict[str, Basis]
    X: dict[str, np.ndarray]
    linear: np.ndarray
    n_donors: int
    n_wells: int


def make_design(panel: pd.DataFrame, n_basis: int = 8) -> Design:
    bases = {s: make_basis(panel[s].to_numpy(), n_basis=n_basis) for s in SMOOTHS}
    X = {s: bases[s](panel[s].to_numpy()) for s in SMOOTHS}
    # The baseline gets the same three covariates on a sensible scale, so the
    # comparison is about the shape of the response, not the variables in it.
    linear = np.column_stack(
        [
            panel["load"].to_numpy(),
            panel["gap"].to_numpy(),
            np.log1p(panel["nbr"].to_numpy()),
        ],
    )
    linear = (linear - linear.mean(axis=0)) / linear.std(axis=0)
    wells = pd.factorize(panel["well"])[0]
    return Design(
        y=panel["y"].to_numpy().astype(int),
        log_offset=panel["log_offset"].to_numpy(),
        donor=panel["donor"].to_numpy().astype(int),
        well=wells,
        bases=bases,
        X=X,
        linear=linear,
        n_donors=int(panel["donor"].nunique()),
        n_wells=int(wells.max() + 1),
    )


def _shared(model: pm.Model, d: Design) -> tuple:
    """Intercept, random effects and dispersion, shared by both models."""
    intercept = pm.Normal("intercept", -2.0, 2.0)
    sd_donor = pm.HalfNormal("sd_donor", 0.5)
    sd_well = pm.HalfNormal("sd_well", 0.3)
    u_donor = pm.Deterministic("u_donor", sd_donor * pm.Normal("z_donor", 0, 1, shape=d.n_donors))
    u_well = pm.Deterministic("u_well", sd_well * pm.Normal("z_well", 0, 1, shape=d.n_wells))
    # Parameterised so that inv_alpha -> 0 is the Poisson limit, which is where
    # this data actually sits. A prior directly on alpha fights that boundary.
    inv_alpha = pm.HalfNormal("inv_alpha", 1.0)
    alpha = pm.Deterministic("alpha", 1.0 / (inv_alpha + 1e-6))
    return intercept, u_donor, u_well, alpha


def _finish(model: pm.Model, d: Design, eta: pt.TensorVariable, alpha, weights: np.ndarray) -> None:
    w = pm.Data("w", weights.astype(float))
    mu = pt.exp(eta + d.log_offset)
    dist = pm.NegativeBinomial.dist(mu=mu, alpha=alpha)
    pm.Potential("loglik", (w * pm.logp(dist, d.y)).sum())


def gam_model(d: Design, weights: np.ndarray, n_basis: int = 8) -> pm.Model:
    """Penalised splines: a second-order random walk on the basis coefficients,
    written non-centred so the sampler sees a unit-scale parameter.
    """
    with pm.Model() as model:
        intercept, u_donor, u_well, alpha = _shared(model, d)
        eta = intercept + u_donor[d.donor] + u_well[d.well]
        for s in SMOOTHS:
            tau = pm.HalfNormal(f"tau_{s}", 0.5)
            z = pm.Normal(f"z_{s}", 0, 1, shape=n_basis)
            raw = tau * pt.cumsum(pt.cumsum(z))
            beta = pm.Deterministic(f"beta_{s}", raw - raw.mean())
            eta = eta + pt.dot(d.X[s], beta)
        _finish(model, d, eta, alpha, weights)
    return model


def linear_model(d: Design, weights: np.ndarray) -> pm.Model:
    """The pre-registered baseline: the same likelihood and offset, linear in
    the same three covariates.
    """
    with pm.Model() as model:
        intercept, u_donor, u_well, alpha = _shared(model, d)
        b = pm.Normal("b", 0, 1, shape=d.linear.shape[1])
        eta = intercept + u_donor[d.donor] + u_well[d.well] + pt.dot(d.linear, b)
        _finish(model, d, eta, alpha, weights)
    return model


# ------------------------------------------------------------------ scoring


def _draws(trace, name: str) -> np.ndarray:
    """Posterior draws as (draw, ...), flattening chains."""
    arr = trace.posterior[name].to_numpy()
    return arr.reshape((-1,) + arr.shape[2:])


def heldout_logscore(
    trace,
    d: Design,
    rows: np.ndarray,
    kind: str,
    rng: np.random.Generator,
    max_draws: int = 400,
) -> np.ndarray:
    """Pointwise log predictive density on held-out rows, for a *new* donor.

    The held-out donor was never in the fit, so its offset is drawn from the
    fitted population of donors rather than read off a fitted value. Scoring it
    with a fitted donor effect would flatter every model that has one.
    """
    intercept = _draws(trace, "intercept")
    sd_donor = _draws(trace, "sd_donor")
    sd_well = _draws(trace, "sd_well")
    alpha = _draws(trace, "alpha")
    n = intercept.shape[0]
    idx = rng.choice(n, size=min(max_draws, n), replace=False)

    eta = intercept[idx][:, None] + np.zeros((1, rows.size))
    if kind == "gam":
        for s in SMOOTHS:
            eta = eta + _draws(trace, f"beta_{s}")[idx] @ d.X[s][rows].T
    else:
        eta = eta + _draws(trace, "b")[idx] @ d.linear[rows].T

    # One new donor, and one new well per held-out well.
    eta = eta + rng.normal(0, 1, size=(idx.size, 1)) * sd_donor[idx][:, None]
    wells = pd.factorize(d.well[rows])[0]
    eta = eta + (rng.normal(0, 1, size=(idx.size, wells.max() + 1)) * sd_well[idx][:, None])[:, wells]

    mu = np.exp(eta + d.log_offset[rows])
    a = alpha[idx][:, None]
    lp = nbinom.logpmf(d.y[rows], a, a / (a + mu))
    # log mean exp over draws, in a numerically safe form
    m = lp.max(axis=0)
    return m + np.log(np.exp(lp - m).mean(axis=0))


@dataclass
class Fold:
    donor: int
    gam: np.ndarray
    linear: np.ndarray


def elpd_summary(folds: list[Fold]) -> dict[str, float]:
    gam = np.concatenate([f.gam for f in folds])
    lin = np.concatenate([f.linear for f in folds])
    diff = gam - lin
    return {
        "elpd_gam": float(gam.sum()),
        "elpd_linear": float(lin.sum()),
        "diff": float(diff.sum()),
        "se": float(np.sqrt(diff.size) * diff.std(ddof=1)),
        "n": int(diff.size),
    }


# -------------------------------------------------------------------- power


def naive_share(cells: pd.DataFrame) -> pd.DataFrame:
    """Share of second-wave uptake taken by cells that ate nothing in wave 1.

    This is the quantity the arms actually move. Mean uptake per cell is nearly
    identical across arms by construction -- that is the point of the post -- so
    powering the design on the mean would be powering it on the one number the
    experiment was built not to change.
    """
    c = cells.copy()
    c["naive"] = c["wave1_uptake"] == 0
    g = c.groupby(["donor", "arm"], as_index=False).apply(
        lambda s: pd.Series({
            "share": s.loc[s["naive"], "wave2_uptake"].sum() / max(s["wave2_uptake"].sum(), 1),
            "naive_frac": float(s["naive"].mean()),
            "total": float(s["wave2_uptake"].sum()),
        }),
        include_groups=False,
    )
    return g


def _logit(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def arm_contrast(cells: pd.DataFrame) -> float:
    """Donor-paired contrast in the naive share, focal minus broad, on the
    logit scale.
    """
    g = naive_share(cells).pivot(index="donor", columns="arm", values="share")
    return float(np.mean(_logit(g["focal"].to_numpy()) - _logit(g["broad"].to_numpy())))


def power_curve(donor_counts, n_rep: int = 200, seed: int = 4,
                base_seed: int = 90000, **kw) -> pd.DataFrame:
    """Probability of detecting the assigned broad-versus-focal contrast.

    The test is a donor-paired t-test on the logit naive share, which is
    cheaper than the fitted intensity model and therefore conservative.
    """
    from scipy import stats

    rows = []
    for n_donors in donor_counts:
        hits = 0
        for r in range(n_rep):
            cells = _power_draw(n_donors, base_seed + 1000 * n_donors + r, **kw)
            g = naive_share(cells).pivot(index="donor", columns="arm", values="share")
            contrast = _logit(g["focal"].to_numpy()) - _logit(g["broad"].to_numpy())
            if contrast.size > 1 and contrast.std(ddof=1) > 0:
                _, pval = stats.ttest_1samp(contrast, 0.0)
                hits += int(pval < 0.05)
        rows.append({"donors": int(n_donors), "power": hits / n_rep, "n_rep": n_rep})
    return pd.DataFrame(rows)


def _power_draw(n_donors, seed, **kw):
    """One synthetic experiment, cells only, for the power loop."""
    import sim as _sim

    return _sim.simulate_experiment(
        seed=seed, n_donors=n_donors, arms=("broad", "focal"), **kw,
    ).cells


# ------------------------------------------- fast leave-one-donor-out scoring


def _nb_logpmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    return nbinom.logpmf(y, alpha, alpha / (alpha + mu))


def _independent_columns(X: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Indices of a maximal set of linearly independent columns, by pivoted QR.

    Three centred spline blocks plus an intercept are rank-deficient in more
    than one way here: each centred block is short one dimension by
    construction, and the coarsened grid leaves some basis functions with
    almost no data under them. Rather than guess which columns to drop, take
    the ones a rank-revealing factorisation keeps.
    """
    from scipy.linalg import qr

    _, r, piv = qr(X, mode="economic", pivoting=True)
    rank = int((np.abs(np.diag(r)) > tol * abs(r[0, 0])).sum())
    return np.sort(piv[:rank])


def glm_designs(panel: pd.DataFrame, d: Design) -> tuple[np.ndarray, np.ndarray]:
    """Design matrices for the maximum-likelihood fits, reduced to full rank.

    A centred B-spline basis is exactly rank-deficient against an intercept:
    the raw basis rows sum to one, so the centred columns sum to zero. The
    Bayesian fit tolerates that because its random-walk prior is proper; an
    unpenalised GLM does not.
    """
    Xg = np.column_stack([np.ones(len(panel))] + [d.X[s] for s in SMOOTHS])
    Xl = np.column_stack([np.ones(len(panel)), d.linear])
    return Xg[:, _independent_columns(Xg)], Xl[:, _independent_columns(Xl)]


def estimate_alpha(panel: pd.DataFrame, d: Design) -> float:
    """One dispersion estimate on the full data, reused across folds so every
    fold scores on the same scale.
    """
    import statsmodels.api as sm

    X, _ = glm_designs(panel, d)
    fitted = sm.GLM(
        d.y, X, family=sm.families.Poisson(), offset=d.log_offset,
    ).fit()
    mu = np.asarray(fitted.fittedvalues)
    excess = float((((d.y - mu) ** 2) - mu).sum())
    if excess <= 0:
        return 1e4  # indistinguishable from Poisson
    return float((mu**2).sum() / excess)


def lodo_glm(panel: pd.DataFrame, d: Design, alpha: float) -> pd.DataFrame:
    """Leave-one-donor-out log score for the spline model and its linear
    baseline, both fitted by maximum likelihood.

    This is a plug-in predictive score at the fitted coefficients, not an
    expected log predictive density: it ignores coefficient uncertainty and
    predicts a held-out donor at the average donor. It is the same
    approximation for both models, so the comparison is fair even though
    neither number is an ELPD.
    """
    import statsmodels.api as sm

    Xg, Xl = glm_designs(panel, d)
    rows = []
    for held in np.sort(panel["donor"].unique()):
        train = d.donor != held
        test = ~train
        got = {}
        for name, X in (("gam", Xg), ("linear", Xl)):
            res = sm.GLM(
                d.y[train], X[train],
                family=sm.families.NegativeBinomial(alpha=alpha),
                offset=d.log_offset[train],
            ).fit()
            mu = np.exp(X[test] @ res.params + d.log_offset[test])
            got[name] = _nb_logpmf(d.y[test], mu, alpha)
        rows.append(pd.DataFrame({"donor": held, **got}))
    return pd.concat(rows, ignore_index=True)


def score_summary(folds: pd.DataFrame) -> dict[str, float]:
    diff = folds["gam"].to_numpy() - folds["linear"].to_numpy()
    return {
        "gam": float(folds["gam"].sum()),
        "linear": float(folds["linear"].sum()),
        "diff": float(diff.sum()),
        "se": float(np.sqrt(diff.size) * diff.std(ddof=1)),
        "n": int(diff.size),
    }


def calibration_table(panel: pd.DataFrame, d: Design, alpha: float,
                      n_bins: int = 8) -> pd.DataFrame:
    """Observed against predicted counts on held-out donors, by bin of
    predicted rate.
    """
    import statsmodels.api as sm

    Xg, Xl = glm_designs(panel, d)
    out = []
    for name, X in (("gam", Xg), ("linear", Xl)):
        pred = np.zeros(len(panel))
        for held in np.sort(panel["donor"].unique()):
            train = d.donor != held
            res = sm.GLM(
                d.y[train], X[train],
                family=sm.families.NegativeBinomial(alpha=alpha),
                offset=d.log_offset[train],
            ).fit()
            pred[~train] = np.exp(X[~train] @ res.params + d.log_offset[~train])
        edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
        idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            m = idx == b
            if m.sum() < 2:
                continue
            obs = d.y[m]
            out.append({
                "model": name, "bin": b,
                "predicted": float(pred[m].mean()),
                "observed": float(obs.mean()),
                "se": float(obs.std(ddof=1) / np.sqrt(m.sum())),
                "n": int(m.sum()),
            })
    return pd.DataFrame(out)
