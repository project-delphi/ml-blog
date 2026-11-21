"""Goodness-of-fit power against held-out predictive gain (Breiman 2001, sec. 5.2).

Breiman reports a seven-dimensional simulated regression with a controlled amount
of nonlinearity, in which standard goodness-of-fit tests did not reject linearity
until the nonlinearity was extreme. He gives no further details, so the design
here was fixed before any results were seen:

- Predictors: seven real meteorological columns from the LA ozone data
  (standardised), so their correlations are those of real data.
- Response: y = sqrt(1 - eta) * L + sqrt(eta) * G + noise, where L is a linear
  function of the predictors and G a nonlinear "regime" term, both scaled to unit
  variance on the sample, and the noise has variance `noise_var`.
- G (`regime`) = relu(temp) * relu(-ibh), standardised: ozone builds when it is hot *and*
  the inversion base is low. eta is the share of signal variance that is
  nonlinear.
- Tests on the OLS fit: Ramsey's RESET (fitted-value powers 2 and 3), Utts'
  rainbow test, and an omnibus F test of all 35 squares and cross-products.
- Predictive comparison: OLS against a random forest on a random two-thirds /
  one-third split of the rows, scored by test mean squared error.
"""

import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.diagnostic import linear_rainbow, linear_reset


def _standardise(v):
    return (v - v.mean()) / v.std()


SHAPES = ("regime", "wiggle")


def nonlinear(Z, cols, shape):
    """The unit-variance nonlinear term.

    `regime` is the design fixed in advance. `wiggle`, sin(3 * temp), was added
    after the regime results were seen, to look for the case Breiman describes:
    a nonlinearity that squares and fitted-value powers do not point at.
    """
    t, b = Z[:, cols.index("temp")], Z[:, cols.index("ibh")]
    if shape == "regime":
        return _standardise(np.maximum(t, 0) * np.maximum(-b, 0))
    if shape == "wiggle":
        return _standardise(np.sin(3 * t))
    raise ValueError(shape)


def signal(Z, cols, eta, beta, shape="regime"):
    """Unit-variance signal at nonlinearity share eta."""
    L = _standardise(Z @ beta)
    return np.sqrt(1 - eta) * L + np.sqrt(eta) * nonlinear(Z, cols, shape)


def quad_f_pvalue(Z, y):
    """F test that all squares and cross-products add nothing to a linear fit."""
    lin = sm.add_constant(Z)
    quad = PolynomialFeatures(2, include_bias=False).fit_transform(Z)
    full = sm.add_constant(quad)
    r0 = sm.OLS(y, lin).fit()
    r1 = sm.OLS(y, full).fit()
    return float(r1.compare_f_test(r0)[1])


def one_replicate(Z, cols, eta, beta, noise_var, rep, seed=0, shape="regime"):
    rng = np.random.default_rng([seed, int(round(1000 * eta)), rep])
    n = len(Z)
    y = signal(Z, cols, eta, beta, shape) + rng.normal(0, np.sqrt(noise_var), n)
    ols = sm.OLS(y, sm.add_constant(Z)).fit()
    p_reset = float(linear_reset(ols, power=3, use_f=True).pvalue)
    p_rainbow = float(linear_rainbow(ols)[1])
    p_quad = quad_f_pvalue(Z, y)
    idx = rng.permutation(n)
    tr, te = idx[: 2 * n // 3], idx[2 * n // 3 :]
    lr = LinearRegression().fit(Z[tr], y[tr])
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=rep, n_jobs=1
    ).fit(Z[tr], y[tr])
    mse_lr = float(np.mean((lr.predict(Z[te]) - y[te]) ** 2))
    mse_rf = float(np.mean((rf.predict(Z[te]) - y[te]) ** 2))
    return {
        "shape": shape,
        "eta": eta,
        "rep": rep,
        "p_reset": p_reset,
        "p_rainbow": p_rainbow,
        "p_quad": p_quad,
        "mse_ols": mse_lr,
        "mse_rf": mse_rf,
        "r2_ols": float(ols.rsquared),
    }


def sweep(Z, cols, etas, reps, noise_var=0.5, seed=0, shapes=SHAPES):
    """All replicates at all nonlinearity levels, as a list of dicts."""
    beta = np.linspace(1.0, -0.5, Z.shape[1])
    return Parallel(n_jobs=-1)(
        delayed(one_replicate)(Z, cols, e, beta, noise_var, r, seed, s)
        for s in shapes
        for e in etas
        for r in range(reps)
    )


def residual_sample(Z, cols, eta, noise_var=0.5, seed=0, shape="regime"):
    """Fitted values and residuals of the OLS fit for one replicate (for plots)."""
    beta = np.linspace(1.0, -0.5, Z.shape[1])
    rng = np.random.default_rng([seed, int(round(1000 * eta)), 0])
    y = signal(Z, cols, eta, beta, shape) + rng.normal(0, np.sqrt(noise_var), len(Z))
    ols = sm.OLS(y, sm.add_constant(Z)).fit()
    return ols.fittedvalues, ols.resid


COLS = ["vh", "wind", "humidity", "temp", "ibh", "dpg", "vis"]
ETAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def predictors():
    """The seven standardised meteorological columns of faraway::ozone."""
    from . import data

    f = data.ozone_faraway()
    Z = f[COLS].to_numpy(float)
    return (Z - Z.mean(0)) / Z.std(0)


def run(reps: int = 200):
    """The full sweep used by Part 1, as a DataFrame."""
    import pandas as pd

    return pd.DataFrame(sweep(predictors(), COLS, ETAS, reps))
