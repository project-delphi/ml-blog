"""Four ways to localise a classification onto the parts of a tabular dataset.

The parallel to :mod:`attribution` is exact, and the post leans on it:

======================== ================== ==================================
function                 part               responsible means
======================== ================== ==================================
``fit_logistic``         feature            its coefficient is far from zero,
                                            relative to its standard error
``anova_table``          feature            it explains a large share of
                                            between-species variance
``pca_fit``              linear combination it carries a large share of total
                                            variance
``permutation_importance``  feature         shuffling it costs the model
                                            accuracy
======================== ================== ==================================

Only the first three are classical. The fourth is the ML tradition's answer, and
it is here to show that the two traditions differ less in what they ask than in
whether they report uncertainty about the answer.

Every estimator is written out rather than called, and cross-checked against the
library implementation at build time — see ``self_check``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris

FEATURES = ["sepal length", "sepal width", "petal length", "petal width"]
SPECIES = ["setosa", "versicolor", "virginica"]


def load() -> tuple[pd.DataFrame, np.ndarray]:
    """Load Iris as a tidy frame plus an integer target.

    Returns:
        Tuple of (frame with the four measurements in cm and a ``species``
        column, integer species codes in 0/1/2 order matching SPECIES).
    """
    raw = load_iris()
    df = pd.DataFrame(raw.data, columns=FEATURES)
    df["species"] = [SPECIES[i] for i in raw.target]
    return df, raw.target


def standardise(X: np.ndarray) -> np.ndarray:
    """Centre and scale each column to mean 0, sd 1.

    Coefficients on raw centimetres are not comparable across features, because
    sepal length varies over a range three times wider than petal width. On the
    standardised scale a coefficient is the change in log-odds per standard
    deviation, which is the scale on which "which feature matters more" is a
    well-posed question.

    Args:
        X: Design matrix, shape (n, p).

    Returns:
        Standardised matrix of the same shape.
    """
    return (X - X.mean(0)) / X.std(0, ddof=1)


# --------------------------------------------------------------------------
# 1. Logistic regression: coefficients with standard errors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LogitFit:
    """A fitted binary logistic regression and its Wald inference."""

    beta: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    iterations: int
    converged: bool
    loglik: float
    names: list[str]

    @property
    def z(self) -> np.ndarray:
        """Wald z statistic per coefficient."""
        return self.beta / self.se

    @property
    def pvalue(self) -> np.ndarray:
        """Two-sided Wald p-value per coefficient."""
        return 2 * stats.norm.sf(np.abs(self.z))

    def ci(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Wald confidence interval per coefficient.

        Args:
            level: Coverage level.

        Returns:
            Tuple of (lower bounds, upper bounds).
        """
        q = stats.norm.ppf(0.5 + level / 2)
        return self.beta - q * self.se, self.beta + q * self.se

    def table(self, level: float = 0.95) -> pd.DataFrame:
        """Summarise the fit as a coefficient table.

        Args:
            level: Coverage level for the interval columns.

        Returns:
            Frame with one row per coefficient.
        """
        lo, hi = self.ci(level)
        return pd.DataFrame(
            {
                "term": self.names,
                "coef": self.beta,
                "std err": self.se,
                "z": self.z,
                "p": self.pvalue,
                "lo": lo,
                "hi": hi,
            }
        )


def fit_logistic(
    X: np.ndarray, y: np.ndarray, names: list[str], max_iter: int = 100, tol: float = 1e-10
) -> LogitFit:
    """Fit a binary logistic regression by Newton-Raphson, unpenalised.

    The standard errors come from the observed information: at the maximum, the
    log-likelihood's curvature is ``X'WX`` with ``W = diag(p(1-p))``, and its
    inverse is the asymptotic covariance of the coefficients. Sharp curvature —
    the data pinning the coefficient down — means small standard errors. That is
    the whole content of a confidence interval on a coefficient, and it is what
    the attribution methods in Part I have no analogue of.

    No penalty is applied, deliberately. Ridge would keep every coefficient
    finite even where the likelihood has no maximum, and hide the failure the
    post is about.

    Args:
        X: Design matrix without an intercept column, shape (n, p).
        y: Binary response in {0, 1}, shape (n,).
        names: Names for the p features; an "intercept" term is prepended.
        max_iter: Maximum Newton steps.
        tol: Convergence threshold on the coefficient increment.

    Returns:
        The fit, its covariance, and whether Newton converged.
    """
    Xd = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xd.shape[1])
    converged, used = False, max_iter

    for step in range(max_iter):
        eta = Xd @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1.0 - p)
        # Ridge-free Newton step. pinv rather than solve: on separable data the
        # information matrix goes singular, and we want the run to continue and
        # report a diverging coefficient rather than raise.
        hessian = Xd.T @ (W[:, None] * Xd)
        score = Xd.T @ (y - p)
        delta = np.linalg.pinv(hessian) @ score
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            converged, used = True, step + 1
            break

    eta = Xd @ beta
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-15, 1 - 1e-15)
    W = p * (1.0 - p)
    cov = np.linalg.pinv(Xd.T @ (W[:, None] * Xd))
    loglik = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return LogitFit(
        beta=beta,
        se=np.sqrt(np.diag(cov)),
        cov=cov,
        iterations=used,
        converged=converged,
        loglik=loglik,
        names=["intercept"] + list(names),
    )


def separation_path(X: np.ndarray, y: np.ndarray, caps: tuple[int, ...]) -> pd.DataFrame:
    """Refit with increasing Newton budgets and watch the estimates move.

    On separable data the maximum likelihood estimate does not exist: the
    likelihood increases without bound as the coefficients grow, so every extra
    iteration buys a larger coefficient and a larger standard error, and neither
    settles. This function makes that visible instead of asserting it.

    Args:
        X: Design matrix, shape (n, p).
        y: Binary response in {0, 1}.
        caps: Iteration budgets to try.

    Returns:
        Frame with one row per budget: coefficient norm, largest standard error,
        log-likelihood, and the fitted probability closest to a mistake.
    """
    rows = []
    for cap in caps:
        fit = fit_logistic(X, y, FEATURES, max_iter=cap, tol=0.0)
        rows.append(
            {
                # Not "||coef||": the pipes would be read as cell delimiters
                # when this frame is rendered into a Markdown table.
                "newton steps": cap,
                "coefficient norm": float(np.linalg.norm(fit.beta[1:])),
                "largest std err": float(fit.se.max()),
                "log-likelihood": fit.loglik,
            }
        )
    return pd.DataFrame(rows)


def is_separable(X: np.ndarray, y: np.ndarray) -> bool:
    """Report whether the two classes are linearly separable.

    Solves the linear feasibility problem directly rather than inferring
    separation from a diverging fit.

    Args:
        X: Design matrix, shape (n, p).
        y: Binary response in {0, 1}.

    Returns:
        True if some hyperplane separates the classes with a positive margin.
    """
    from scipy.optimize import linprog

    s = np.where(y == 1, 1.0, -1.0)[:, None]
    A = -s * np.column_stack([np.ones(len(X)), X])  # want A @ w <= -margin
    n_var = A.shape[1]
    # Maximise the margin m subject to s_i (w'x_i + b) >= m, |w| bounded.
    c = np.zeros(n_var + 1)
    c[-1] = -1.0
    A_ub = np.column_stack([A, np.ones(len(A))])
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=np.zeros(len(A)),
        bounds=[(-1, 1)] * n_var + [(0, 1)],
        method="highs",
    )
    return bool(res.success and res.x[-1] > 1e-9)


# --------------------------------------------------------------------------
# 2. ANOVA: variance decomposition
# --------------------------------------------------------------------------


def anova_table(df: pd.DataFrame) -> pd.DataFrame:
    """One-way ANOVA of each measurement across the three species.

    The decomposition is the identity ``SS_total = SS_between + SS_within``: the
    spread of a measurement splits into the part explained by knowing the
    species and the part left over. ``eta^2`` is the first as a fraction of the
    total — a variance-share attribution, computed here from the sums of squares
    rather than read off a library.

    Args:
        df: Frame with the four measurement columns and a ``species`` column.

    Returns:
        Frame with one row per feature: F, degrees of freedom, p, and eta^2.
    """
    rows = []
    for feature in FEATURES:
        groups = [df.loc[df.species == s, feature].to_numpy() for s in SPECIES]
        values = np.concatenate(groups)
        grand = values.mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
        df_between = len(groups) - 1
        df_within = len(values) - len(groups)
        f = (ss_between / df_between) / (ss_within / df_within)
        rows.append(
            {
                "feature": feature,
                "F": f,
                "df": f"{df_between}, {df_within}",
                "p": float(stats.f.sf(f, df_between, df_within)),
                "eta^2": ss_between / (ss_between + ss_within),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. PCA: variance localised to directions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PCAFit:
    """A principal component decomposition and the scores it induces."""

    ratio: np.ndarray
    loadings: np.ndarray  # (p, k): columns are components in feature space
    scores: np.ndarray  # (n, k)
    standardised: bool


def pca_fit(X: np.ndarray, standardised: bool = True, k: int = 2) -> PCAFit:
    """Principal components by SVD of the centred (optionally scaled) matrix.

    PCA changes what a *part* is. The parts are no longer the four measurements
    but linear combinations of them, chosen so that each successive one captures
    as much residual variance as possible. That buys concentration — most of the
    variance in few parts — at the cost of interpretability, since a component is
    only as nameable as its loadings allow.

    Args:
        X: Data matrix, shape (n, p).
        standardised: Whether to scale columns to unit variance first.
        k: Number of components to return scores and loadings for.

    Returns:
        Explained-variance ratios for all components, plus the first k loadings
        and scores.
    """
    Z = standardise(X) if standardised else X - X.mean(0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = S**2 / (len(Z) - 1)
    # Sign convention: make each component's largest-magnitude loading positive,
    # so the biplot does not flip between runs or between standardisations.
    V = Vt.T
    flip = np.sign(V[np.abs(V).argmax(0), np.arange(V.shape[1])])
    V = V * flip
    scores = Z @ V
    return PCAFit(
        ratio=var / var.sum(),
        loadings=V[:, :k],
        scores=scores[:, :k],
        standardised=standardised,
    )


# --------------------------------------------------------------------------
# 4. Permutation importance: occlusion for tables
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """A stratified train/test split and the model fitted on it."""

    model: object
    Xtr: np.ndarray
    Xte: np.ndarray
    ytr: np.ndarray
    yte: np.ndarray

    @property
    def test_accuracy(self) -> float:
        """Accuracy of the fitted model on the held-out half."""
        return float((self.model.predict(self.Xte) == self.yte).mean())


def fit_multinomial(seed: int, test_size: float = 0.4) -> Split:
    """Fit a three-class logistic regression on a stratified split.

    Penalised, unlike :func:`fit_logistic`. The L2 penalty is what keeps setosa's
    coefficients finite despite its separability — the price is that the
    coefficients are no longer maximum-likelihood estimates and their standard
    errors no longer mean what a Wald interval assumes.

    Args:
        seed: Seed for the split.
        test_size: Fraction held out.

    Returns:
        The split and the fitted model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    df, target = load()
    X = standardise(df[FEATURES].to_numpy())
    Xtr, Xte, ytr, yte = train_test_split(
        X, target, test_size=test_size, random_state=seed, stratify=target
    )
    model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return Split(model=model, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)


def permutation_importance(
    predict, X: np.ndarray, y: np.ndarray, repeats: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Accuracy lost when each feature's column is shuffled.

    This is occlusion, moved to a table. Breaking a feature's link to the label
    while keeping its marginal distribution intact, and measuring what the model
    loses, is the same "responsibility = effect of removal" definition that the
    occlusion map uses on pixels. The difference is that the shuffle is random,
    so repeating it gives a *distribution* of importances rather than a number —
    which is how a removal-based ML attribution acquires an error bar.

    Args:
        predict: Callable mapping a design matrix to predicted labels.
        X: Held-out design matrix, shape (n, p).
        y: Held-out labels.
        repeats: Number of independent shuffles per feature.
        rng: Generator supplying the permutations.

    Returns:
        Frame with one row per feature: mean drop, its standard deviation, and
        the standard error of the mean.
    """
    baseline = float((predict(X) == y).mean())
    rows = []
    for j, name in enumerate(FEATURES):
        drops = np.empty(repeats)
        for r in range(repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            drops[r] = baseline - float((predict(Xp) == y).mean())
        rows.append(
            {
                "feature": name,
                "mean drop": drops.mean(),
                "sd": drops.std(ddof=1),
                "se": drops.std(ddof=1) / np.sqrt(repeats),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Build-time cross-checks
# --------------------------------------------------------------------------


def self_check() -> list[str]:
    """Verify every hand-written estimator against its library counterpart.

    Returns:
        One human-readable line per check, each stating the agreement achieved.

    Raises:
        AssertionError: If any check fails, which fails the render.
    """
    import statsmodels.api as sm
    from sklearn.decomposition import PCA

    df, target = load()
    X = df[FEATURES].to_numpy()
    lines = []

    # 1. Coefficients and standard errors, versicolor vs virginica.
    mask = target > 0
    Xs = standardise(X[mask])
    y = (target[mask] == 2).astype(float)
    mine = fit_logistic(Xs, y, FEATURES)
    theirs = sm.Logit(y, sm.add_constant(Xs)).fit(disp=0, method="newton")
    d_beta = float(np.max(np.abs(mine.beta - theirs.params)))
    d_se = float(np.max(np.abs(mine.se - theirs.bse)))
    assert d_beta < 1e-6 and d_se < 1e-6, (d_beta, d_se)
    lines.append(
        f"logistic coefficients match statsmodels to {d_beta:.1e}, "
        f"standard errors to {d_se:.1e}"
    )

    # 2. ANOVA F statistics.
    table = anova_table(df)
    worst = 0.0
    for _, row in table.iterrows():
        groups = [df.loc[df.species == s, row.feature].to_numpy() for s in SPECIES]
        worst = max(worst, abs(row.F - stats.f_oneway(*groups).statistic) / row.F)
    assert worst < 1e-10, worst
    lines.append(f"ANOVA F statistics match scipy.stats.f_oneway to {worst:.1e} relative")

    # 3. Explained-variance ratios.
    fit = pca_fit(X, standardised=True, k=4)
    ref = PCA().fit(standardise(X))
    d_pca = float(np.max(np.abs(fit.ratio - ref.explained_variance_ratio_)))
    assert d_pca < 1e-12, d_pca
    lines.append(f"PCA explained-variance ratios match scikit-learn to {d_pca:.1e}")

    # 4. Separation is detected by the feasibility program, not inferred.
    y_setosa = (target == 0).astype(float)
    assert is_separable(standardise(X), y_setosa)
    assert not is_separable(Xs, y)
    lines.append("setosa is linearly separable from the rest; versicolor and virginica are not")

    # 5. Permutation importance against scikit-learn's implementation. The two
    # differ in their random draws, so they are compared within Monte-Carlo
    # error rather than to machine precision.
    from sklearn.inspection import permutation_importance as sk_perm

    split = fit_multinomial(seed=0)
    mine = permutation_importance(
        split.model.predict, split.Xte, split.yte, repeats=200, rng=np.random.default_rng(0)
    )
    ref = sk_perm(split.model, split.Xte, split.yte, n_repeats=200, random_state=0,
                  scoring="accuracy")
    gap = np.abs(mine["mean drop"].to_numpy() - ref.importances_mean)
    tolerance = 3 * np.hypot(mine["se"].to_numpy(), ref.importances_std / np.sqrt(200))
    assert np.all(gap < np.maximum(tolerance, 1e-12)), (gap, tolerance)
    lines.append(
        "permutation importances agree with scikit-learn's within Monte-Carlo error "
        f"(largest gap {gap.max():.4f}, largest 3-sigma bound {tolerance.max():.4f})"
    )

    return lines


if __name__ == "__main__":
    for line in self_check():
        print("ok:", line)
