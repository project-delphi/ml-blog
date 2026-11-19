"""Model factories and evaluation protocols used across the series.

Tuning always happens inside the training data passed to `fit`: the pruned tree
picks its complexity by cross-validation on that data alone, so a held-out test
set never influences a model fitted to predict it.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


class PrunedTree(BaseEstimator, ClassifierMixin):
    """CART with cost-complexity pruning chosen by the one-standard-error rule.

    This follows Breiman, Friedman, Olshen and Stone (1984): grow a large tree,
    compute the nested sequence of pruned subtrees, estimate each subtree's error
    by V-fold cross-validation on the training data, and keep the smallest tree
    whose error is within one standard error of the minimum.
    """

    def __init__(self, folds: int = 10, max_alphas: int = 30, random_state: int = 0):
        self.folds = folds
        self.max_alphas = max_alphas
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        base = DecisionTreeClassifier(random_state=self.random_state)
        alphas = base.cost_complexity_pruning_path(X, y).ccp_alphas
        # CART evaluates each subtree at the geometric midpoint of its alpha interval.
        mids = np.sqrt(alphas[:-1] * alphas[1:]) if len(alphas) > 1 else alphas
        mids = np.concatenate([[0.0], mids[mids > 0]])
        if len(mids) > self.max_alphas:
            mids = mids[np.linspace(0, len(mids) - 1, self.max_alphas).astype(int)]
        # The last path alpha prunes to the root alone. Midpoints stop short of
        # it, so add it, after thinning so that no other candidate moves: the
        # majority-class stump must be selectable when it is within one SE.
        if len(alphas) and alphas[-1] > mids[-1]:
            mids = np.append(mids, alphas[-1])
        _, counts = np.unique(y, return_counts=True)
        k = int(min(self.folds, counts.min())) if counts.min() >= 2 else 2
        cv = StratifiedKFold(k, shuffle=True, random_state=self.random_state)
        errs = np.zeros((k, len(mids)))
        for f, (tr, te) in enumerate(cv.split(X, y)):
            for j, a in enumerate(mids):
                t = DecisionTreeClassifier(ccp_alpha=a, random_state=self.random_state)
                errs[f, j] = np.mean(t.fit(X[tr], y[tr]).predict(X[te]) != y[te])
        mean = errs.mean(0)
        se = errs.std(0, ddof=1) / np.sqrt(k)
        best = np.argmin(mean)
        # Largest alpha (smallest tree) within one SE of the minimum.
        ok = np.where(mean <= mean[best] + se[best])[0]
        self.alpha_ = float(mids[ok.max()])
        self.tree_ = DecisionTreeClassifier(
            ccp_alpha=self.alpha_, random_state=self.random_state
        ).fit(X, y)
        self.classes_ = self.tree_.classes_
        return self

    def predict(self, X):
        return self.tree_.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.tree_.predict_proba(np.asarray(X))


def tree(seed: int = 0):
    """A median-imputed, CV-pruned CART classifier."""
    return make_pipeline(
        SimpleImputer(strategy="median"), PrunedTree(random_state=seed)
    )


def forest(seed: int = 0, n_estimators: int = 500, n_jobs: int = 1, **kw):
    """A median-imputed random forest with Breiman's default sqrt(p) split candidates."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs, **kw
        ),
    )


def _one_holdout(models: dict, X, y, rep: int, test_frac: float, seed: int) -> dict:
    rng = np.random.default_rng([seed, rep])
    n = len(y)
    test = rng.choice(n, size=max(1, round(test_frac * n)), replace=False)
    train = np.setdiff1d(np.arange(n), test)
    out = {"rep": rep}
    for name, make in models.items():
        m = make(rep)
        m.fit(X[train], y[train])
        out[name] = float(np.mean(m.predict(X[test]) != y[test]))
    return out


def repeated_holdout(
    models: dict, X, y, reps: int = 100, test_frac: float = 0.1, seed: int = 0
) -> list[dict]:
    """Breiman's protocol for the small data sets: leave out a random 10%, fit on
    the other 90%, score the 10%, and repeat.

    `models` maps a name to a factory `rep -> unfitted estimator`. Returns one
    dict of test errors per repetition; every model sees the same splits.
    """
    X, y = np.asarray(X), np.asarray(y)
    return Parallel(n_jobs=-1)(
        delayed(_one_holdout)(models, X, y, r, test_frac, seed) for r in range(reps)
    )


def test_error(model, Xtr, ytr, Xte, yte) -> float:
    """Misclassification rate on a fixed test set."""
    m = clone(model).fit(Xtr, ytr)
    return float(np.mean(m.predict(Xte) != np.asarray(yte)))
