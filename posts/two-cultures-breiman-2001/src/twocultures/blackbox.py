"""Part 4's experiments: information from a black box (Breiman 2001, sec. 11).

`OOBForest` is a plain random forest that keeps each tree's out-of-bag cases, so
it can compute Breiman's importance measure: for each tree, permute one
variable among the cases the tree did not see, and record the rise in their
misclassification rate. Only variables a tree splits on can change its
predictions, so the others are skipped (their contribution is zero).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import data, models


class OOBForest:
    def __init__(self, n_trees=1000, max_features="sqrt", seed=0, n_jobs=-1):
        self.n_trees, self.max_features, self.seed, self.n_jobs = (
            n_trees,
            max_features,
            seed,
            n_jobs,
        )

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        boots = [rng.integers(0, n, n) for _ in range(self.n_trees)]
        seeds = rng.integers(0, 2**31 - 1, self.n_trees)

        def grow(b, s):
            t = DecisionTreeClassifier(
                max_features=self.max_features, random_state=int(s)
            )
            return t.fit(X[b], y[b])

        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(grow)(b, s) for b, s in zip(boots, seeds)
        )
        self.oob_ = [np.setdiff1d(np.arange(n), b) for b in boots]
        self.X_, self.y_ = X, y
        return self

    def _votes(self, X):
        v = np.zeros((len(X), len(self.classes_)))
        for t in self.trees_:
            p = t.predict_proba(X)
            v[:, np.searchsorted(self.classes_, t.classes_)] += p
        return v / len(self.trees_)

    def predict_proba(self, X):
        return self._votes(np.asarray(X, float))

    def predict(self, X):
        return self.classes_[self._votes(np.asarray(X, float)).argmax(1)]

    def oob_proba(self):
        n, k = len(self.y_), len(self.classes_)
        s, c = np.zeros((n, k)), np.zeros(n)
        for t, o in zip(self.trees_, self.oob_):
            p = t.predict_proba(self.X_[o])
            s[np.ix_(o, np.searchsorted(self.classes_, t.classes_))] += p
            c[o] += 1
        return s / np.maximum(c, 1)[:, None]

    def oob_error(self):
        return float(np.mean(self.classes_[self.oob_proba().argmax(1)] != self.y_))

    def permutation_importance(self, groups=None, conditional=None, seed=0):
        """Percent rise in OOB error when a variable (or group) is permuted.

        `groups`: dict name -> list of column indices permuted together.
        `conditional`: dict j -> column index c; permute j only within the
        levels of c (Strobl et al. 2008 with a one-variable conditioning set).
        """
        rng = np.random.default_rng(seed)
        p = self.X_.shape[1]
        groups = groups or {j: [j] for j in range(p)}
        base_err, perm_err = 0.0, {g: 0.0 for g in groups}
        for t, o in zip(self.trees_, self.oob_):
            if len(o) == 0:
                continue
            Xo, yo = self.X_[o], self.y_[o]
            e0 = np.mean(t.predict(Xo) != yo)
            base_err += e0
            used = set(t.tree_.feature[t.tree_.feature >= 0].tolist())
            for g, cols in groups.items():
                if not used.intersection(cols):
                    perm_err[g] += e0
                    continue
                Z = Xo.copy()
                if conditional and g in conditional:
                    c = conditional[g]
                    for lev in np.unique(Xo[:, c]):
                        m = np.where(Xo[:, c] == lev)[0]
                        Z[m[:, None], cols] = Xo[rng.permutation(m)[:, None], cols]
                else:
                    perm = rng.permutation(len(o))
                    Z[:, cols] = Xo[perm][:, cols]
                perm_err[g] += np.mean(t.predict(Z) != yo)
        return {g: 100 * (perm_err[g] - base_err) / base_err for g in groups}

    def impurity_importance(self):
        return np.mean([t.feature_importances_ for t in self.trees_], axis=0)


# ------------------------------------------------------------------ helpers
def logit():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1e6, max_iter=5000),
    )


def rf(seed):
    return models.forest(seed=seed, n_estimators=500)


def holdout(models_, X, y, reps=100, seed=0):
    return pd.DataFrame(models.repeated_holdout(models_, X, y, reps=reps, seed=seed))


def standardized_coefficients(X, y):
    """Coefficient / standard error of a logistic fit on median-imputed data."""
    Z = SimpleImputer(strategy="median").fit_transform(X)
    Z = (Z - Z.mean(0)) / Z.std(0)
    r = sm.Logit(y, sm.add_constant(Z)).fit(disp=0, maxiter=200)
    return np.asarray(r.params[1:] / r.bse[1:])


def logit_test_permutation(X, y, reps=100, seed=0):
    """Breiman's check: permutation importance for logistic regression, scored on
    the held-out 10% of each repetition.
    """
    p = X.shape[1]

    def one(r):
        g = np.random.default_rng([seed, r])
        te = g.choice(len(y), size=round(0.1 * len(y)), replace=False)
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = logit().fit(X[tr], y[tr])
        e0 = np.mean(m.predict(X[te]) != y[te])
        out = []
        for j in range(p):
            Z = X[te].copy()
            Z[:, j] = Z[g.permutation(len(te)), j]
            out.append(np.mean(m.predict(Z) != y[te]) - e0)
        return out

    return np.array(Parallel(n_jobs=-1)(delayed(one)(r) for r in range(reps))).mean(0)


def forward_stepwise_logit(X, y, alpha=0.05):
    """Forward selection by likelihood-ratio test at level alpha."""
    Z = SimpleImputer(strategy="median").fit_transform(X)
    chosen = []

    def llf(cols):
        A = sm.add_constant(Z[:, cols]) if cols else np.ones((len(y), 1))
        try:
            return sm.Logit(y, A).fit(disp=0, maxiter=200).llf
        except Exception:  # perfect separation or singular fit
            return np.nan

    cur = llf([])
    while True:
        best, best_p, best_ll = None, alpha, None
        for j in range(Z.shape[1]):
            if j in chosen:
                continue
            ll = llf(chosen + [j])
            if not np.isfinite(ll):
                continue
            pval = stats.chi2.sf(2 * (ll - cur), 1)
            if pval < best_p:
                best, best_p, best_ll = j, pval, ll
        if best is None:
            return chosen
        chosen.append(best)
        cur = best_ll


def bootstrap_forward(X, y, boots=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(boots)]
    return Parallel(n_jobs=-1)(delayed(forward_stepwise_logit)(X[i], y[i]) for i in idx)


def hepatitis_with_noise(seed=0):
    """Hepatitis plus two pure-noise columns: one continuous, one binary."""
    X, y, names = data.hepatitis()
    rng = np.random.default_rng(seed)
    noise = np.c_[rng.normal(size=len(y)), rng.integers(1, 3, len(y))]
    return np.c_[X, noise], y, names + ["NOISE_CONT", "NOISE_BIN"]


def importance_replicates(boots=30, n_trees=500, seed=0):
    """Four importance measures on bootstrap resamples of hepatitis + noise."""
    X, y, names = hepatitis_with_noise(seed)
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    asc, alb = names.index("ASCITES"), names.index("ALBUMIN")
    rows = []
    rng = np.random.default_rng(seed)
    for b in range(boots + 1):
        idx = np.arange(len(y)) if b == 0 else rng.integers(0, len(y), len(y))
        f = OOBForest(n_trees=n_trees, seed=b).fit(Xi[idx], y[idx])
        perm = f.permutation_importance(seed=b)
        cond = f.permutation_importance(
            groups={alb: [alb], asc: [asc]}, conditional={alb: asc}, seed=b
        )
        pair = f.permutation_importance(groups={"pair": [asc, alb]}, seed=b)["pair"]
        mdi = f.impurity_importance()
        for j, nm in enumerate(names):
            rows.append(
                {
                    "boot": b,
                    "var": nm,
                    "j": j,
                    "mdi": float(mdi[j]),
                    "perm": float(perm[j]),
                    "cond": float(cond[j]) if j == alb else np.nan,
                }
            )
        rows.append(
            {
                "boot": b,
                "var": "ASCITES+ALBUMIN",
                "j": -1,
                "mdi": np.nan,
                "perm": float(pair),
                "cond": np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- bupa
BUPA_COLS = ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks"]


def proximity(forest: OOBForest, X):
    """Breiman's proximity: share of trees in which two cases share a leaf."""
    X = np.asarray(X, float)
    leaves = np.array([t.apply(X) for t in forest.trees_]).T
    n = len(X)
    P = np.zeros((n, n))
    for k in range(leaves.shape[1]):
        P += leaves[:, k][:, None] == leaves[:, k][None, :]
    return P / leaves.shape[1]
