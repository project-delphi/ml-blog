"""Part 2's experiments: many good models (Breiman 2001, sec. 8).

1. `ozone_subsets`: every five-variable regression on the 30 ozone columns.
2. `tree_perturbations`: pruned trees on hepatitis training sets with 3% removed.
3. `net_restarts`: a small network refitted from 100 random starts on three ozone
   predictors.
4. `bagged_logistic`: best-subset logistic regression (19 -> 4 covariates) on
   hepatitis, single against bagged, in Breiman's repeated 10% holdouts.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data, models, ozone, rashomon


# ------------------------------------------------------------------ subsets
def ozone_subsets():
    df = ozone.design()
    Xtr, ytr, Xte, yte = ozone.split(df)
    fits = rashomon.subset_fits(Xtr.to_numpy(), ytr, Xte.to_numpy(), yte, 5)
    return list(Xtr.columns), fits


# -------------------------------------------------------------------- trees
def _tree_signature(tree, names, depth=2):
    """Split variables of the top `depth` levels, in breadth-first order."""
    t = tree.tree_
    out, frontier = [], [0]
    for _ in range(depth):
        nxt = []
        for node in frontier:
            if t.children_left[node] == -1:
                out.append("leaf")
                continue
            out.append(names[t.feature[node]])
            nxt += [t.children_left[node], t.children_right[node]]
        frontier = nxt
    return " / ".join(out)


def tree_structure(tree, names, max_depth=3):
    """Nested dict of the top levels of a fitted tree, for the widget."""
    t = tree.tree_

    def node(i, d):
        counts = t.value[i][0]
        p_die = float(counts[1] / counts.sum())
        n = int(t.n_node_samples[i])
        if t.children_left[i] == -1 or d == max_depth:
            return {
                "leaf": True,
                "p": round(p_die, 3),
                "n": n,
                "deeper": bool(t.children_left[i] != -1),
            }
        return {
            "leaf": False,
            "var": names[t.feature[i]],
            "thr": round(float(t.threshold[i]), 2),
            "p": round(p_die, 3),
            "n": n,
            "l": node(t.children_left[i], d + 1),
            "r": node(t.children_right[i], d + 1),
        }

    return node(0, 0)


def tree_perturbations(reps=100, drop=0.03, test_frac=0.3, seed=0):
    """Remove a random `drop` share of the training set, refit, repeat."""
    X, y, names = data.hepatitis()
    rng = np.random.default_rng(seed)
    test = rng.choice(len(y), size=round(test_frac * len(y)), replace=False)
    train = np.setdiff1d(np.arange(len(y)), test)
    imp = SimpleImputer(strategy="median").fit(X[train])
    Xi = imp.transform(X)

    def one(r):
        g = np.random.default_rng([seed, r])
        keep = (
            train
            if r == 0
            else np.sort(
                g.choice(
                    train, size=len(train) - round(drop * len(train)), replace=False
                )
            )
        )
        m = models.PrunedTree(random_state=0).fit(Xi[keep], y[keep])
        err = float(np.mean(m.predict(Xi[test]) != y[test]))
        return {
            "rep": r,
            "removed": int(len(train) - len(keep)),
            "test_error": err,
            "leaves": int(m.tree_.get_n_leaves()),
            "signature": _tree_signature(m.tree_, names),
            "root": names[m.tree_.tree_.feature[0]]
            if m.tree_.tree_.node_count > 1
            else "leaf",
            "structure": tree_structure(m.tree_, names),
        }

    rows = Parallel(n_jobs=-1)(delayed(one)(r) for r in range(reps + 1))
    return rows, {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_deaths": int(y[test].sum()),
    }


# --------------------------------------------------------------------- nets
NET_COLS = ["temp_sandburg_lag0", "ibh_lag0", "dpg_lag0"]


def net_restarts(reps=100, hidden=3, alpha=0.01):
    """One small tanh network, `reps` random starts, same data each time."""
    df = ozone.design()
    Xtr, ytr, Xte, yte = ozone.split(df)
    A, B = Xtr[NET_COLS].to_numpy(), Xte[NET_COLS].to_numpy()
    sc = StandardScaler().fit(A)
    A, B = sc.transform(A), sc.transform(B)
    my, sy = ytr.mean(), ytr.std()

    def one(r):
        m = MLPRegressor(
            hidden_layer_sizes=(hidden,),
            activation="tanh",
            solver="lbfgs",
            alpha=alpha,
            max_iter=20000,
            tol=1e-10,
            random_state=r,
        )
        m.fit(A, (ytr - my) / sy)
        tr = m.predict(A) * sy + my
        te = m.predict(B) * sy + my
        return {
            "rep": r,
            "train_mse": float(np.mean((tr - ytr) ** 2)),
            "test_mse": float(np.mean((te - yte) ** 2)),
            "fit_train": tr.tolist(),
            "pd_temp": _pd(m, A, 0, sy, my),
        }

    return Parallel(n_jobs=-1)(delayed(one)(r) for r in range(reps))


def _pd(m, A, j, sy, my, n=21):
    vals = np.linspace(-2, 2, n)
    out = []
    for v in vals:
        Z = A.copy()
        Z[:, j] = v
        out.append(float(m.predict(Z).mean() * sy + my))
    return out


def distinct_minima(rows, tol=0.1):
    """Group restarts whose fitted values on the training days differ by less
    than `tol` (RMS, in ozone units): two starts that land on the same minimum
    give the same function on the data, whatever their weights. Returns a group
    label per restart.
    """
    G = np.array([r["fit_train"] for r in rows])
    labels = -np.ones(len(G), dtype=int)
    reps = []
    for i in np.argsort([r["train_mse"] for r in rows]):
        for k, j in enumerate(reps):
            if np.sqrt(np.mean((G[i] - G[j]) ** 2)) < tol:
                labels[i] = k
                break
        else:
            reps.append(i)
            labels[i] = len(reps) - 1
    return labels


# ------------------------------------------------------------ bagged logit
def _logit():
    return LogisticRegression(C=1e6, max_iter=5000)


class BestSubsetLogit:
    """Median-impute, standardise, pick the best k covariates by the score
    criterion, fit an unpenalised logistic regression on them.
    """

    def __init__(self, k=4):
        self.k = k

    def fit(self, X, y):
        self.prep_ = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler()
        ).fit(X)
        Z = self.prep_.transform(X)
        self.vars_ = rashomon.best_score_subset(Z, y, self.k)
        self.m_ = _logit().fit(Z[:, self.vars_], y)
        return self

    def predict_proba1(self, X):
        Z = self.prep_.transform(X)
        return self.m_.predict_proba(Z[:, self.vars_])[:, 1]

    def predict(self, X):
        return (self.predict_proba1(X) >= 0.5).astype(int)


class Bagged:
    def __init__(self, base, n=50, seed=0):
        self.base, self.n, self.seed = base, n, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.models_ = []
        for _ in range(self.n):
            idx = rng.choice(len(y), size=len(y), replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            self.models_.append(self.base().fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        p = np.mean([m.predict_proba1(X) for m in self.models_], axis=0)
        return (p >= 0.5).astype(int)


def bagged_logistic(reps=100, bags=50, seed=0):
    X, y, _ = data.hepatitis()
    res = models.repeated_holdout(
        {
            "single": lambda r: BestSubsetLogit(4),
            "bagged": lambda r: Bagged(lambda: BestSubsetLogit(4), n=bags, seed=r),
        },
        X,
        y,
        reps=reps,
        seed=seed,
    )
    return pd.DataFrame(res)


def selection_frequency(boots=500, k=4, seed=0):
    """Diaconis-Efron style: which covariates the best-subset rule picks across
    bootstrap resamples of the whole data set.
    """
    X, y, names = data.hepatitis()
    rng = np.random.default_rng(seed)
    picks = []
    for b in range(boots):
        idx = rng.choice(len(y), size=len(y), replace=True)
        picks.append(BestSubsetLogit(k).fit(X[idx], y[idx]).vars_.tolist())
    full = BestSubsetLogit(k).fit(X, y).vars_.tolist()
    return picks, full, names
