"""Part 3's experiments beyond Table 2.

- `credit_models`: Hoadley's argument on the Taiwan credit-card default data: a
  binned logistic "scorecard" against a pruned tree, a random forest and plain
  logistic regression.
- `svm_degrees`: Breiman's sec. 10.3 claim on the USPS ZIP-code digits: add
  polynomial features (a polynomial kernel) until the classes separate.
- `random_features`: test error against the number of random ReLU features for
  minimum-norm least squares, the double-descent curve of Belkin et al. (2019).
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from . import data, models


# ------------------------------------------------------------------- credit
def credit_frame():
    df = data._read("default_credit")
    y = df["default payment next month"].to_numpy(int)
    raw = df.drop(columns=["ID", "default payment next month"])
    return raw, y


def characteristics(raw: pd.DataFrame) -> pd.DataFrame:
    """Fair, Isaac-style derived characteristics from the raw monthly columns."""
    out = pd.DataFrame(index=raw.index)
    lim = raw["LIMIT_BAL"].clip(lower=1)
    pay = [f"PAY_{k}" for k in (0, 2, 3, 4, 5, 6)]
    bill = [f"BILL_AMT{k}" for k in range(1, 7)]
    amt = [f"PAY_AMT{k}" for k in range(1, 7)]
    out["limit"] = raw["LIMIT_BAL"]
    out["age"] = raw["AGE"]
    out["status_last"] = raw["PAY_0"]
    out["status_worst6"] = raw[pay].max(axis=1)
    out["months_late6"] = (raw[pay] > 0).sum(axis=1)
    out["util_last"] = raw["BILL_AMT1"] / lim
    out["util_mean6"] = raw[bill].mean(axis=1) / lim
    out["paid_ratio_last"] = raw["PAY_AMT1"] / raw["BILL_AMT1"].clip(lower=1)
    out["paid_ratio6"] = raw[amt].sum(axis=1) / raw[bill].sum(axis=1).clip(lower=1)
    out["bill_trend"] = (raw["BILL_AMT1"] - raw["BILL_AMT6"]) / lim
    return out


CATEGORICAL = ["status_last", "status_worst6", "months_late6"]


def scorecard(n_bins=8):
    """Logistic regression on binned characteristics: a scorecard."""
    cont = [
        "limit",
        "age",
        "util_last",
        "util_mean6",
        "paid_ratio_last",
        "paid_ratio6",
        "bill_trend",
    ]
    pre = ColumnTransformer(
        [
            (
                "bins",
                KBinsDiscretizer(
                    n_bins=n_bins,
                    encode="onehot",
                    strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                ),
                cont,
            ),
            (
                "cats",
                OneHotEncoder(handle_unknown="ignore", min_frequency=50),
                CATEGORICAL,
            ),
        ]
    )
    return make_pipeline(pre, LogisticRegression(C=1.0, max_iter=5000))


def _credit_split(y, seed):
    return train_test_split(
        np.arange(len(y)), test_size=1 / 3, random_state=seed, stratify=y
    )


def credit_scorecard(seed=0):
    """The scorecard alone, fitted on `credit_models(seed)`'s training split."""
    raw, y = credit_frame()
    ch = characteristics(raw)
    tr, _ = _credit_split(y, seed)
    return scorecard().fit(ch.iloc[tr], y[tr])


def credit_models(seed=0):
    raw, y = credit_frame()
    ch = characteristics(raw)
    tr, te = _credit_split(y, seed)
    out = []
    specs = [
        (
            "Logistic regression, 23 raw inputs",
            raw,
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        ),
        ("Pruned tree (CART), raw inputs", raw, models.PrunedTree(random_state=seed)),
        (
            "Random forest, raw inputs",
            raw,
            RandomForestClassifier(
                n_estimators=500, min_samples_leaf=5, random_state=seed, n_jobs=-1
            ),
        ),
        ("Scorecard: logistic on binned characteristics", ch, scorecard()),
    ]
    fitted = {}
    for name, X, m in specs:
        keep_frame = name.startswith("Scorecard")
        Xtr = X.iloc[tr] if keep_frame else X.to_numpy(float)[tr]
        Xte = X.iloc[te] if keep_frame else X.to_numpy(float)[te]
        m.fit(Xtr, y[tr])
        p = m.predict_proba(Xte)[:, 1]
        out.append(
            {
                "model": name,
                "auc": float(roc_auc_score(y[te], p)),
                "error": float(np.mean((p >= 0.5) != y[te])),
                "n_params": _n_params(m),
            }
        )
        fitted[name] = m
    base = float(min(y[te].mean(), 1 - y[te].mean()))
    return pd.DataFrame(out), base, fitted, ch.iloc[tr], y[tr]


def _n_params(m):
    if hasattr(m, "steps"):
        last = m.steps[-1][1]
        if hasattr(last, "coef_"):
            return int(last.coef_.size + 1)
    if hasattr(m, "tree_"):
        return int(m.tree_.get_n_leaves())
    if hasattr(m, "estimators_"):
        return int(sum(e.tree_.node_count for e in m.estimators_))
    return -1


# ---------------------------------------------------------------------- svm
def svm_degrees(degrees=(1, 2, 3, 4, 5), C=1e6):
    """Hard-margin (very large C) polynomial-kernel SVMs on USPS digits."""
    Xtr, ytr, Xte, yte = data.usps()

    def one(d):
        # gamma = 1/p keeps the kernel's scale comparable across degrees.
        m = SVC(
            kernel="poly",
            degree=d,
            gamma=1 / Xtr.shape[1],
            coef0=1.0,
            C=C,
            decision_function_shape="ovo",
            cache_size=2000,
        )
        m.fit(Xtr, ytr)
        return {
            "degree": d,
            "train_error": float(np.mean(m.predict(Xtr) != ytr)),
            "test_error": float(np.mean(m.predict(Xte) != yte)),
            "support_vectors": int(m.n_support_.sum()),
            "n_features": int(_n_monomials(Xtr.shape[1], d)),
        }

    return pd.DataFrame(Parallel(n_jobs=len(degrees))(delayed(one)(d) for d in degrees))


def _n_monomials(p, d):
    """Monomials of degree at most d in p variables, constant excluded."""
    from math import comb

    return comb(p + d, d) - 1


# -------------------------------------------------------- random features
def random_features(n_train=1000, widths=None, seeds=(0, 1, 2, 3, 4), ridge=0.0):
    """Minimum-norm (or ridge) least squares on random ReLU features of USPS."""
    Xtr, ytr, Xte, yte = data.usps()
    if widths is None:
        widths = np.unique(np.round(np.logspace(1, np.log10(20000), 40)).astype(int))
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = rng.choice(len(ytr), size=n_train, replace=False)
        A, ya = Xtr[idx], ytr[idx]
        Ya = np.eye(10)[ya]
        W = rng.normal(size=(A.shape[1], max(widths))) / np.sqrt(A.shape[1])
        for p in widths:
            Fa = np.maximum(A @ W[:, :p], 0)
            Ft = np.maximum(Xte @ W[:, :p], 0)
            if ridge > 0:
                B = (
                    np.linalg.solve(Fa.T @ Fa + ridge * np.eye(p), Fa.T @ Ya)
                    if p <= n_train
                    else Fa.T @ np.linalg.solve(Fa @ Fa.T + ridge * np.eye(n_train), Ya)
                )
            else:
                B = np.linalg.pinv(Fa, rcond=1e-10) @ Ya
            tr_err = float(np.mean((Fa @ B).argmax(1) != ya))
            te_err = float(np.mean((Ft @ B).argmax(1) != yte))
            rows.append(
                {
                    "seed": s,
                    "width": int(p),
                    "ridge": ridge,
                    "train_error": tr_err,
                    "test_error": te_err,
                    "norm": float(np.linalg.norm(B)),
                }
            )
    return pd.DataFrame(rows)
