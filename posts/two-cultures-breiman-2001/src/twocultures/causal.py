"""Effect estimators for the Lalonde example in Part 5.

All three take an outcome y, a binary treatment d and covariates X:

- `s_learner`: fit one predictive model of y on (d, X) and read the effect off
  it as mean[f(1, X) - f(0, X)] over the treated, the partial-dependence reading
  of a black box.
- `dml_plr`: double/debiased machine learning for the partially linear model
  (Chernozhukov et al. 2018): cross-fitted residuals of y and d on X, then the
  slope of one residual on the other.
- `aipw_att`: the cross-fitted doubly robust estimate of the effect on the treated.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold


def s_learner(model, y, d, X):
    """Effect on the treated read off a single predictive model."""
    Z = np.c_[d, X]
    m = clone(model).fit(Z, y)
    Xt = X[d == 1]
    z1, z0 = np.c_[np.ones(len(Xt)), Xt], np.c_[np.zeros(len(Xt)), Xt]
    return float(np.mean(m.predict(z1) - m.predict(z0)))


def _crossfit(model, target, X, folds, seed, classify):
    out = np.zeros(len(target))
    cv = (StratifiedKFold if classify else KFold)(
        folds, shuffle=True, random_state=seed
    )
    for tr, te in cv.split(X, target if classify else None):
        m = clone(model).fit(X[tr], target[tr])
        out[te] = m.predict_proba(X[te])[:, 1] if classify else m.predict(X[te])
    return out


def dml_plr(reg, clf, y, d, X, folds=5, seeds=(0, 1, 2, 3, 4)):
    """Cross-fitted partially linear estimate, median over sample splits."""
    thetas, ses = [], []
    for s in seeds:
        ry = y - _crossfit(reg, y, X, folds, s, False)
        rd = d - _crossfit(clf, d, X, folds, s, True)
        theta = np.sum(rd * ry) / np.sum(rd * rd)
        psi = (ry - theta * rd) * rd
        se = np.sqrt(np.mean(psi**2) / np.mean(rd * rd) ** 2 / len(y))
        thetas.append(theta)
        ses.append(se)
    i = int(np.argsort(thetas)[len(thetas) // 2])
    return float(thetas[i]), float(ses[i])


def aipw_att(reg, clf, y, d, X, folds=5, seeds=(0, 1, 2, 3, 4), clip=0.01):
    """Cross-fitted doubly robust effect on the treated, median over splits."""
    ests, ses = [], []
    p = d.mean()
    for s in seeds:
        e = np.clip(_crossfit(clf, d, X, folds, s, True), clip, 1 - clip)
        m0 = np.zeros(len(y))
        cv = StratifiedKFold(folds, shuffle=True, random_state=s)
        for tr, te in cv.split(X, d):
            ctl = tr[d[tr] == 0]
            m0[te] = clone(reg).fit(X[ctl], y[ctl]).predict(X[te])
        psi = (d * (y - m0) - (1 - d) * e / (1 - e) * (y - m0)) / p
        att = np.mean(psi)
        se = np.std(psi - d * att / p, ddof=1) / np.sqrt(len(y))
        ests.append(att)
        ses.append(se)
    i = int(np.argsort(ests)[len(ests) // 2])
    return float(ests[i]), float(ses[i])


LALONDE_COVARIATES = [
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]


def lalonde(seeds=(0, 1, 2, 3, 4)):
    """Every estimate Part 5 reports, from the NSW experiment and the PSID
    comparison group (Dehejia and Wahba 1999 version).
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    from . import data

    nsw, psid = data._read("lalonde_nsw"), data._read("lalonde_psid")
    t, c = nsw[nsw["treat"] == 1], nsw[nsw["treat"] == 0]
    exp = float(t["re78"].mean() - c["re78"].mean())
    exp_se = float(np.sqrt(t["re78"].var() / len(t) + c["re78"].var() / len(c)))
    obs = pd.concat([t, psid], ignore_index=True)
    X = obs[LALONDE_COVARIATES].to_numpy(float)
    d, y = obs["treat"].to_numpy(int), obs["re78"].to_numpy(float)
    out = {
        "experimental": exp,
        "experimental_se": exp_se,
        "n_treated": int(len(t)),
        "n_exp_controls": int(len(c)),
        "n_psid": int(len(psid)),
        "naive": float(y[d == 1].mean() - y[d == 0].mean()),
        "ols": float(LinearRegression().fit(np.c_[d, X], y).coef_[0]),
    }
    r2, sl, plr, plr_se, att, att_se = [], [], [], [], [], []
    for s in seeds:
        rf = RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, random_state=s, n_jobs=-1
        )
        r2.append(float(cross_val_score(rf, np.c_[d, X], y, cv=5, scoring="r2").mean()))
        sl.append(s_learner(rf, y, d, X))
        reg = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=s, n_jobs=-1
        )
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=s, n_jobs=-1
        )
        a, b = dml_plr(reg, clf, y, d, X, seeds=(s,))
        plr.append(a)
        plr_se.append(b)
        a, b = aipw_att(reg, clf, y, d, X, seeds=(s,))
        att.append(a)
        att_se.append(b)
        if s == seeds[0]:
            e = _crossfit(clf, d, X, 5, s, True)
            out["psid_share_low_propensity"] = float(np.mean(e[d == 0] < 0.01))
            out["treated_share_low_propensity"] = float(np.mean(e[d == 1] < 0.01))
    out.update(
        {
            "rf_r2": r2,
            "s_learner": sl,
            "dml_plr": plr,
            "dml_plr_se": plr_se,
            "aipw_att": att,
            "aipw_att_se": att_se,
        }
    )
    return out
