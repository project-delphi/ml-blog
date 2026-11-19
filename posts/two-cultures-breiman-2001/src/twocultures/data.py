"""Loaders for the series' data sets.

Every file lives in `posts/two-cultures-breiman-2001/data/raw/` and is described,
with its source, access date, licence and checksum, in `data/README.md`. The
loaders do only mechanical work: pick the target column, drop identifiers, and
keep the original train/test split where the source has one. Missing values stay
missing; each experiment imputes inside its own training data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{name}.csv.gz")


# --------------------------------------------------------------- Table 2 sets
def _xy(df: pd.DataFrame, target: str, drop=()):
    y = df[target].astype(str).to_numpy()
    X = df.drop(columns=[target, *drop]).to_numpy(dtype=float)
    return X, y


def table2_small() -> dict:
    """The five sets Breiman scored by repeated 10% holdouts: name -> (X, y)."""
    soy = pd.concat([_read("soybean_train"), _read("soybean_test")], ignore_index=True)
    return {
        "Breast cancer": _xy(
            _read("breast_cancer_wisconsin"), "Class", drop=["Sample_code_number"]
        ),
        "Ionosphere": _xy(_read("ionosphere"), "class"),
        "Diabetes": _xy(_read("pima_diabetes"), "diabetes"),
        "Glass": _xy(_read("glass"), "Type", drop=["Id"]),
        "Soybean": _xy(soy, "class"),
    }


def table2_large() -> dict:
    """The five sets with a separate test set: name -> (Xtr, ytr, Xte, yte)."""
    out = {}
    let = _read("letters")
    Xl, yl = _xy(let, "lettr")
    # Breiman's Table 1 uses 15,000 training and 5,000 test cases; the UCI file
    # has no designated split, so the first 15,000 rows train.
    out["Letters"] = (Xl[:15000], yl[:15000], Xl[15000:], yl[15000:])
    for name, stem, target in [
        ("Satellite", "satimage", "class"),
        ("Shuttle", "shuttle", "class"),
        ("DNA", "dna", "Class"),
        ("Digit", "usps_zip", "digit"),
    ]:
        Xtr, ytr = _xy(_read(f"{stem}_train"), target)
        Xte, yte = _xy(_read(f"{stem}_test"), target)
        out[name] = (Xtr, ytr, Xte, yte)
    return out


def usps():
    """USPS ZIP-code digits: (Xtr, ytr, Xte, yte), pixels in [-1, 1]."""
    tr, te = _read("usps_zip_train"), _read("usps_zip_test")
    return (
        tr.drop(columns="digit").to_numpy(float),
        tr["digit"].to_numpy(int),
        te.drop(columns="digit").to_numpy(float),
        te["digit"].to_numpy(int),
    )


# -------------------------------------------------------------- medical sets
# UCI hepatitis.names counts the class as attribute 1; Breiman counts covariates
# only, so his variable k is column k of this list (Efron's comment, p. 219:
# Breiman's 12 or 17 are Efron and Gong's 13 or 18).
HEPATITIS_COVARIATES = [
    "AGE",
    "SEX",
    "STEROID",
    "ANTIVIRALS",
    "FATIGUE",
    "MALAISE",
    "ANOREXIA",
    "LIVER_BIG",
    "LIVER_FIRM",
    "SPLEEN_PALPABLE",
    "SPIDERS",
    "ASCITES",
    "VARICES",
    "BILIRUBIN",
    "ALK_PHOSPHATE",
    "SGOT",
    "ALBUMIN",
    "PROTIME",
    "HISTOLOGY",
]


def hepatitis():
    """Hepatitis: X (155 x 19, NaN where missing), y (1 = died), covariate names.

    The class is coded 1 = DIE, 2 = LIVE (32 and 123 patients, as the
    documentation says). hepatitis.names lists the binary findings as "no, yes"
    without numeric codes; the data say 1 = present, 2 = absent (70% of the 20
    patients coded 1 for ascites died, against 13% of the 130 coded 2). SEX is
    1 = male, 2 = female by the same listing order. All codes are kept as given.
    """
    df = _read("hepatitis")
    y = (df["Class"] == 1).astype(int).to_numpy()
    X = df[HEPATITIS_COVARIATES].to_numpy(float)
    return X, y, list(HEPATITIS_COVARIATES)


def bupa():
    """Bupa liver disorders: the six covariates, `drinks`, and `selector`."""
    return _read("bupa")


def lymphoma():
    """Lymphoma microarray: X (81 x 4,682), y (class label)."""
    df = _read("lymphoma_breiman")
    target = [c for c in df.columns if not c.startswith("g")]
    (tcol,) = target
    return df.drop(columns=tcol).to_numpy(float), df[tcol].astype(str).to_numpy()


# ------------------------------------------------------------------- ozone
OZONE_MET = ["vh", "wind", "humidity", "temp", "ibh", "dpg", "ibt", "vis"]


def ozone_faraway() -> pd.DataFrame:
    """faraway::ozone: 330 complete days of 1976 LA ozone and meteorology."""
    return _read("ozone_faraway")


def ozone_mlbench() -> pd.DataFrame:
    """mlbench::Ozone with descriptive column names (366 days of 1976, NAs kept)."""
    df = _read("ozone_mlbench")
    names = {
        "V1": "month",
        "V2": "day",
        "V3": "weekday",
        "V4": "ozone",
        "V5": "vh",
        "V6": "wind",
        "V7": "humidity",
        "V8": "temp_sandburg",
        "V9": "temp_elmonte",
        "V10": "ibh",
        "V11": "dpg",
        "V12": "ibt",
        "V13": "vis",
    }
    df = df.rename(columns=names)
    df["date"] = pd.to_datetime(
        dict(year=1976, month=df["month"], day=df["day"]), errors="coerce"
    )
    return df


def next_day(df: pd.DataFrame, cols, target, lags=(0,), day="doy"):
    """Pairs (predictors on days t - lag, target on day t + 1) for consecutive days.

    A row is kept only when every needed day is present, so gaps in the record
    never pair a predictor with the wrong day's ozone.
    """
    d = df.set_index(day).sort_index()
    rows, idx = [], []
    for t in d.index:
        need = [t - lag for lag in lags] + [t + 1]
        if not all(k in d.index for k in need):
            continue
        feats = np.concatenate([d.loc[t - lag, cols].to_numpy(float) for lag in lags])
        if np.isnan(feats).any() or pd.isna(d.loc[t + 1, target]):
            continue
        rows.append(np.append(feats, d.loc[t + 1, target]))
        idx.append(t)
    names = [f"{c}_lag{lag}" for lag in lags for c in cols] + ["target"]
    return pd.DataFrame(rows, columns=names, index=pd.Index(idx, name=day))
