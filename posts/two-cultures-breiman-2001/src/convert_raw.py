"""Convert the downloaded raw files into data/raw/*.csv.gz.

Usage: convert_raw.py DL_DIR OUT_DIR SUMMARY_JSON. It is the script that wrote
every file in data/raw/ (see data/README.md); it reads only the downloads.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pyreadr

DL, OUT, SUMMARY = sys.argv[1], sys.argv[2], sys.argv[3]
summary = {}


def p(*parts):
    return os.path.join(DL, *parts)


def write(name, df, target=None):
    df = df.copy()
    for c in df.columns:
        s = df[c]
        if (
            s.dtype.kind == "f"
            and s.notna().any()
            and (s.dropna() == s.dropna().round()).all()
        ):
            df[c] = s.astype("Int64")
    path = os.path.join(OUT, name + ".csv.gz")
    df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0}, na_rep="")
    info = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_col": {c: int(v) for c, v in df.isna().sum().items() if v},
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "bytes": os.path.getsize(path),
    }
    if target is not None:
        vc = df[target].value_counts(dropna=False).sort_index()
        info["target"] = target
        info["class_counts"] = {str(k): int(v) for k, v in vc.items()}
    summary[name] = info


def uci_csv(path, names, na="?", sep=",", **kw):
    return pd.read_csv(
        path,
        header=None,
        names=names,
        na_values=[na],
        keep_default_na=False,
        sep=sep,
        **kw,
    )


# 1 hepatitis
hep_names = [
    "Class",
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
write("hepatitis", uci_csv(p("hepatitis", "hepatitis.data"), hep_names), "Class")

# 2 bupa
bupa = uci_csv(
    p("bupa", "bupa.data"),
    ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"],
)
write("bupa", bupa, "selector")

# 3 ozone
oz = pyreadr.read_r(p("mlbench_2.1-11", "x", "mlbench", "data", "Ozone.rda"))["Ozone"]
write("ozone_mlbench", oz, None)
summary["ozone_mlbench"]["dtypes"] = {c: str(t) for c, t in oz.dtypes.items()}
ozf = pyreadr.read_r(p("faraway_1.0.9", "x", "faraway", "data", "ozone.rda"))["ozone"]
write("ozone_faraway", ozf, None)

# 4 cancer
bcw_names = [
    "Sample_code_number",
    "Clump_Thickness",
    "Uniformity_of_Cell_Size",
    "Uniformity_of_Cell_Shape",
    "Marginal_Adhesion",
    "Single_Epithelial_Cell_Size",
    "Bare_Nuclei",
    "Bland_Chromatin",
    "Normal_Nucleoli",
    "Mitoses",
    "Class",
]
write(
    "breast_cancer_wisconsin",
    uci_csv(p("bcw", "breast-cancer-wisconsin.data"), bcw_names),
    "Class",
)

# ionosphere
write(
    "ionosphere",
    uci_csv(p("iono", "ionosphere.data"), [f"a{i}" for i in range(1, 35)] + ["class"]),
    "class",
)

# diabetes (Pima, mlbench 2.1-9)
pima = pyreadr.read_r(
    p("mlbench_2.1-9", "x", "mlbench", "data", "PimaIndiansDiabetes.rda")
)["PimaIndiansDiabetes"]
write("pima_diabetes", pima, "diabetes")
summary["pima_diabetes"]["zeros_by_col"] = {
    c: int((pima[c] == 0).sum()) for c in pima.columns if c != "diabetes"
}

# glass
glass = uci_csv(
    p("glass", "glass.data"),
    ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"],
)
write("glass", glass, "Type")

# soybean (UCI large, original train/test)
soy_names = [
    "class",
    "date",
    "plant-stand",
    "precip",
    "temp",
    "hail",
    "crop-hist",
    "area-damaged",
    "severity",
    "seed-tmt",
    "germination",
    "plant-growth",
    "leaves",
    "leafspots-halo",
    "leafspots-marg",
    "leafspot-size",
    "leaf-shread",
    "leaf-malf",
    "leaf-mild",
    "stem",
    "lodging",
    "stem-cankers",
    "canker-lesion",
    "fruiting-bodies",
    "external-decay",
    "mycelium",
    "int-discolor",
    "sclerotia",
    "fruit-pods",
    "fruit-spots",
    "seed",
    "mold-growth",
    "seed-discolor",
    "seed-size",
    "shriveling",
    "roots",
]
soy_tr = uci_csv(p("soybean", "soybean-large.data"), soy_names)
soy_te = uci_csv(p("soybean", "soybean-large.test"), soy_names)
write("soybean_train", soy_tr, "class")
write("soybean_test", soy_te, "class")
# cross-check vs mlbench Soybean
ms = pyreadr.read_r(p("mlbench_2.1-11", "x", "mlbench", "data", "Soybean.rda"))[
    "Soybean"
]
both = pd.concat([soy_tr, soy_te], ignore_index=True)
summary["soybean_check"] = {
    "mlbench_shape": list(ms.shape),
    "mlbench_class_counts": {
        str(k): int(v) for k, v in ms["Class"].value_counts().sort_index().items()
    },
    "uci_combined_class_counts": {
        str(k): int(v) for k, v in both["class"].value_counts().sort_index().items()
    },
    "mlbench_missing_cells": int(ms.isna().sum().sum()),
    "uci_missing_cells": int(both.isna().sum().sum()),
}
# row-by-row compare of codes (mlbench factor levels are the integer codes as strings)
try:
    a = (
        both.drop(columns="class")
        .astype("Float64")
        .to_numpy(dtype=float, na_value=np.nan)
    )
    b = (
        ms.drop(columns="Class")
        .apply(
            lambda s: pd.to_numeric(
                s.astype(str).replace({"nan": None}), errors="coerce"
            )
        )
        .to_numpy(dtype=float)
    )
    same = np.all((a == b) | (np.isnan(a) & np.isnan(b)), axis=1)
    summary["soybean_check"]["rows_identical_in_order"] = int(same.sum())
    summary["soybean_check"]["class_identical_in_order"] = int(
        (both["class"].to_numpy() == ms["Class"].astype(str).to_numpy()).sum()
    )
except Exception as e:  # noqa: BLE001
    summary["soybean_check"]["compare_error"] = repr(e)

# letters
letter_names = [
    "lettr",
    "x-box",
    "y-box",
    "width",
    "high",
    "onpix",
    "x-bar",
    "y-bar",
    "x2bar",
    "y2bar",
    "xybar",
    "x2ybr",
    "xy2br",
    "x-ege",
    "xegvy",
    "y-ege",
    "yegvx",
]
write("letters", uci_csv(p("letter", "letter-recognition.data"), letter_names), "lettr")

# satellite
sat_names = [f"A{i}" for i in range(1, 37)] + ["class"]
write("satimage_train", uci_csv(p("sat", "sat.trn"), sat_names, sep=" "), "class")
write("satimage_test", uci_csv(p("sat", "sat.tst"), sat_names, sep=" "), "class")

# shuttle
sh_names = [f"A{i}" for i in range(1, 10)] + ["class"]
write("shuttle_train", uci_csv(p("shuttle", "shuttle.trn"), sh_names, sep=" "), "class")
write("shuttle_test", uci_csv(p("shuttle", "shuttle.tst"), sh_names, sep=" "), "class")

# DNA: mlbench, split by row position, verified against libsvm
dna = pyreadr.read_r(p("mlbench_2.1-11", "x", "mlbench", "data", "DNA.rda"))["DNA"]
summary["dna_mlbench_shape"] = list(dna.shape)
X = dna.drop(columns="Class").apply(lambda s: pd.to_numeric(s.astype(str))).to_numpy()


def read_libsvm(path, d=180):
    ys, rows = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            ys.append(int(float(parts[0])))
            r = np.zeros(d)
            for kv in parts[1:]:
                k, v = kv.split(":")
                r[int(k) - 1] = float(v)
            rows.append(r)
    return np.array(ys), np.vstack(rows)


ytr, Xtr = read_libsvm(p("libsvm", "dna.scale"))
yte, Xte = read_libsvm(p("libsvm", "dna.scale.t"))
Xl = np.vstack([Xtr, Xte])
yl = np.concatenate([ytr, yte])
rowmatch = np.all(X == Xl, axis=1)
lab = dna["Class"].astype(str).to_numpy()
pairs = pd.crosstab(lab, yl)
summary["dna_check"] = {
    "libsvm_train": int(len(ytr)),
    "libsvm_test": int(len(yte)),
    "rows_identical_in_order_total": int(rowmatch.sum()),
    "rows_identical_first2000": int(rowmatch[:2000].sum()),
    "rows_identical_last1186": int(rowmatch[2000:].sum()),
    "label_crosstab_mlbench_vs_libsvm": {
        str(i): {str(j): int(pairs.loc[i, j]) for j in pairs.columns}
        for i in pairs.index
    },
}
write("dna_train", dna.iloc[:2000].reset_index(drop=True), "Class")
write("dna_test", dna.iloc[2000:].reset_index(drop=True), "Class")

# digits (USPS)
zip_names = ["digit"] + [f"p{i}" for i in range(1, 257)]
for split, fn in [("train", "zip.train.gz"), ("test", "zip.test.gz")]:
    z = pd.read_csv(p("zip", fn), header=None, sep=r"\s+", engine="python")
    z = z.dropna(axis=1, how="all")
    z.columns = zip_names
    z["digit"] = z["digit"].astype(int)
    write(f"usps_zip_{split}", z, "digit")

# lymphoma (Breiman tryw)
rows = []
with open(p("breiman_wb2", "tryw")) as fh:
    for line in fh:
        parts = line.split()
        if parts:
            rows.append([float(x) for x in parts])
assert all(len(r) == 4683 for r in rows), {len(r) for r in rows}
lym = pd.DataFrame(rows, columns=[f"g{i}" for i in range(1, 4683)] + ["class"])
lym["class"] = lym["class"].astype(int)
write("lymphoma_breiman", lym, "class")

# credit default
cr = pd.read_excel(p("credit", "x", "default of credit card clients.xls"), header=1)
summary["credit_columns"] = list(cr.columns)
write("default_credit", cr, "default payment next month")

# lalonde
nsw = pd.read_stata(p("lalonde", "nsw_dw.dta"))
psid = pd.read_stata(p("lalonde", "psid_controls.dta"))
for _d in (nsw, psid):
    for _c in ["treat", "age", "education", "black", "hispanic", "married", "nodegree"]:
        assert (_d[_c] == _d[_c].round()).all()
        _d[_c] = _d[_c].astype(int)
summary["lalonde_columns"] = [list(nsw.columns), list(psid.columns)]
write("lalonde_nsw", nsw, "treat")
write("lalonde_psid", psid, "treat")
m1 = nsw.loc[nsw.treat == 1, "re78"].mean()
m0 = nsw.loc[nsw.treat == 0, "re78"].mean()
summary["lalonde_effect"] = {
    "treated_mean_re78": float(m1),
    "control_mean_re78": float(m0),
    "diff": float(m1 - m0),
}
# cross-check .txt versions
tt = pd.read_csv(p("lalonde", "nswre74_treated.txt"), header=None, sep=r"\s+")
tc = pd.read_csv(p("lalonde", "nswre74_control.txt"), header=None, sep=r"\s+")
tp = pd.read_csv(p("lalonde", "psid_controls.txt"), header=None, sep=r"\s+")
summary["lalonde_txt_shapes"] = [list(tt.shape), list(tc.shape), list(tp.shape)]
summary["lalonde_txt_diff"] = float(tt.iloc[:, -1].mean() - tc.iloc[:, -1].mean())

with open(SUMMARY, "w") as fh:
    json.dump(summary, fh, indent=1)
print("ok")
