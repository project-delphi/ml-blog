"""Export the cached experiment results the series' widgets draw, as widget-data/*.js.

Run from the repository root with the series venv:

    .venv-umap/bin/python posts/when-umap-lies/src/export_widget_data.py          # all
    .venv-umap/bin/python posts/when-umap-lies/src/export_widget_data.py elephant
    .venv-umap/bin/python posts/when-umap-lies/src/export_widget_data.py dimension
    .venv-umap/bin/python posts/when-umap-lies/src/export_widget_data.py signal
    .venv-umap/bin/python posts/when-umap-lies/src/export_widget_data.py ranking
    .venv-umap/bin/python posts/when-umap-lies/src/export_widget_data.py umap

Part 1's widgets write to ``widget-data/`` here; later parts' widgets write into
their own post folder (``../when-umap-lies-dimension/widget-data/`` for Part 2).

One function per widget. Each reads only ``cache/`` (plus the reference spaces it
must recompute a metric against) and writes ``widget-data/<name>.js`` in the format
documented in ``widgetdata.py``. Nothing here runs an embedding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POST = HERE.parent
sys.path.insert(0, str(HERE))

from widgetdata import (  # noqa: E402
    FORMAT,
    compact,
    encode_embedding,
    encode_labels,
    encode_values,
    write_widget_data,
)

CACHE = POST / "cache"
OUT = POST / "widget-data"
METRICS = ("inter_corr", "intra_corr", "knn_acc", "knn_recall", "silhouette", "max_ami")
SEEDS = range(5)
LARGE_TYPE = 100  # cells; the restricted inter-type correlation keeps types this big


def _inter_corr(ref: np.ndarray, Y: np.ndarray, labels: np.ndarray) -> float:
    from umaplab.elephant import _centroid_dists

    return float(
        np.corrcoef(_centroid_dists(ref, labels), _centroid_dists(Y, labels))[0, 1]
    )


def export_elephant() -> int:
    """Part 1: the elephant in the spirit of Picasso, scored with both papers' metrics.

    Ships five layouts per dataset: 2-D PCA (f = 0), the matched elephant (f = 1), the
    random elephant, and seed 0 of UMAP and t-SNE. The layout at any f is the straight
    blend (1 - f) * PCA + f * elephant, which the widget recomputes; the check below
    confirms the cached f-layouts are exactly that blend.

    For PBMC it adds ``inter_corr_large``: the inter-type centroid-distance correlation
    restricted to the types with at least LARGE_TYPE cells. The all-type version is
    dominated by the two tiny types (megakaryocytes, dendritic cells), whose centroids
    sit far from everything in any layout that scatters a small group at random.
    """
    import pandas as pd
    from umaplab.elephant import F_GRID
    from umaplab.generators import gaussian_clusters
    from umaplab.realdata import load_pbmc3k

    scores = pd.read_parquet(CACHE / "elephant" / "scores.parquet")
    lay = np.load(CACHE / "elephant" / "layouts.npz")
    pbmc = load_pbmc3k()
    refs = {
        "pbmc3k": (pbmc.X, pbmc.meta["label_names"], "cell type"),
        "clusters": (
            gaussian_clusters(2000, 0, n_clusters=10, dim=10, separation=4.0).X,
            [f"Cluster {k + 1}" for k in range(10)],
            "cluster",
        ),
    }
    fs = [f"{f:.1f}" for f in F_GRID]
    out = {"format": FORMAT, "f": [float(f) for f in F_GRID], "datasets": {}}
    for name, (ref, class_names, noun) in refs.items():
        labels = lay[f"{name}/labels"].astype(int)
        assert labels.max() + 1 == len(class_names), name
        pca, eleph = lay[f"{name}/elephant·f0.0"], lay[f"{name}/elephant·f1.0"]
        for f, key in zip(F_GRID, fs):  # the cached sweep is the straight blend
            blend = (1 - f) * pca.astype(float) + f * eleph.astype(float)
            assert np.abs(lay[f"{name}/elephant·f{key}"] - blend).max() < 1e-5

        df = scores[scores.dataset == name]

        def vals(emb: str, metric: str) -> float:
            v = df[(df.embedding == emb) & (df.metric == metric)].value
            assert len(v) == 1, (name, emb, metric)
            return float(v.iloc[0])

        el = df[df.embedding == "elephant"].pivot(
            index="f", columns="metric", values="value"
        )
        el = el.loc[list(F_GRID)]
        sc = {
            "elephant": {m: el[m].tolist() for m in METRICS},
            "random": {m: vals("random", m) for m in METRICS},
            "umap": {m: [vals(f"umap·s{s}", m) for s in SEEDS] for m in METRICS},
            "tsne": {m: [vals(f"tsne·s{s}", m) for s in SEEDS] for m in METRICS},
        }
        # The cached all-type score recomputed from the layouts, as a check that the
        # layouts and the reference are the ones that were scored.
        for emb, key in [
            ("random", "random_elephant"),
            ("umap", "umap·s0"),
            ("tsne", "tsne·s0"),
        ]:
            got = _inter_corr(ref, lay[f"{name}/{key}"], labels)
            assert (
                abs(got - vals(emb if emb == "random" else f"{emb}·s0", "inter_corr"))
                < 1e-3
            )
        metrics = list(METRICS)
        counts = np.bincount(labels, minlength=len(class_names))
        if name == "pbmc3k":
            keep = np.flatnonzero(counts >= LARGE_TYPE)
            mask = np.isin(labels, keep)
            r, lab = ref[mask], labels[mask]

            def large(key: str) -> float:
                return _inter_corr(r, lay[f"{name}/{key}"][mask], lab)

            sc["elephant"]["inter_corr_large"] = [large(f"elephant·f{k}") for k in fs]
            sc["random"]["inter_corr_large"] = large("random_elephant")
            for m in ("umap", "tsne"):
                sc[m]["inter_corr_large"] = [large(f"{m}·s{s}") for s in SEEDS]
            metrics.insert(1, "inter_corr_large")
            large_types = [class_names[k] for k in keep]
        else:
            large_types = None

        out["datasets"][name] = {
            "n": int(len(labels)),
            "noun": noun,
            "classes": class_names,
            "counts": counts.tolist(),
            "largeTypes": large_types,
            "largeMin": LARGE_TYPE if large_types else None,
            "labels": encode_labels(labels),
            "layouts": {
                "pca": encode_embedding(pca),
                "elephant": encode_embedding(eleph),
                "random": encode_embedding(lay[f"{name}/random_elephant"]),
                "umap": encode_embedding(lay[f"{name}/umap·s0"]),
                "tsne": encode_embedding(lay[f"{name}/tsne·s0"]),
            },
            "metrics": metrics,
            "scores": sc,
        }
    return write_widget_data(OUT / "elephant.js", "uml-elephant", out)


DIM_BASES = {"swiss_roll": "Swiss roll", "gaussian_clusters": "Gaussian clusters"}
DIM_CONFIGS = ("pca", "umap", "tsne", "umap·pca50")
DIM_SHOWN = 1000  # points per panel, the same indices in every panel


def export_dimension() -> int:
    """Part 2: pure-noise columns added to a fixed structure, D from p to 5,000.

    Per base (Swiss roll, Gaussian clusters; branching is left out) and per D: the
    seed-0 embeddings of PCA, UMAP, t-SNE and UMAP on PCA-50, each Procrustes-aligned
    (umaplab.stability.procrustes_align) to the truth panel and subsampled to the
    same DIM_SHOWN points; the k = 10 neighbour preservation against the truth of all
    five draws for each config; the same for the kNN graph of the input and of its
    PCA-50 (graph_fidelity.parquet); UMAP's fit seconds; and the noise index eta.

    The truth panel is the Swiss roll's latent (arc length s, height h), coloured by
    s, or the 2-D PCA of the clean clusters, coloured by cluster.
    """
    import json

    import pandas as pd
    from sklearn.decomposition import PCA
    from umaplab import report as R
    from umaplab.benchmark import Cell, build_dataset
    from umaplab.experiments import eta
    from umaplab.stability import procrustes_align

    df = R.load("noise_dims")
    q10 = R.pick(df, "knn_preservation", 10, "truth")
    runs = R.runs("noise_dims")
    runs = runs[runs.status == "ok"]
    emb = R.embeddings("noise_dims")
    gf = pd.read_parquet(CACHE / "noise_dims" / "graph_fidelity.parquet")
    out = {"format": FORMAT, "k": 10, "configs": list(DIM_CONFIGS), "datasets": {}}
    for gen, title in DIM_BASES.items():
        r = runs[runs.generator == gen]
        Ds = sorted(int(d) for d in r.D.unique())
        row = r[r.data_seed == 0].sort_values("D").iloc[0]
        cell = Cell(
            experiment="noise_dims",
            generator=gen,
            gen_params=json.loads(row.gen_params),
            transforms=tuple(json.loads(row.transforms)),
            n=int(row.n),
            data_seed=0,
            method="pca",
            method_params={},
            algo_seed=0,
        )
        ds = build_dataset(cell)
        p = int(ds.clean.shape[1])
        sigmas = r.noise_sigma.unique()
        assert len(sigmas) == 1, (gen, sigmas)  # one sigma0 per base
        sigma = float(sigmas[0])
        g = gf[gf.generator == gen]
        rk2 = g[(g.space == "clean_r10_sq") & (g.data_seed == 0)].value
        # The clean data does not move with D, so neither does its r10.
        assert rk2.max() - rk2.min() < 1e-9 * rk2.max()
        rk2 = float(rk2.iloc[0])

        n = ds.clean.shape[0]
        idx = np.sort(np.random.default_rng(0).choice(n, DIM_SHOWN, replace=False))
        if gen == "swiss_roll":
            truth = np.asarray(ds.latent, dtype=float)  # (s, h): the roll unrolled
            colour = {
                "kind": "values",
                "label": "arc length s",
                "values": encode_values(truth[idx, 0]),
            }
            truth_title = "Truth: the sheet unrolled (s, h)"
        else:
            truth = PCA(2, random_state=0).fit_transform(ds.clean)
            lab = np.asarray(ds.labels).astype(int)
            k = int(lab.max()) + 1
            colour = {
                "kind": "labels",
                "label": "Cluster",
                "classes": [f"Cluster {i + 1}" for i in range(k)],
                "counts": np.bincount(lab[idx], minlength=k).tolist(),
                "labels": encode_labels(lab[idx]),
            }
            truth_title = "Truth: 2-D PCA of the clean clusters"

        layouts, disparity = {}, {}
        for D in Ds:
            layouts[str(D)], disparity[str(D)] = {}, {}
            for c in DIM_CONFIGS:
                ids = r[(r.D == D) & (r.data_seed == 0) & (r.config == c)].cell_id
                assert len(ids) == 1 and ids.iloc[0] in emb, (gen, D, c)
                Y, dis = procrustes_align(truth, emb[ids.iloc[0]])
                layouts[str(D)][c] = encode_embedding(Y[idx])
                disparity[str(D)][c] = dis

        def draws(frame: pd.DataFrame, D: int) -> list[float] | None:
            v = frame[frame.D == D].sort_values("data_seed").value.tolist()
            return v or None

        qg = q10[q10.generator == gen]
        quality = {c: [draws(qg[qg.config == c], D) for D in Ds] for c in DIM_CONFIGS}
        for c in DIM_CONFIGS:
            assert all(v is not None and len(v) == 5 for v in quality[c]), (gen, c)
        g10 = g[g.k == 10]
        graph = {
            s: [draws(g10[g10.space == s], D) for D in Ds] for s in ("input", "pca50")
        }
        fits = r[r.config == "umap"].rename(columns={"seconds": "value"})
        secs = {"umap": [draws(fits, D) for D in Ds]}

        out["datasets"][gen] = {
            "title": title,
            "p": p,
            "n": int(n),
            "shown": DIM_SHOWN,
            "sigma": sigma,
            "r10sq": rk2,
            "D": Ds,
            "eta": [float(eta(sigma, D, p, rk2)) for D in Ds],
            "truth": encode_embedding(truth[idx]),
            "truthTitle": truth_title,
            "colour": colour,
            "layouts": layouts,
            "disparity": disparity,
            "q10": quality,
            "graph": graph,
            "seconds": secs,
        }
    return write_widget_data(
        POST.parent / "when-umap-lies-dimension" / "widget-data" / "dimension.js",
        "uml-dimension",
        out,
    )


SIGNAL_METHODS = ("pca", "umap", "tsne")
SIGNAL_SHOWN = 500  # points per panel: all of n = 500, a fixed 500 of n = 2,000


def export_signal() -> int:
    """Part 2: six Gaussian clusters with d internal dimensions, padded with noise.

    One draw (data seed 0) per cell of the grid d x D x sigma x n. Per (d, n): the
    truth panel (2-D PCA of the clean data), labels, and r10^2, the median squared
    distance to the 10th neighbour in the clean data, for eta = sigma^2 sqrt(8 (D - d))
    / r10^2. Per cell: PCA, UMAP and t-SNE, Procrustes-aligned to the truth and
    subsampled to the same points; k = 10 neighbour preservation and trustworthiness
    against the truth; 15-NN class accuracy in the map; and the input graph's k = 10
    preservation.
    """
    import json

    from sklearn.decomposition import PCA
    from umaplab import report as R
    from umaplab.benchmark import Cell, build_dataset
    from umaplab.metrics import local_radius
    from umaplab.stability import procrustes_align

    df = R.load("signal_widget")
    runs = R.runs("signal_widget")
    runs = runs[runs.status == "ok"].copy()
    emb = R.embeddings("signal_widget")
    runs["d"] = runs.gen_params.map(lambda g: int(json.loads(g)["dim"]))
    sigma_of = runs.noise_sigma.round(6)
    runs["sigma"] = sigma_of
    assert set(runs.data_seed) == {0}, "the widget shows one draw"
    grid = {
        "d": sorted(int(v) for v in runs.d.unique()),
        "D": sorted(int(v) for v in runs.D.unique()),
        "sigma": sorted(float(v) for v in runs.sigma.unique()),
        "n": sorted(int(v) for v in runs.n.unique()),
    }

    def metric(cell_id: str, name: str, k: float | None, ref: str) -> float:
        m = df[(df.cell_id == cell_id) & (df.metric == name) & (df.reference == ref)]
        if k is not None:
            m = m[m.k == k]
        assert len(m) == 1, (cell_id, name, k, ref, len(m))
        return float(m.value.iloc[0])

    base, cells = {}, {}
    for (d, n), rr in runs.groupby(["d", "n"]):
        row = rr.sort_values(["D", "sigma"]).iloc[0]
        ds = build_dataset(
            Cell(
                experiment="signal_widget",
                generator=row.generator,
                gen_params=json.loads(row.gen_params),
                transforms=tuple(json.loads(row.transforms)),
                n=int(row.n),
                data_seed=0,
                method="pca",
                method_params={},
                algo_seed=0,
            )
        )
        clean = np.asarray(ds.clean, dtype=float)
        truth = PCA(2, random_state=0).fit_transform(clean)
        lab = np.asarray(ds.labels).astype(int)
        shown = min(int(n), SIGNAL_SHOWN)
        idx = np.sort(np.random.default_rng(0).choice(int(n), shown, replace=False))
        rk2 = float(np.median(local_radius(clean, 10)) ** 2)
        k = int(lab.max()) + 1
        base[f"{d}/{n}"] = {
            "truth": encode_embedding(truth[idx]),
            "labels": encode_labels(lab[idx]),
            "classes": [f"Cluster {i + 1}" for i in range(k)],
            "counts": np.bincount(lab[idx], minlength=k).tolist(),
            "shown": shown,
            "r10sq": rk2,
        }
        for (D, sg), cr in rr.groupby(["D", "sigma"]):
            key = f"{d}/{int(D)}/{float(sg):g}/{n}"
            ids = {c: cr[cr.config == c].cell_id for c in SIGNAL_METHODS + ("input",)}
            assert all(len(v) == 1 for v in ids.values()), key
            ids = {c: v.iloc[0] for c, v in ids.items()}
            layouts, disparity = {}, {}
            for c in SIGNAL_METHODS:
                Y, dis = procrustes_align(truth, emb[ids[c]])
                layouts[c], disparity[c] = encode_embedding(Y[idx]), dis
            cells[key] = {
                "eta": float(sg) ** 2 * np.sqrt(8 * max(int(D) - d, 0)) / rk2,
                "layouts": layouts,
                "disparity": disparity,
                "q10": {
                    c: metric(ids[c], "knn_preservation", 10, "truth")
                    for c in SIGNAL_METHODS + ("input",)
                },
                "tw10": {
                    c: metric(ids[c], "trustworthiness", 10, "truth")
                    for c in SIGNAL_METHODS
                },
                "acc": {
                    c: metric(ids[c], "knn_accuracy", None, "emb")
                    for c in SIGNAL_METHODS
                },
            }
    assert len(cells) == np.prod([len(v) for v in grid.values()])
    out = {
        "format": FORMAT,
        "grid": grid,
        "methods": list(SIGNAL_METHODS),
        "base": base,
        "cells": cells,
    }
    return write_widget_data(
        POST.parent / "when-umap-lies-dimension" / "widget-data" / "signal.js",
        "uml-signal",
        out,
    )


RANK_METHODS = (
    "pca",
    "umap",
    "tsne",
    "pacmap",
    "trimap",
    "phate",
    "isomap",
    "spectral",
    "diffmap",
    "umap·densmap",
)
# (key, title, experiment, row filter on the loaded frame, note)
RANK_CASES = (
    (
        "swiss",
        "Swiss roll",
        "isometry",
        lambda f: (f.generator == "swiss_roll") & (f.D == 5),
        "isometry D = 5: a rotation of the 3-D roll, every distance unchanged (no D = 3 run)",
    ),
    (
        "clusters",
        "Gaussian clusters",
        "isometry",
        lambda f: (f.generator == "gaussian_clusters") & (f.D == 10),
        "isometry D = 10, the clean data",
    ),
    (
        "tree",
        "Tree (noisy, two levels)",
        "topology",
        lambda f: (
            (f.generator == "branching") & (f.gp.map(lambda g: g.get("levels")) == 2)
        ),
        "",
    ),
    (
        "circle",
        "Circle",
        "topology",
        lambda f: (f.generator == "circle") & (f.D == 2),
        "",
    ),
    ("sphere", "Sphere", "topology", lambda f: f.generator == "sphere", ""),
    (
        "density",
        "Variable density (30×)",
        "density",
        lambda f: f.gp.map(lambda g: g.get("density_ratio")) == 30.0,
        "",
    ),
    (
        "overlap",
        "Two overlapping Gaussians (δ = 3)",
        "overlap",
        lambda f: f.gp.map(lambda g: g.get("delta")) == 3.0,
        "",
    ),
    (
        "pbmc3k",
        "PBMC 3k",
        "real",
        lambda f: f.generator == "pbmc3k",
        "one dataset, five method seeds",
    ),
    (
        "mnist",
        "MNIST",
        "real",
        lambda f: f.generator == "mnist_784",
        "three 5,000-image subsamples",
    ),
    (
        "fmnist",
        "Fashion-MNIST",
        "real",
        lambda f: f.generator == "Fashion-MNIST",
        "three 5,000-image subsamples",
    ),
)
# (key, label, [(metric, k, reference), ...] tried in order: the first present is used)
RANK_METRICS = (
    ("neighbours", "Neighbours", [("knn_preservation", 10, "truth")]),
    ("trust", "False neighbours (trustworthiness)", [("trustworthiness", 10, "truth")]),
    ("continuity", "Torn neighbourhoods (continuity)", [("continuity", 10, "truth")]),
    (
        "distances",
        "Large distances",
        [("geodesic_spearman", None, "truth"), ("dist_spearman", None, "truth")],
    ),
    (
        "density",
        "Density",
        [("true_density_spearman", 15, "truth"), ("density_spearman", 15, "truth")],
    ),
    ("classes", "Classes", [("knn_accuracy", None, "emb")]),
    ("gaps", "Gap faithfulness", []),  # case 7 only, from cache/overlap/gaps.parquet
)


def export_ranking() -> int:
    """Part 4: write the ranking widget's data (see ``ranking_data``)."""
    return write_widget_data(
        POST.parent
        / "when-umap-lies-gaps-density-holes"
        / "widget-data"
        / "ranking.js",
        "uml-ranking",
        ranking_data(),
    )


def ranking_wins(data: dict) -> dict:
    """Per (case, metric): the methods whose median is within ``tie`` of the best."""
    wins = {}
    for case in data["cases"]:
        for mkey, sc in case["scores"].items():
            # compact() turns NaN into None, so drop both.
            vals = {
                m: s["v"]
                for m, s in sc.items()
                if s["v"] is not None and s["v"] == s["v"]
            }
            if not vals:
                continue
            best = max(vals.values())
            wins[(case["key"], mkey)] = sorted(
                # Same test as the widget's model.js best(): tie and 1e-12 slack.
                m
                for m, v in vals.items()
                if v >= best - data["tie"] - 1e-12
            )
    return wins


def ranking_data() -> dict:
    """Part 4: which method is best depends on the ruler.

    For each case (a dataset), method (default settings only) and metric: the median,
    min and max over draws (data seeds for synthetic data and the MNIST subsamples,
    method seeds for PBMC 3k), and how many draws. "Large distances" uses the geodesic
    Spearman where the case has a geodesic, else the Euclidean one; "Density" uses the
    true-density Spearman where the case has a known density, else the kNN-density one.
    Gap faithfulness (overlap case only) is -|valley ratio of the map - valley ratio of
    the input| per draw, then the median: 0 is perfect.
    """
    import pandas as pd
    from umaplab import report as R

    frames = {}
    out_cases = []
    for key, title, exp, sel, note in RANK_CASES:
        if exp not in frames:
            frames[exp] = R.load(exp)
        f = frames[exp]
        f = f[sel(f)]
        assert len(f), key
        configs = set(f.config)
        methods = [m for m in RANK_METHODS if m in configs]
        if key != "density":
            methods = [m for m in methods if m != "umap·densmap"]
        not_run = [m for m in RANK_METHODS[:-1] if m not in configs]
        scores, sources = {}, {}
        for mkey, _label, tries in RANK_METRICS:
            for metric, k, ref in tries:
                rows = f[(f.metric == metric) & (f.reference == ref)]
                if k is not None:
                    rows = rows[rows.k == k]
                # A NaN score is no score: a uniform true density has no rank
                # correlation, so those cases fall back to the next metric.
                rows = rows[rows.value.notna()]
                if len(rows):
                    break
            else:
                continue
            sources[mkey] = metric
            g = rows[rows.config.isin(methods)].groupby("config").value
            s = g.agg(["median", "min", "max", "count"])
            scores[mkey] = {
                m: {
                    "v": r["median"],
                    "lo": r["min"],
                    "hi": r["max"],
                    "n": int(r["count"]),
                }
                for m, r in s.iterrows()
            }
        if key == "overlap":
            gaps = pd.read_parquet(CACHE / "overlap" / "gaps.parquet")
            v = gaps[(gaps.delta == 3.0) & (gaps.metric == "valley_ratio")]
            w = v.pivot_table(index="data_seed", columns="config", values="value")
            err = -(w.drop(columns="input").sub(w["input"], axis=0)).abs()
            scores["gaps"] = {
                m: {
                    "v": float(err[m].median()),
                    "lo": float(err[m].min()),
                    "hi": float(err[m].max()),
                    "n": int(err[m].count()),
                }
                for m in methods
                if m in err
            }
            sources["gaps"] = "valley_ratio"
        out_cases.append(
            {
                "key": key,
                "title": title,
                "note": note,
                "methods": methods,
                "notRun": not_run,
                "sources": sources,
                "scores": scores,
            }
        )
    out = {
        "format": FORMAT,
        "methods": list(RANK_METHODS),
        "metrics": [{"key": k, "label": lab} for k, lab, _ in RANK_METRICS],
        "tie": 0.01,
        "cases": out_cases,
    }
    # Round here, not only in the exporter, so the post's tally (ranking_wins on this
    # output) and the widget decide ties on the same 4-significant-digit numbers.
    return compact(out, 4)


UMAP_BASES = {"gaussian_clusters": "Gaussian clusters", "swiss_roll": "Swiss roll"}
UMAP_SHOWN = 600  # points per map: the same subsample in every panel of a dataset
UMAP_REF = (15, 0.1, 0)  # n_neighbors, min_dist, seed: every map is aligned to this one


def export_umap() -> int:
    """Part 3: UMAP's two settings on 1,000 points, three seeds per setting.

    Per dataset: the truth panel (clusters: 2-D PCA of the clean data, coloured by
    label; Swiss roll: the unrolled sheet (s, h), coloured by s), and for every
    (n_neighbors, min_dist, seed) the map, Procrustes-aligned to the map at UMAP_REF so
    a change of setting does not spin the picture, with its k = 10 neighbour
    preservation and distance Spearman, both against the truth.
    """
    import json

    from sklearn.decomposition import PCA
    from umaplab import report as R
    from umaplab.benchmark import Cell, build_dataset
    from umaplab.stability import procrustes_align

    df = R.load("umap_widget")
    runs = R.runs("umap_widget")
    runs = runs[(runs.status == "ok") & (runs.method == "umap")].copy()
    emb = R.embeddings("umap_widget")
    mp = runs.method_params.map(json.loads)
    runs["nn"] = mp.map(lambda m: int(m.get("n_neighbors", 15)))
    runs["md"] = mp.map(lambda m: float(m.get("min_dist", 0.1)))

    def metric(cell_id: str, name: str, k: float | None) -> float:
        m = df[
            (df.cell_id == cell_id) & (df.metric == name) & (df.reference == "truth")
        ]
        if k is not None:
            m = m[m.k == k]
        assert len(m) == 1, (cell_id, name, len(m))
        return float(m.value.iloc[0])

    out = {"format": FORMAT, "ref": list(UMAP_REF), "datasets": {}}
    for gen, title in UMAP_BASES.items():
        r = runs[runs.generator == gen]
        nns = sorted(int(v) for v in r.nn.unique())
        mds = sorted(float(v) for v in r.md.unique())
        seeds = sorted(int(v) for v in r.algo_seed.unique())
        row = r.iloc[0]
        ds = build_dataset(
            Cell(
                experiment="umap_widget",
                generator=gen,
                gen_params=json.loads(row.gen_params),
                transforms=tuple(json.loads(row.transforms)),
                n=int(row.n),
                data_seed=0,
                method="pca",
                method_params={},
                algo_seed=0,
            )
        )
        n = int(ds.clean.shape[0])
        idx = np.sort(np.random.default_rng(0).choice(n, UMAP_SHOWN, replace=False))
        if gen == "swiss_roll":
            truth = np.asarray(ds.latent, dtype=float)
            colour = {
                "kind": "values",
                "label": "arc length s",
                "values": encode_values(truth[idx, 0]),
            }
        else:
            truth = PCA(2, random_state=0).fit_transform(ds.clean)
            lab = np.asarray(ds.labels).astype(int)
            k = int(lab.max()) + 1
            colour = {
                "kind": "labels",
                "label": "Cluster",
                "classes": [f"Cluster {i + 1}" for i in range(k)],
                "counts": np.bincount(lab, minlength=k).tolist(),  # of all n points
                "labels": encode_labels(lab[idx]),
            }

        def cell_id(nn: int, md: float, seed: int) -> str:
            ids = r[(r.nn == nn) & (r.md == md) & (r.algo_seed == seed)].cell_id
            assert len(ids) == 1, (gen, nn, md, seed, len(ids))
            return ids.iloc[0]

        ref = emb[cell_id(*UMAP_REF)].astype(float)
        maps, q10, dsp = {}, {}, {}
        for nn in nns:
            for md in mds:
                key = f"{nn}/{md:g}"
                maps[key], q10[key], dsp[key] = [], [], []
                for s in seeds:
                    cid = cell_id(nn, md, s)
                    Y, _ = procrustes_align(ref, emb[cid])
                    maps[key].append(encode_embedding(Y[idx]))
                    q10[key].append(metric(cid, "knn_preservation", 10))
                    dsp[key].append(metric(cid, "dist_spearman", None))
        out["datasets"][gen] = {
            "title": title,
            "n": n,
            "shown": UMAP_SHOWN,
            "nn": nns,
            "md": mds,
            "seeds": seeds,
            "truth": encode_embedding(truth[idx]),
            "colour": colour,
            "maps": maps,
            "q10": q10,
            "dsp": dsp,
        }
    return write_widget_data(
        POST.parent / "when-umap-lies-tuning" / "widget-data" / "umap.js",
        "uml-umap",
        out,
    )


EXPORTS = {
    "elephant": export_elephant,
    "dimension": export_dimension,
    "signal": export_signal,
    "ranking": export_ranking,
    "umap": export_umap,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "names", nargs="*", help=f"widgets to export (default all: {list(EXPORTS)})"
    )
    args = ap.parse_args()
    for name in args.names or EXPORTS:
        size = EXPORTS[name]()
        print(f"{name}: {size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
