"""Real datasets: PBMC 3k single-cell RNA-seq, MNIST and Fashion-MNIST.

Raw downloads go to ``posts/when-umap-lies/data/`` (git-ignored). The posts never read
them: they read results cached under ``cache/`` by ``src/run_experiments.py``.

Provenance:

- PBMC 3k: 2,700 peripheral blood mononuclear cells from one healthy donor, sequenced by
  10x Genomics (Chromium v1 chemistry, 2016) as a demonstration dataset. Raw counts come
  from scverse's mirror of the 10x filtered matrices; cell-type annotations come from
  the processed copy that accompanies the scanpy clustering tutorial (Louvain clusters
  on a kNN graph of the PCA of the same cells, named from marker genes).
- MNIST / Fashion-MNIST: 70,000 28x28 greyscale images each, fetched from OpenML
  (``mnist_784``, ``Fashion-MNIST``), labels from the original datasets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .generators import Dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PBMC_RAW_URL = "https://exampledata.scverse.org/scanpy/pbmc3k_raw.h5ad"
PBMC_PROCESSED_URL = "https://exampledata.scverse.org/scanpy/pbmc3k.h5ad"

# OpenML dataset ids. The Parquet copies are read directly: scikit-learn 1.9's ARFF
# parser fails on these files ("expected string or bytes-like object, got 'int'").
OPENML_IDS = {"mnist_784": 554, "Fashion-MNIST": 40996}


def openml_parquet_url(data_id: int) -> str:
    """OpenML's Parquet mirror for a dataset id."""
    return (
        f"https://data.openml.org/datasets/{data_id // 10000:04d}/"
        f"{data_id:04d}/dataset_{data_id}.pq"
    )


def _pbmc_preprocess(adata, n_pcs: int, seed: int):
    """Seurat-tutorial preprocessing, written out so every step is visible.

    1. QC: keep cells with >= 200 detected genes, < 2500 detected genes and < 5%
       mitochondrial counts; keep genes detected in >= 3 cells.
    2. Library-size normalisation to 10,000 counts per cell, then log1p.
    3. Highly variable genes: Seurat flavour, min_mean 0.0125, max_mean 3, min_disp 0.5.
    4. Scale each gene to unit variance, clipped at 10.
    5. PCA to ``n_pcs`` components (ARPACK, seeded).
    """
    import scanpy as sc

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[
        (adata.obs.n_genes_by_counts < 2500) & (adata.obs.pct_counts_mt < 5)
    ].copy()
    counts_per_cell = np.asarray(adata.X.sum(axis=1)).ravel()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    return adata, counts_per_cell


def load_pbmc3k(n_pcs: int = 50, seed: int = 0, data_dir: Path = DATA_DIR) -> Dataset:
    """PBMC 3k: X is the ``n_pcs``-dim PCA of scaled log-normalised HVG expression.

    ``latent`` is X itself (there is no ground truth for real cells). ``labels`` are the
    tutorial's cell-type annotations, which were themselves produced by clustering a
    kNN graph of a PCA of these cells, so agreement with them is partly circular.
    ``extra["hvg"]`` holds the scaled HVG matrix, a second reference space.
    """
    import scanpy as sc

    data_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.datasetdir = data_dir
    raw = sc.read(data_dir / "pbmc3k_raw.h5ad", backup_url=PBMC_RAW_URL)
    raw.var_names_make_unique()
    processed = sc.read(
        data_dir / "pbmc3k_processed.h5ad", backup_url=PBMC_PROCESSED_URL
    )
    adata, _ = _pbmc_preprocess(raw.copy(), n_pcs=n_pcs, seed=seed)
    common = adata.obs_names.intersection(processed.obs_names)
    adata = adata[common].copy()
    cell_type = processed.obs.loc[common, "louvain"].astype(str).to_numpy()
    names = np.array(sorted(set(cell_type)))
    labels = np.searchsorted(names, cell_type)
    X = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    hvg = np.asarray(adata.X, dtype=np.float64)
    meta = {
        "generator": "pbmc3k",
        "n": X.shape[0],
        "seed": seed,
        "n_pcs": n_pcs,
        "n_hvg": int(hvg.shape[1]),
        "label_names": names.tolist(),
        "intrinsic_dim": None,
        "ambient_dim": X.shape[1],
        "betti": None,
        "transforms": [],
        "source": PBMC_RAW_URL,
        "labels_source": PBMC_PROCESSED_URL,
    }
    return Dataset(
        name="pbmc3k",
        X=X,
        latent=X,
        clean=X,
        labels=labels.astype(int),
        density=None,
        extra={"hvg": hvg, "cell_type": cell_type},
        meta=meta,
    )


def load_openml_images(
    name: str = "mnist_784", n: int = 10_000, seed: int = 0, data_dir: Path = DATA_DIR
) -> Dataset:
    """A stratified random subsample of MNIST or Fashion-MNIST, pixels scaled to [0, 1].

    X is the raw 784-dim pixel space; ``latent`` is X (no ground truth beyond labels).
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data_id = OPENML_IDS[name]
    path = data_dir / f"openml_{data_id}.pq"
    if not path.exists():
        import urllib.request

        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(openml_parquet_url(data_id), path)
    frame = pd.read_parquet(path)
    X_all = frame.drop(columns="class").to_numpy(dtype=np.float64) / 255.0
    y_all = frame["class"].astype(int).to_numpy()
    if n < len(y_all):
        idx, _ = train_test_split(
            np.arange(len(y_all)), train_size=n, stratify=y_all, random_state=seed
        )
        idx = np.sort(idx)
    else:
        idx = np.arange(len(y_all))
    X = X_all[idx]
    meta = {
        "generator": name,
        "n": len(idx),
        "seed": seed,
        "intrinsic_dim": None,
        "ambient_dim": X.shape[1],
        "betti": None,
        "transforms": [],
        "source": f"https://www.openml.org/d/{data_id}",
        "openml_name": name,
    }
    return Dataset(
        name=name,
        X=X,
        latent=X,
        clean=X,
        labels=y_all[idx],
        density=None,
        extra={"index": idx},
        meta=meta,
    )
