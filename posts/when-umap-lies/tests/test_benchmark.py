"""Tests for umaplab.benchmark: labels, ids, dataset seeding, failure handling, grid."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from umaplab import benchmark as B
from umaplab import embeddings as E


def _cell(method: str = "pca", D: int = 10, **kw) -> B.Cell:
    base = dict(
        experiment="test",
        generator="swiss_roll",
        gen_params={},
        transforms=({"op": "add_noise_dims", "D": D, "sigma": 0.1},),
        n=200,
        data_seed=0,
        method=method,
        method_params={},
        algo_seed=0,
    )
    base.update(kw)
    return B.Cell(**base)


# ------------------------------------------------------------------ config labels


@pytest.mark.parametrize(
    ("method", "params", "label"),
    [
        ("umap", {}, "umap"),
        ("umap", None, "umap"),
        ("umap", {"pca_dims": 50}, "umap·pca50"),
        ("umap", {"n_neighbors": 5, "min_dist": 0.5}, "umap·nn5·md0.5"),
        ("umap", {"min_dist": 0.5, "n_neighbors": 5}, "umap·nn5·md0.5"),
        ("umap", {"n_neighbors": 15}, "umap"),  # equal to the default: omitted
        ("umap", {"pca_dims": 50, "n_neighbors": 50}, "umap·pca50·nn50"),
        ("umap", {"densmap": True}, "umap·densmap"),
        ("tsne", {"perplexity": 5}, "tsne·perp5"),
        ("tsne", {"init": "random", "perplexity": 5}, "tsne·perp5·init-random"),
        ("tsne", {"exaggeration": 4.0}, "tsne·ex4"),
        ("phate", {"knn": 10, "t": 20}, "phate·knn10·t20"),
        ("pacmap", {"num_iters": (10, 10, 20)}, "pacmap·num_iters10-10-20"),
        ("umap", {"zeta": "x", "alpha_x": 2}, "umap·alpha_x2·zeta-x"),
    ],
)
def test_config_label(method: str, params: dict | None, label: str) -> None:
    """Labels are short, ordered deterministically, and skip default values."""
    assert B.config_label(method, params) == label


# ------------------------------------------------------------------------- ids


def test_cell_id_is_stable_and_sensitive() -> None:
    """Same content -> same id regardless of dict order; any field change -> new id."""
    a = _cell(method_params={"n_neighbors": 5, "min_dist": 0.5})
    b = _cell(method_params={"min_dist": 0.5, "n_neighbors": 5})
    assert B.cell_id(a) == B.cell_id(b)
    assert len(B.cell_id(a)) == 40
    assert B.cell_id(a) != B.cell_id(_cell(method_params={"n_neighbors": 5}))
    assert B.cell_id(_cell()) != B.cell_id(_cell(algo_seed=1))
    assert B.data_key(_cell()) == B.data_key(_cell(method="umap", algo_seed=3))
    assert B.data_key(_cell()) != B.data_key(_cell(data_seed=1))


def test_dataset_depends_on_data_seed_not_algo_seed() -> None:
    """Transform seeds derive from data_seed: algo_seed never changes the data."""
    x0 = B.build_dataset(_cell(algo_seed=0)).X
    x1 = B.build_dataset(_cell(method="umap", algo_seed=7)).X
    x2 = B.build_dataset(_cell(data_seed=1)).X
    np.testing.assert_array_equal(x0, x1)
    assert not np.allclose(x0, x2)
    assert x0.shape == (200, 10)


def test_dataset_columns_from_meta() -> None:
    """D, intrinsic_dim and noise_sigma come from the built dataset."""
    cols = B._dataset_columns(B.build_dataset(_cell(D=50)))
    assert cols["D"] == 50
    assert cols["intrinsic_dim"] == 2
    assert cols["noise_sigma"] == pytest.approx(0.1)
    cols = B._dataset_columns(B.build_dataset(_cell(transforms=())))
    assert cols["D"] == 3 and math.isnan(cols["noise_sigma"])


# ------------------------------------------------------------------ run_cell


def test_run_cell_ok() -> None:
    """A healthy cell gives an ok row and long-format metric rows."""
    row, mrows, Y = B.run_cell(_cell(keep_embedding=True))
    assert row["status"] == "ok", row["error"]
    assert row["config"] == "pca" and row["D"] == 10
    assert Y is not None and Y.shape == (200, 2)
    assert {r["reference"] for r in mrows} >= {"truth", "input"}
    assert all(set(r) == set(B.METRIC_COLUMNS) for r in mrows)


def test_run_cell_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A method that raises becomes status='failed'; Infeasible becomes 'infeasible'."""

    def boom(*a, **k):
        raise ValueError("kaboom")

    monkeypatch.setattr(E, "embed", boom)
    row, mrows, Y = B.run_cell(_cell())
    assert row["status"] == "failed" and "kaboom" in row["error"]
    assert mrows == [] and Y is None

    def infeasible(*a, **k):
        raise E.Infeasible("too big")

    monkeypatch.setattr(E, "embed", infeasible)
    row, _, _ = B.run_cell(_cell())
    assert row["status"] == "infeasible"


def test_float16_storage_rescales_large_values() -> None:
    """Embeddings beyond float16 range are scaled by a recorded power of two."""
    info: dict = {}
    Y = np.array([[1e5, -2e5], [3.0, 4.0]], dtype=np.float32)
    S = B._storable(Y, info)
    assert S.dtype == np.float16 and np.all(np.isfinite(S))
    np.testing.assert_allclose(
        S.astype(np.float64) / info["embedding_scale"], Y, rtol=2e-3
    )
    info = {}
    B._storable(np.ones((3, 2), np.float32), info)
    assert "embedding_scale" not in info


# ----------------------------------------------------------------- integration


def _grid(keep: bool = True) -> list[B.Cell]:
    return [
        _cell(method=m, D=D, keep_embedding=keep and m == "umap")
        for m in ("pca", "umap")
        for D in (5, 10, 50)
    ]


def _check_schema(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_parquet(out / "runs.parquet")
    mets = pd.read_parquet(out / "metrics.parquet")
    assert list(runs.columns) == B.RUN_COLUMNS
    assert list(mets.columns) == B.METRIC_COLUMNS
    for col in ("seconds", "peak_rss_mb", "noise_sigma"):
        assert runs[col].dtype == np.float64
    for col in ("n", "data_seed", "algo_seed"):
        assert runs[col].dtype == np.int64
    assert mets["k"].dtype == np.float64 and mets["value"].dtype == np.float64
    assert set(mets["cell_id"]) <= set(runs["cell_id"])
    return runs, mets


def test_grid_end_to_end_with_resume(tmp_path: Path) -> None:
    """6 cells on 2 spawn workers: both tables, embeddings, input rows, then resume."""
    cells = _grid() + [_cell(method="umap", D=5, method_params={"n_neighbors": -1})]
    B.run_grid(cells, tmp_path, n_workers=2, flush_every=3)
    runs, mets = _check_schema(tmp_path)

    grid = runs[runs["method"] != "input"]
    assert len(grid) == 7
    bad = grid[grid["method_params"] == '{"n_neighbors":-1}']
    assert bad["status"].tolist() == ["failed"] and bad["error"].iloc[0]
    good = grid.drop(bad.index)
    assert (good["status"] == "ok").all(), good["error"].tolist()
    assert sorted(good["D"].tolist()) == [5, 5, 10, 10, 50, 50]
    assert set(good["config"]) == {"pca", "umap"}
    assert (good["seconds"] > 0).all() and good["peak_rss_mb"].isna().all()
    assert set(good["cell_id"]) <= set(mets["cell_id"])

    inputs = runs[runs["method"] == "input"]
    assert len(inputs) == 3 and inputs["cell_id"].str.endswith(":input").all()
    assert (inputs["status"] == "ok").all()
    assert set(inputs["cell_id"]) <= set(mets["cell_id"])

    with np.load(tmp_path / "embeddings.npz") as npz:
        kept = set(good.loc[good["method"] == "umap", "cell_id"])
        assert set(npz.files) == kept
        assert all(
            npz[k].dtype == np.float16 and npz[k].shape == (200, 2) for k in kept
        )

    # Resume reruns only the failed cell (failures are retried); ok rows are kept
    # as they were, and nothing is duplicated.
    ok_before = runs[runs["status"] == "ok"].set_index("cell_id")["seconds"]
    B.run_grid(cells, tmp_path, n_workers=2)
    after = pd.read_parquet(tmp_path / "runs.parquet")
    assert len(after) == len(runs) and after["cell_id"].is_unique
    ok_after = after[after["status"] == "ok"].set_index("cell_id")["seconds"]
    pd.testing.assert_series_equal(ok_before.sort_index(), ok_after.sort_index())
    assert (
        after.loc[after["cell_id"] == bad["cell_id"].iloc[0], "status"].item()
        == "failed"
    )


def test_grid_survives_a_failing_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A monkeypatched failing method gives status='failed' and the grid finishes.

    Monkeypatching cannot reach spawn workers, so this runs the grid in-process.
    """
    real = E.embed

    def flaky(method, X, seed, params=None, n_jobs=1):
        if method == "umap":
            raise RuntimeError("umap exploded")
        return real(method, X, seed, params, n_jobs)

    monkeypatch.setattr(E, "embed", flaky)
    B.run_grid(_grid(keep=False), tmp_path, n_workers=0)
    runs, _ = _check_schema(tmp_path)
    grid = runs[runs["method"] != "input"]
    assert len(grid) == 6
    assert (grid.loc[grid["method"] == "umap", "status"] == "failed").all()
    assert (grid.loc[grid["method"] == "pca", "status"] == "ok").all()
    assert grid["error"].str.contains("umap exploded").sum() == 3


def test_worker_crash_suspects_split_singly(tmp_path: Path) -> None:
    """A crashed task's unfinished cells become one-cell suspects, none failed yet."""
    store = B._Store(tmp_path, resume=False)
    a, b = _cell(D=5), _cell(D=10)
    task = B._Task((a, b), a)
    suspects = B._split_unfinished(task, store)
    assert [t.cells for t in suspects] == [(a,), (b,), ()]
    assert suspects[2].input_cell == a
    assert store.new_runs == []  # being in flight during a crash is not a failure


def test_crash_alone_is_recorded_failed(tmp_path: Path) -> None:
    store = B._Store(tmp_path, resume=False)
    a = _cell(D=5)
    B._record_crash(B._Task((a,)), store)
    B._record_crash(B._Task((), a), store)
    assert {B.cell_id(a), B.input_cell_id(a)} <= store.done
    assert all(r["status"] == "failed" for r in store.new_runs)


def test_failed_rows_are_retried_on_resume(tmp_path: Path) -> None:
    store = B._Store(tmp_path, resume=False)
    a = _cell(D=5)
    B._record_crash(B._Task((a,)), store)
    store.flush()
    again = B._Store(tmp_path, resume=True)
    assert B.cell_id(a) not in again.done


def test_dataset_cache_is_a_bounded_lru() -> None:
    """Repeated data keys hit the per-process cache; old keys are evicted."""
    B._CACHE.clear()
    c = _cell(D=7)
    assert B._dataset(c) is B._dataset(_cell(D=7, method="umap", algo_seed=4))
    for seed in range(1, B.DATASET_CACHE_SIZE + 2):
        B._dataset(_cell(D=7, data_seed=seed))
    assert len(B._CACHE) == B.DATASET_CACHE_SIZE
    assert B._dataset(c) is not None and len(B._CACHE) == B.DATASET_CACHE_SIZE
