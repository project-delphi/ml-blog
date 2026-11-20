"""Grid runner: cells, deterministic ids, dataset caching, and the result tables.

A :class:`Cell` is one (dataset, method, seed) combination. :func:`run_cell` turns it
into one ``runs`` row, a list of long-format ``metrics`` rows and optionally the
embedding; :func:`run_grid` runs many cells in a spawn process pool and appends the
results to ``runs.parquet``, ``metrics.parquet`` and ``embeddings.npz``.

Conventions beyond lab-spec section 6, all additive:

* ``runs.parquet`` carries one extra column, ``info`` (JSON): the method's effective
  params, warnings caught during the fit, method diagnostics (KL divergence, PHATE's
  chosen t, ...) and, when the stored float16 embedding had to be rescaled to fit,
  ``embedding_scale``.
* Input-fidelity rows (``method="input"``) get a ``runs`` row too, with cell_id
  ``<data key>:input``, ``config="input"`` and ``algo_seed=-1``, so every metrics
  cell_id joins to a run. The experiment is that of the first cell seen for the key.
* Topology rows are computed when the base generator's ``meta["betti"]`` has an H1
  entry, or when the experiment name contains ``"topology"``.
* ``peak_rss_mb`` is measured (fit in a fresh subprocess) only for experiments whose
  name starts with ``"scaling"``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from umaplab import embeddings as E

CODE_VERSION = "v1"

KS = (5, 10, 30, 100)
DATASET_CACHE_SIZE = 4
DATASET_CACHE_MB = 512  # per worker; 8 workers x 512 MB stays well inside 16 GB
FLOAT16_SAFE = 30_000.0  # rescale stored embeddings whose |values| exceed this

RUN_COLUMNS = [
    "cell_id", "experiment", "generator", "gen_params", "transforms", "D",
    "intrinsic_dim", "noise_sigma", "n", "data_seed", "method", "method_params",
    "config", "algo_seed", "seconds", "peak_rss_mb", "status", "error", "info",
]  # fmt: skip
METRIC_COLUMNS = ["cell_id", "metric", "k", "reference", "value"]


@dataclass(frozen=True)
class Cell:
    """One experiment cell: a dataset recipe, a method with params, and the seeds."""

    experiment: str
    generator: str
    gen_params: dict
    transforms: tuple[dict, ...]
    n: int
    data_seed: int
    method: str
    method_params: dict
    algo_seed: int
    keep_embedding: bool = False


# ------------------------------------------------------------------- identifiers


def _jsonable(obj: Any) -> Any:
    """Convert tuples, numpy scalars and arrays to plain JSON types, recursively."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, tuples as lists."""
    return json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"))


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def cell_id(c: Cell) -> str:
    """Stable sha1 over a canonical JSON of every field of the cell plus CODE_VERSION.

    Note that JSON distinguishes 1 from 1.0, so write params consistently.
    """
    return _sha1(canonical_json({"cell": asdict(c), "code_version": CODE_VERSION}))


def _data_spec(c: Cell) -> dict[str, Any]:
    return {
        "generator": c.generator,
        "gen_params": c.gen_params,
        "transforms": list(c.transforms),
        "n": c.n,
        "data_seed": c.data_seed,
    }


def data_key(c: Cell) -> str:
    """Short stable id of the cell's dataset (generator, params, transforms, n, seed)."""
    spec = {"data": _data_spec(c), "code_version": CODE_VERSION}
    return _sha1(canonical_json(spec))[:16]


def input_cell_id(c: Cell) -> str:
    """The synthetic cell_id of the input-fidelity rows of the cell's dataset."""
    return f"{data_key(c)}:input"


_LABELS: dict[str, str] = {
    "pca_dims": "pca",
    "n_neighbors": "nn",
    "knn": "knn",
    "k": "k",
    "perplexity": "perp",
    "min_dist": "md",
    "spread": "sp",
    "init": "init-",
    "initialization": "init-",
    "early_exaggeration": "ee",
    "exaggeration": "ex",
    "learning_rate": "lr",
    "lr": "lr",
    "n_epochs": "ep",
    "n_iter": "it",
    "n_iters": "it",
    "MN_ratio": "mn",
    "FP_ratio": "fp",
    "decay": "decay",
    "alpha": "alpha",
    "t": "t",
    "densmap": "densmap",
    "dof": "dof",
}
_LABEL_ORDER = list(_LABELS)


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "-".join(_fmt_value(x) for x in v)
    return str(v)


def config_label(method: str, method_params: dict | None) -> str:
    """Method name plus a short label per non-default param, joined with "·".

    Known params get fixed short names in a fixed order (``pca_dims=50`` -> ``pca50``,
    ``n_neighbors=5`` -> ``nn5``, ``min_dist=0.5`` -> ``md0.5``, ``perplexity=5`` ->
    ``perp5``, ``init="random"`` -> ``init-random``); unknown ones follow sorted by
    name as ``name<value>``, with ``-`` before a string value. Params equal to the
    registry default are omitted. Booleans: True -> ``name``, False -> ``no-name``.
    """
    params = dict(method_params or {})
    defaults: dict[str, Any] = {"pca_dims": None}
    if method in E.METHODS:
        defaults.update(E.METHODS[method].defaults)
    params = {k: v for k, v in params.items() if k not in defaults or defaults[k] != v}
    known = [k for k in _LABEL_ORDER if k in params]
    unknown = sorted(k for k in params if k not in _LABELS)
    parts = [method]
    for key in known + unknown:
        v = params[key]
        short = _LABELS.get(key, key)
        if isinstance(v, bool):
            parts.append(short.rstrip("-") if v else f"no-{short.rstrip('-')}")
            continue
        if isinstance(v, str) and not short.endswith("-"):
            short += "-"
        if v is None:
            parts.append(f"{short.rstrip('-')}-none")
            continue
        parts.append(f"{short}{_fmt_value(v)}")
    return "·".join(parts)


# ---------------------------------------------------------------------- datasets


def _transform_seed(data_seed: int, position: int) -> int:
    """Seed for the transform at ``position``, derived from data_seed only."""
    return int(np.random.SeedSequence([data_seed, 7919, position]).generate_state(1)[0])


def _build(spec_json: str) -> Any:
    from umaplab import generators as G

    spec = json.loads(spec_json)
    if spec["generator"] == "pbmc3k":
        from umaplab.realdata import load_pbmc3k

        ds = load_pbmc3k(seed=0, **spec["gen_params"])  # one fixed dataset; n ignored
    elif spec["generator"] in ("mnist_784", "Fashion-MNIST"):
        from umaplab.realdata import load_openml_images

        ds = load_openml_images(
            spec["generator"], n=spec["n"], seed=spec["data_seed"], **spec["gen_params"]
        )
    else:
        gen = G.GENERATORS[spec["generator"]]
        ds = gen(spec["n"], seed=spec["data_seed"], **spec["gen_params"])
    for pos, t in enumerate(spec["transforms"]):
        t = dict(t)
        op = t.pop("op")
        if op not in ("add_noise_dims", "add_isotropic_noise", "random_isometry"):
            raise ValueError(f"unknown transform op {op!r}")
        t.setdefault("seed", _transform_seed(spec["data_seed"], pos))
        ds = getattr(G, op)(ds, **t)
    return ds


_CACHE: OrderedDict[str, Any] = OrderedDict()


def _nbytes(ds: Any) -> int:
    arrays = [ds.X, ds.latent, ds.clean, ds.labels, ds.density, *ds.extra.values()]
    return sum(a.nbytes for a in arrays if isinstance(a, np.ndarray))


def _build_cached(spec_json: str) -> Any:
    """LRU over datasets: at most DATASET_CACHE_SIZE entries and DATASET_CACHE_MB."""
    if spec_json in _CACHE:
        _CACHE.move_to_end(spec_json)
        return _CACHE[spec_json]
    ds = _build(spec_json)
    _CACHE[spec_json] = ds
    budget = DATASET_CACHE_MB * 2**20
    while len(_CACHE) > 1 and (
        len(_CACHE) > DATASET_CACHE_SIZE
        or sum(_nbytes(d) for d in _CACHE.values()) > budget
    ):
        _CACHE.popitem(last=False)
    return ds


def build_dataset(c: Cell) -> Any:
    """Generator + transforms; transform seeds derive from data_seed, never algo_seed.

    The base generator gets ``seed=data_seed``; the transform at position i gets
    ``SeedSequence([data_seed, 7919, i])`` unless its dict names a seed explicitly.
    """
    return _build(canonical_json(_data_spec(c)))


def _dataset(c: Cell) -> Any:
    """build_dataset through this process's small LRU cache (key = data spec)."""
    return _build_cached(canonical_json(_data_spec(c)))


def _dataset_columns(ds: Any) -> dict[str, Any]:
    """D, intrinsic_dim and noise_sigma from a built dataset's meta."""
    meta = ds.meta
    D = int(meta.get("ambient_dim", ds.X.shape[1]))
    intrinsic = meta.get("intrinsic_dim")
    sigmas = [float(t["sigma"]) for t in meta.get("transforms", []) if "sigma" in t]
    if sigmas:
        # Only one noise transform per recipe is expected; with several, report the
        # largest (the post's noise axis), and the full history is in `transforms`.
        sigma = max(sigmas)
    else:
        params = meta.get("params", meta)
        noise = params.get("noise")
        sigma = (
            float(noise) if isinstance(noise, (int, float)) and noise > 0 else math.nan
        )
    return {
        "D": D,
        "intrinsic_dim": int(intrinsic) if intrinsic is not None else None,
        "noise_sigma": sigma,
    }


# ------------------------------------------------------------------------ a cell


def _needs_topology(c: Cell, ds: Any) -> bool:
    betti = ds.meta.get("betti")
    return "topology" in c.experiment or (betti is not None and len(betti) >= 2)


def _base_row(c: Cell) -> dict[str, Any]:
    return {
        "cell_id": cell_id(c),
        "experiment": c.experiment,
        "generator": c.generator,
        "gen_params": canonical_json(c.gen_params),
        "transforms": canonical_json(list(c.transforms)),
        "D": None,
        "intrinsic_dim": None,
        "noise_sigma": math.nan,
        "n": c.n,
        "data_seed": c.data_seed,
        "method": c.method,
        "method_params": canonical_json(c.method_params),
        "config": config_label(c.method, c.method_params),
        "algo_seed": c.algo_seed,
        "seconds": math.nan,
        "peak_rss_mb": math.nan,
        "status": "ok",
        "error": "",
        "info": "{}",
    }


def _error_text(exc: BaseException) -> str:
    tb = traceback.extract_tb(exc.__traceback__)
    where = f" [{Path(tb[-1].filename).name}:{tb[-1].lineno}]" if tb else ""
    return f"{type(exc).__name__}: {exc}{where}"[:2000]


def _storable(Y: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    """float16 copy of Y, rescaled by a power of two (recorded) if it would overflow."""
    peak = float(np.max(np.abs(Y))) if Y.size else 0.0
    scale = 1.0
    if peak > FLOAT16_SAFE:
        scale = 2.0 ** -math.ceil(math.log2(peak / FLOAT16_SAFE))
        info["embedding_scale"] = scale
    return (Y * scale).astype(np.float16)


def run_cell(c: Cell) -> tuple[dict, list[dict], np.ndarray | None]:
    """Generate (cached per process by data key), embed, evaluate (+ topology).

    Returns (run_row, metric_rows, embedding_or_None); the embedding is float32 and
    only returned for ``keep_embedding`` cells that succeeded. Never raises: an
    infeasible method gives status="infeasible", any other error status="failed" with
    the message, and no metric rows.
    """
    row = _base_row(c)
    info: dict[str, Any] = {}
    try:
        ds = _dataset(c)
        row.update(_dataset_columns(ds))
        if c.experiment.startswith("scaling"):
            res, rss = E.embed_in_subprocess(
                c.method, ds.X, c.algo_seed, c.method_params
            )
            row["peak_rss_mb"] = rss
        else:
            res = E.embed(c.method, ds.X, c.algo_seed, c.method_params)
        row["seconds"] = res.seconds
        info.update(res.info)
        info["effective_params"] = res.params

        from umaplab import metrics as M

        Y = res.Y.astype(np.float64)
        rows = M.evaluate(ds, Y, ks=KS, seed=c.data_seed)
        if _needs_topology(c, ds):
            from umaplab import topology as T

            rows += T.topology_rows(ds, Y, c.data_seed)
        cid = row["cell_id"]
        metric_rows = [_metric_row(cid, r) for r in rows]
        row["info"] = canonical_json(info)
        return row, metric_rows, (res.Y if c.keep_embedding else None)
    except E.Infeasible as exc:
        row.update(status="infeasible", error=str(exc), info=canonical_json(info))
    except Exception as exc:  # noqa: BLE001 - a cell must never kill the grid
        row.update(status="failed", error=_error_text(exc), info=canonical_json(info))
    return row, [], None


def _metric_row(cid: str, r: dict[str, Any]) -> dict[str, Any]:
    k = r.get("k")
    return {
        "cell_id": cid,
        "metric": str(r["metric"]),
        "k": math.nan if k is None else float(k),
        "reference": str(r["reference"]),
        "value": float(r["value"]),
    }


def run_input(c: Cell) -> tuple[dict, list[dict]]:
    """Input-fidelity rows for the cell's dataset (method="input"); never raises."""
    row = _base_row(c)
    row.update(
        cell_id=input_cell_id(c),
        method="input",
        method_params="{}",
        config="input",
        algo_seed=-1,
    )
    try:
        from umaplab import metrics as M

        ds = _dataset(c)
        row.update(_dataset_columns(ds))
        t0 = time.perf_counter()
        rows = M.input_fidelity(ds, KS)
        row["seconds"] = time.perf_counter() - t0
        return row, [_metric_row(row["cell_id"], r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        row.update(status="failed", error=_error_text(exc))
        return row, []


# ---------------------------------------------------------------------- workers

_WARMED: dict[str, set[int]] = {}


def _init_worker() -> None:
    """Pin every thread pool to one thread before anything numba-based is imported."""
    E.set_thread_env(1)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001, S110 - best effort; the env vars already apply
        pass


@dataclass(frozen=True)
class _Task:
    """A worker task: cells sharing one dataset, and optionally its input rows."""

    cells: tuple[Cell, ...]
    input_cell: Cell | None = None


def _run_task(
    task: _Task, warm: bool = True
) -> tuple[list[tuple[dict, list[dict], np.ndarray | None]], tuple | None]:
    """Run a task's cells, then its input-fidelity rows if asked.

    Each method is warmed up once per process (imports + numba JIT on throwaway data)
    so the first cell's ``seconds`` is not charged with compilation.
    """
    out = []
    for c in task.cells:
        if warm and c.method in E.METHODS:
            done = _WARMED.setdefault(c.method, set())
            todo = tuple(s for s in E.warmup_sizes(c.method, c.n) if s not in done)
            if todo:
                E.warm_up(c.method, c.method_params, sizes=todo)
                done.update(todo)
        out.append(run_cell(c))
    inp = run_input(task.input_cell) if task.input_cell is not None else None
    return out, inp


# ------------------------------------------------------------------------ storage


def _atomic_parquet(df: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _runs_frame(rows: list[dict]) -> Any:
    import pandas as pd

    df = pd.DataFrame(rows, columns=RUN_COLUMNS)
    for col in ("D", "intrinsic_dim"):
        df[col] = df[col].astype("Int64")
    for col in ("n", "data_seed", "algo_seed"):
        df[col] = df[col].astype("int64")
    for col in ("noise_sigma", "seconds", "peak_rss_mb"):
        df[col] = df[col].astype("float64")
    return df


def _metrics_frame(rows: list[dict]) -> Any:
    import pandas as pd

    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    df["k"] = df["k"].astype("float64")
    df["value"] = df["value"].astype("float64")
    return df


class _Store:
    """In-memory copy of the three result files, flushed atomically in batches."""

    def __init__(self, out: Path, resume: bool) -> None:
        import pandas as pd

        self.out = out
        self.runs_path = out / "runs.parquet"
        self.metrics_path = out / "metrics.parquet"
        self.emb_path = out / "embeddings.npz"
        self.runs = (
            pd.read_parquet(self.runs_path)
            if resume and self.runs_path.exists()
            else None
        )
        self.metrics = (
            pd.read_parquet(self.metrics_path)
            if resume and self.metrics_path.exists()
            else None
        )
        if not resume:
            # Clear all three together, so an interrupted first flush cannot leave
            # old runs/metrics beside a deleted embeddings file.
            for path in (self.runs_path, self.metrics_path, self.emb_path):
                path.unlink(missing_ok=True)
        if self.runs is not None:
            # Failed cells (a download error, a native crash) are retried on resume:
            # drop their rows so they run again. Infeasible cells stay done.
            failed = set(self.runs.loc[self.runs.status == "failed", "cell_id"])
            if failed:
                self.runs = self.runs[~self.runs.cell_id.isin(failed)]
                if self.metrics is not None:
                    self.metrics = self.metrics[~self.metrics.cell_id.isin(failed)]
        self.done: set[str] = set() if self.runs is None else set(self.runs["cell_id"])
        self.new_runs: list[dict] = []
        self.new_metrics: list[dict] = []
        self.new_emb: dict[str, np.ndarray] = {}

    def add(self, row: dict, metric_rows: list[dict], Y: np.ndarray | None) -> None:
        if row["cell_id"] in self.done:
            return
        if Y is not None:
            info = json.loads(row["info"])
            self.new_emb[row["cell_id"]] = _storable(Y, info)
            row["info"] = canonical_json(info)
        self.done.add(row["cell_id"])
        self.new_runs.append(row)
        self.new_metrics.extend(metric_rows)

    @property
    def pending(self) -> int:
        return len(self.new_runs)

    def flush(self) -> None:
        import pandas as pd

        if not self.new_runs:
            return
        runs = _runs_frame(self.new_runs)
        self.runs = (
            runs
            if self.runs is None
            else pd.concat([self.runs, runs], ignore_index=True)
        )
        if self.new_metrics or self.metrics is None:
            mets = _metrics_frame(self.new_metrics)
            self.metrics = (
                mets
                if self.metrics is None
                else pd.concat([self.metrics, mets], ignore_index=True)
            )
        if self.new_emb:
            self._append_embeddings()
        # Metrics first: a crash between the two writes leaves runs.parquet without
        # those cells, so a resume recomputes them instead of losing their metrics.
        _atomic_parquet(self.metrics, self.metrics_path)
        _atomic_parquet(self.runs, self.runs_path)
        self.new_runs, self.new_metrics, self.new_emb = [], [], {}

    def _append_embeddings(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        if self.emb_path.exists():
            with np.load(self.emb_path) as old:
                arrays = {k: old[k] for k in old.files}
        arrays.update(self.new_emb)
        tmp = self.emb_path.with_name(self.emb_path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, self.emb_path)


# --------------------------------------------------------------------------- grid


def _plan(cells: Iterable[Cell], store: _Store, size: int) -> list[_Task]:
    """Group pending cells by data key (cache locality) and split groups into tasks.

    The first task of each dataset whose input rows are missing also computes them;
    a dataset whose cells are all done but whose input rows are missing gets an
    input-only task.
    """
    groups: dict[str, list[Cell]] = {}
    firsts: dict[str, Cell] = {}
    seen: set[str] = set()
    for c in cells:
        key = data_key(c)
        firsts.setdefault(key, c)
        cid = cell_id(c)
        if cid in store.done or cid in seen:
            continue
        seen.add(cid)
        groups.setdefault(key, []).append(c)
    tasks = []
    for key, first in firsts.items():
        group = groups.get(key, [])
        want_input = input_cell_id(first) not in store.done
        if not group:
            if want_input:
                tasks.append(_Task((), first))
            continue
        for i in range(0, len(group), size):
            inp = first if (want_input and i == 0) else None
            tasks.append(_Task(tuple(group[i : i + size]), inp))
    return tasks


def _crash_row(c: Cell, *, as_input: bool = False) -> dict:
    row = _base_row(c)
    if as_input:
        row.update(
            cell_id=input_cell_id(c), method="input", method_params="{}",
            config="input", algo_seed=-1,
        )  # fmt: skip
    row.update(
        status="failed",
        error="worker process died running this cell alone "
        "(likely out of memory or a native crash)",
    )
    return row


def run_grid(
    cells: list[Cell],
    out: Path,
    n_workers: int = 8,
    resume: bool = True,
    *,
    flush_every: int = 25,
    chunk_size: int = 4,
) -> None:
    """Run cells in a process pool, appending to out/runs|metrics.parquet and npz.

    Threads are pinned to 1 (set in this process's environment before the pool is
    created, and again in each worker's initializer). Resume skips cell_ids already
    in runs.parquet, including input-fidelity ids. Results are flushed atomically
    every ``flush_every`` finished cells and at the end. ``n_workers=0`` runs serially
    in this process (for debugging; no thread pinning, no warm-up). Without resume,
    existing result files are replaced.

    A worker that dies (out of memory, segfault) breaks the pool, so every cell in
    flight is only a suspect. The pool is rebuilt for the rest of the queue; the
    suspects then run one at a time in a one-worker pool, and only a cell that kills
    its worker there is recorded failed. Failed rows are retried on the next resume.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    store = _Store(out, resume)
    queue = _plan(cells, store, chunk_size)
    try:
        if n_workers <= 0:
            for task in queue:
                _collect(store, _run_task(task, warm=False), flush_every)
            return
        E.set_thread_env(1)
        ctx = get_context("spawn")
        suspects: list[_Task] = []
        while queue:
            queue, crashed = _run_pool(queue, n_workers, ctx, store, flush_every)
            suspects += crashed
        # A crash breaks the whole pool, so every task in flight is only a suspect.
        # Rerun suspects one at a time: a crash there is the cell's own.
        for task in suspects:
            _, crashed = _run_pool([task], 1, ctx, store, flush_every)
            for t in crashed:
                _record_crash(t, store)
    finally:
        store.flush()


def _collect(store: _Store, result: tuple, flush_every: int) -> None:
    outs, inp = result
    for row, mrows, Y in outs:
        store.add(row, mrows, Y)
    if inp is not None:
        store.add(inp[0], inp[1], None)
    if store.pending >= flush_every:
        store.flush()


def _run_pool(
    queue: list[_Task],
    n_workers: int,
    ctx: Any,
    store: _Store,
    flush_every: int,
) -> tuple[list[_Task], list[_Task]]:
    """Run the queue in one pool.

    Returns (tasks not yet started, suspects): when the pool breaks, the unfinished
    cells of every task in flight come back as one-cell suspect tasks.
    """
    queue = list(queue)
    retry: list[_Task] = []
    inflight: dict[Future, _Task] = {}
    broken = False
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx, initializer=_init_worker
    ) as pool:
        while (queue and not broken) or inflight:
            # At most n_workers tasks in flight, so a crash implicates only tasks
            # that were actually running, not the whole backlog.
            while queue and not broken and len(inflight) < n_workers:
                task = queue.pop(0)
                try:
                    inflight[pool.submit(_run_task, task)] = task
                except BrokenProcessPool:
                    broken = True
                    queue.insert(0, task)
            if not inflight:
                break
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                task = inflight.pop(fut)
                try:
                    _collect(store, fut.result(), flush_every)
                except BrokenProcessPool:
                    broken = True
                    retry += _split_unfinished(task, store)
                except Exception as exc:  # noqa: BLE001 - e.g. an unpicklable result
                    for c in task.cells:
                        row = _base_row(c)
                        row.update(status="failed", error=_error_text(exc))
                        store.add(row, [], None)
    return queue, retry


def _split_unfinished(task: _Task, store: _Store) -> list[_Task]:
    """The unfinished cells of a task that was in flight when the pool broke."""
    out: list[_Task] = [_Task((c,)) for c in task.cells if cell_id(c) not in store.done]
    c = task.input_cell
    if c is not None and input_cell_id(c) not in store.done:
        out.append(_Task((), c))
    return out


def _record_crash(task: _Task, store: _Store) -> None:
    """A one-cell task that killed its worker when running alone: record it failed."""
    for c in task.cells:
        store.add(_crash_row(c), [], None)
    if task.input_cell is not None:
        store.add(_crash_row(task.input_cell, as_input=True), [], None)
