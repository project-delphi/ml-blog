"""Regenerate every cached result behind the When UMAP Lies series.

Run from the repository root with the series venv:

    .venv-umap/bin/python posts/when-umap-lies/src/run_experiments.py --all
    .venv-umap/bin/python posts/when-umap-lies/src/run_experiments.py noise_dims real
    .venv-umap/bin/python posts/when-umap-lies/src/run_experiments.py --list

Each experiment writes ``cache/<name>/{runs,metrics}.parquet`` (and ``embeddings.npz``
for the cells whose embeddings a figure or widget shows). Runs resume: a cell already
in ``runs.parquet`` is skipped, so an interrupted run continues where it stopped and a
change to an experiment's definition only runs the new cells. Bump
``benchmark.CODE_VERSION`` to invalidate every cell.

The scaling study (``scaling``) is separate: it fits each method in a fresh
subprocess so peak memory can be read, and writes ``cache/scaling.parquet``.

Raw downloads (PBMC 3k, MNIST, Fashion-MNIST) land in ``posts/when-umap-lies/data/``,
which git ignores; see ``umaplab/realdata.py`` for their provenance.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from umaplab.embeddings import set_thread_env  # noqa: E402

set_thread_env(1)

from umaplab import experiments as E  # noqa: E402
from umaplab.benchmark import run_grid  # noqa: E402

CACHE = HERE.parent / "cache"


def run_scaling(out: Path, n_workers: int) -> None:
    """Fit time and peak memory against n, one fresh subprocess per fit."""
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    from umaplab.benchmark import config_label
    from umaplab.embeddings import Infeasible, check_feasible, measure_memory
    from umaplab.generators import add_noise_dims, gaussian_clusters

    rows = []
    done = set()
    if out.exists():
        old = pd.read_parquet(out)
        rows = old.to_dict("records")
        done = {(r["config"], r["n"], r["seed"]) for r in rows}

    def one(job):
        method, mp, n, seed = job
        label = config_label(method, mp)
        if (label, n, seed) in done:
            return None
        ds = add_noise_dims(gaussian_clusters(n, seed, dim=50), 100, 1.0, seed)
        row = {"config": label, "method": method, "n": n, "seed": seed, "D": 100}
        try:
            check_feasible(method, n, 100)
            sec, mb = measure_memory(method, ds.X, seed, mp)
            row.update(seconds=sec, peak_rss_mb=mb, status="ok", error="")
        except Infeasible as exc:
            row.update(seconds=float("nan"), peak_rss_mb=float("nan"))
            row.update(status="infeasible", error=str(exc))
        except Exception as exc:  # recorded, never fatal
            row.update(seconds=float("nan"), peak_rss_mb=float("nan"))
            row.update(status="failed", error=repr(exc)[:500])
        return row

    jobs = E.scaling_grid()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, row in enumerate(pool.map(one, jobs)):
            if row is not None:
                rows.append(row)
            if i % 10 == 0 or i == len(jobs) - 1:
                tmp = out.with_suffix(".tmp.parquet")
                pd.DataFrame(rows).to_parquet(tmp, index=False)
                tmp.replace(out)


def run_elephant(out: Path) -> None:
    """The elephant experiment on PBMC 3k and on ten synthetic Gaussian clusters."""
    import numpy as np
    import pandas as pd
    from umaplab.elephant import run
    from umaplab.embeddings import embed
    from umaplab.generators import gaussian_clusters
    from umaplab.realdata import load_pbmc3k

    out.mkdir(parents=True, exist_ok=True)
    data = {
        "pbmc3k": load_pbmc3k(),
        "clusters": gaussian_clusters(2000, 0, n_clusters=10, dim=10, separation=4.0),
    }
    frames, layouts = [], {}
    for name, ds in data.items():
        others = {}
        for method in ("umap", "tsne"):
            for s in range(5):
                others[f"{method}·s{s}"] = embed(method, ds.X, s).Y
        df, keep = run(name, ds.X, ds.labels, others)
        frames.append(df)
        layouts.update({f"{name}/{k}": v.astype(np.float32) for k, v in keep.items()})
        layouts[f"{name}/labels"] = ds.labels.astype(np.int16)
    pd.concat(frames).to_parquet(out / "scores.parquet", index=False)
    np.savez_compressed(out / "layouts.npz", **layouts)
    save_pbmc_reference(data["pbmc3k"], out)


def save_pbmc_reference(ds, out: Path) -> None:
    """The PBMC reference space Part 1 scores against, so its render needs no download.

    The 50-PC matrix, the labels and the label names, as preprocessed by
    ``umaplab.realdata.load_pbmc3k``.
    """
    import numpy as np

    np.savez_compressed(
        out / "pbmc_reference.npz",
        X=ds.X.astype(np.float32),
        labels=ds.labels.astype(np.int16),
        label_names=np.array(ds.meta["label_names"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="experiments to run")
    ap.add_argument("--all", action="store_true", help="run every experiment")
    ap.add_argument("--list", action="store_true", help="list experiments and sizes")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.list:
        for name, fn in E.EXPERIMENTS.items():
            print(f"{name:15s} {len(fn()):5d} cells")
        print(f"{'scaling':15s} {len(E.scaling_grid()):5d} fits")
        return
    # "derived" reads the grids' caches, so it runs last.
    everything = list(E.EXPERIMENTS) + ["scaling", "elephant", "derived"]
    names = everything if args.all else args.names
    for name in names:
        t0 = time.perf_counter()
        if name == "derived":
            from umaplab.derived import run_all

            run_all()
            print(f"{name}: done in {time.perf_counter() - t0:.0f}s", flush=True)
            continue
        if name == "scaling":
            run_scaling(CACHE / "scaling.parquet", max(1, args.workers // 2))
        elif name == "elephant":
            run_elephant(CACHE / "elephant")
        else:
            cells = E.EXPERIMENTS[name]()
            run_grid(cells, CACHE / name, n_workers=args.workers)
        print(f"{name}: done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
