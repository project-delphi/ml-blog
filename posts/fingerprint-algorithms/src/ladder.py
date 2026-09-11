"""The ladder itself: four representations, one gallery, one set of numbers.

Each rung answers the same question in a different currency. Rung 0 keeps the
picture and slides it. Rung 1 keeps a measurement of the ridge flow. Rung 2
keeps a list of landmarks and throws the picture away. Rung 3 keeps a vector
nobody chose the meaning of. Internally they have nothing in common, and the
only reason they can be set side by side is that each one ends in a score
matrix -- one row per probe, one column per enrolled finger, higher meaning more
alike -- which ``bench`` then reads for rank-1, EER and the rest.

Two rules keep the comparison honest, and both cost accuracy:

* **Nothing is scored on a finger it was fitted on.** ``data.split`` divides the
  fingers in two before anything runs. The network trains on one half and every
  rung, trained or not, is scored on the other. The hand-built rungs would score
  the same either way, which is the point: they get no advantage from the rule
  and the network gets no advantage from breaking it.
* **Every rung is charged for the ridge geometry it needs.** Rungs 0+, 1 and 2
  all begin by asking ``ridges.analyse`` which way the ridges run and how far
  apart. Computing that once and sharing it is the only sane way to run the
  ladder, but attributing it to nobody would make three rungs look free, so the
  shared pass is timed once and added to the clock of each rung that reads it.
  The seconds column is therefore what that rung alone would cost, not what this
  script spent.

Run it from the post directory, with ``src`` on the path::

    PYTHONPATH=src ../../.venv-fingerprint-algorithms/bin/python src/fetch_data.py
    PYTHONPATH=src ../../.venv-fingerprint-algorithms/bin/python src/ladder.py

The first line is needed once: the print cache is not committed, so a fresh
checkout has to build it (a 190 MB clone of the NIST repo, thrown away after).
``posts/fingerprint-algorithms/requirements.txt`` has the venv recipe.

``--quick`` cuts the gallery and the training schedule down to something that
finishes while you are still looking at it. The numbers it prints are worse than
the real ones and are for checking that the wiring works, not for quoting.

One warning about reading the output. Rung 2 identifies 3% of probes where rung
0+ identifies 33%, and that gap is not a fact about minutiae -- it is a limit of
this extractor, traced and documented at the top of ``minutiae.py``. The ladder
is a comparison of four implementations, and only one of them is anywhere near
what its method can do.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import bench
import data
import embed
import enhance
import minutiae
import numpy as np
import pixels
import ridges


@contextmanager
def clock(label: str, verbose: bool = True):
    """Time a block and report what it cost. Yields a one-element list to read after."""
    out = [0.0]
    start = time.perf_counter()
    if verbose:
        print(f"  {label} ...", end="", flush=True)
    try:
        yield out
    finally:
        # In a finally, so a rung that raises forty minutes in still closes its
        # line and reports what it had spent, rather than leaving the run with an
        # unterminated "  rung 2 ..." and no timings at all.
        out[0] = time.perf_counter() - start
        if verbose:
            print(f" {out[0]:.1f}s", flush=True)


def analyse(images: np.ndarray) -> list[ridges.Analysis]:
    """Ridge geometry for a stack of prints, in the order they came in."""
    return [ridges.analyse(img) for img in images]


def correlation(gallery, probe, gallery_analyses, probe_analyses, enhanced: bool):
    """Rung 0: best shift-aligned correlation, on the raw or the enhanced image."""
    analyses = (gallery_analyses, probe_analyses) if enhanced else (None, None)

    def prepare(prints, own):
        pairs = zip(prints.images, own if own is not None else [None] * len(prints))
        return np.stack([pixels.code(img, enhanced, a) for img, a in pairs])

    enrolled = prepare(gallery, analyses[0])
    searched = prepare(probe, analyses[1])
    return np.stack([pixels.match_all(p, enrolled) for p in searched])


def fingercode(gallery, probe, gallery_analyses, probe_analyses):
    """Rung 1: Gabor energy over a polar tessellation, compared over rotations."""
    enrolled = np.stack(
        [
            enhance.fingercode(img, a)
            for img, a in zip(gallery.images, gallery_analyses)
        ],
    )
    return np.stack(
        [
            enhance.match(enhance.fingercode(img, a), enrolled)
            for img, a in zip(probe.images, probe_analyses)
        ],
    )


def landmarks(gallery, probe, gallery_analyses, probe_analyses):
    """Rung 2: minutiae, compared by rotation-invariant neighbourhood codes."""
    enrolled = [
        minutiae.extract(img, a) for img, a in zip(gallery.images, gallery_analyses)
    ]
    counts = np.array([len(m) for m in enrolled])
    print(
        f"    minutiae per print: median {np.median(counts):.0f},"
        f" range {counts.min()}-{counts.max()}",
        flush=True,
    )
    return np.stack(
        [
            minutiae.match_all(minutiae.extract(img, a), enrolled)
            for img, a in zip(probe.images, probe_analyses)
        ],
    )


def learned(train, gallery, probe, epochs: int, seed: int):
    """Rung 3: a metric-learning embedding.

    Returns the score matrix, the seconds spent training, and the seconds spent
    encoding and searching. Training is reported apart from the rest because it
    happens once, before any finger is enrolled, and charging it to a search
    would compare a fixed cost with a per-probe one.
    """
    with clock(f"training on {train.n_fingers} fingers, {epochs} epochs") as training:
        model, history = embed.train(
            train.images,
            train.finger,
            epochs=epochs,
            seed=seed,
        )
    print(f"    triplet loss {history[0]:.3f} -> {history[-1]:.3f}", flush=True)

    with clock("encoding and searching") as spent:
        enrolled = embed.encode(model, gallery.images)
        searched = embed.encode(model, probe.images)
        matrix = searched @ enrolled.T
    return matrix, training[0], spent[0]


def run(
    prints: data.Prints,
    epochs: int = 60,
    train_fraction: float = 0.5,
    seed: int = 7,
) -> list[bench.Result]:
    """Score every rung on the test fingers and return the results in ladder order."""
    train, test = data.split(prints, train_fraction, seed)
    gallery, probe, truth = data.gallery_probe(test)
    print(
        f"{prints.n_fingers} fingers: {train.n_fingers} to learn from,"
        f" {test.n_fingers} to search.\n"
        f"Gallery {len(gallery)}, probes {len(probe)}, images {prints.size}px.\n",
        flush=True,
    )

    with clock(f"ridge geometry for {len(test)} prints") as shared:
        gallery_analyses = analyse(gallery.images)
        probe_analyses = analyse(probe.images)
    geometry = shared[0]

    results = []

    with clock("rung 0  pixel correlation") as spent:
        matrix = correlation(gallery, probe, None, None, enhanced=False)
    results.append(bench.score("0  pixel correlation", matrix, truth, spent[0]))

    with clock("rung 0+ pixel correlation, enhanced") as spent:
        matrix = correlation(
            gallery,
            probe,
            gallery_analyses,
            probe_analyses,
            enhanced=True,
        )
    results.append(
        bench.score(
            "0+ pixel correlation, enhanced",
            matrix,
            truth,
            spent[0] + geometry,
        ),
    )

    with clock("rung 1  FingerCode") as spent:
        matrix = fingercode(gallery, probe, gallery_analyses, probe_analyses)
    results.append(
        bench.score(
            "1  FingerCode, Gabor tessellation",
            matrix,
            truth,
            spent[0] + geometry,
        ),
    )

    with clock("rung 2  minutiae") as spent:
        matrix = landmarks(gallery, probe, gallery_analyses, probe_analyses)
    results.append(
        bench.score(
            "2  minutiae, neighbourhood codes",
            matrix,
            truth,
            spent[0] + geometry,
        ),
    )

    print("  rung 3  learned embedding", flush=True)
    matrix, training, searching = learned(train, gallery, probe, epochs, seed)
    results.append(
        bench.score("3  learned embedding, triplet", matrix, truth, searching),
    )
    print(f"  (training cost {training:.0f}s once, and is not in the table)\n")

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=data.DATA / "prints.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="a fifth of the fingers and a tenth of the training; for wiring checks only",
    )
    ap.add_argument("--out", type=Path, help="write the table to this JSON file")
    args = ap.parse_args()

    prints = data.load(args.data)
    epochs = args.epochs
    if args.quick:
        fingers = np.unique(prints.finger)[::5]
        prints = prints.subset(np.isin(prints.finger, fingers))
        epochs = max(1, args.epochs // 10)

    results = run(prints, epochs=epochs, seed=args.seed)
    print(bench.table(results))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "fingers": int(prints.n_fingers),
                    "epochs": epochs,
                    "seed": args.seed,
                    "rows": [r.row() for r in results],
                },
                indent=2,
            )
            + "\n",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
