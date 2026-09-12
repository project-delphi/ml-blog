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
this extractor, traced and documented at the top of ``minutiae.py``. Rung 2+ is
the attempt to fix it, by ridge following with a quality map instead of crossing
numbers on a thinned skeleton (``minutiae_follow.py``); it finds more landmarks
and scores no better, and its own ``--repeatability`` gate says why. The ladder
is a comparison of six implementations, and only two of them -- rung 0+ and
rung 1 -- are anywhere near what their method can do.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import bench
import data
import embed
import enhance
import minutiae
import minutiae_follow
import numpy as np
import pixels
import ridges

IMPOSTOR_SAMPLE = 10000  # impostor scores kept per rung in --dump


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


def landmarks(gallery, probe, gallery_analyses, probe_analyses, extractor=None):
    """Rung 2: minutiae, compared by rotation-invariant neighbourhood codes.

    `extractor` selects which minutiae finder is on trial -- the crossing-number
    pipeline in ``minutiae`` or the ridge-following one in ``minutiae_follow``.
    Both end in the same descriptor and the same matcher, so the rung measures
    the extractor and nothing else.
    """
    extractor = extractor or minutiae.extract
    enrolled = [extractor(img, a) for img, a in zip(gallery.images, gallery_analyses)]
    counts = np.array([len(m) for m in enrolled])
    print(
        f"    minutiae per print: median {np.median(counts):.0f},"
        f" range {counts.min()}-{counts.max()}",
        flush=True,
    )
    return np.stack(
        [
            minutiae.match_all(extractor(img, a), enrolled)
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
    # The loss history is reported because its *level* is the diagnostic: a
    # batch-hard triplet loss that settles at the margin has collapsed the
    # embedding. See the note at the top of embed.py.
    spread = float(enrolled.std(axis=0).mean())
    return matrix, training[0], spent[0], history, spread


def turn_probes(probe: data.Prints, degrees: float, seed: int) -> data.Prints:
    """Rotate every probe by its own random angle, up to +/- `degrees`.

    The inked cards in this cache are all roughly upright, which quietly flatters
    the two rungs that cannot search over rotation at all. A finger on a phone
    lands at whatever angle it lands at, so this is the edge case that decides
    whether a representation is usable there -- and it is cheaper to measure it
    than to assert it.
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-degrees, degrees, len(probe.images))
    turned = np.stack(
        [
            np.clip(
                minutiae_follow.rotate_about_centre(img.astype(float), np.deg2rad(a)),
                0,
                255,
            ).astype(np.uint8)
            for img, a in zip(probe.images, angles)
        ],
    )
    return replace(probe, images=turned)


def run(
    prints: data.Prints,
    epochs: int = 60,
    train_fraction: float = 0.5,
    seed: int = 7,
    rotate_probes: float = 0.0,
) -> tuple[list[bench.Result], dict]:
    """Score every rung on the test fingers; return the results and the run's notes."""
    train, test = data.split(prints, train_fraction, seed)
    gallery, probe, truth = data.gallery_probe(test)
    if rotate_probes:
        probe = turn_probes(probe, rotate_probes, seed)
        print(f"Probes rotated by up to +/-{rotate_probes:.0f} degrees.\n", flush=True)
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

    with clock("rung 2  minutiae, crossing number") as spent:
        matrix = landmarks(gallery, probe, gallery_analyses, probe_analyses)
    results.append(
        bench.score(
            "2  minutiae, crossing number",
            matrix,
            truth,
            spent[0] + geometry,
        ),
    )

    with clock("rung 2+ minutiae, ridge following") as spent:
        matrix = landmarks(
            gallery,
            probe,
            gallery_analyses,
            probe_analyses,
            extractor=minutiae_follow.extract,
        )
    results.append(
        bench.score(
            "2+ minutiae, ridge following",
            matrix,
            truth,
            spent[0] + geometry,
        ),
    )

    print("  rung 3  learned embedding", flush=True)
    matrix, training, searching, history, spread = learned(
        train,
        gallery,
        probe,
        epochs,
        seed,
    )
    results.append(
        bench.score("3  learned embedding, triplet", matrix, truth, searching),
    )
    print(f"  (training cost {training:.0f}s once, and is not in the table)")
    print(
        f"    embedding spread per dimension {spread:.2e};"
        f" loss ended at {history[-1]:.3f} against a margin of {embed.MARGIN}"
        f" -- at the margin means collapsed\n",
    )

    meta = {
        "loss_history": [float(h) for h in history],
        "margin": float(embed.MARGIN),
        "embedding_spread": spread,
        "training_seconds": float(training),
    }
    return results, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=data.DATA / "prints.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--rotate-probes",
        type=float,
        default=0.0,
        help="rotate every probe by its own random angle, up to this many"
        " degrees, before anything sees it",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="a fifth of the fingers and a tenth of the training; for wiring checks only",
    )
    ap.add_argument("--out", type=Path, help="write the table to this JSON file")
    ap.add_argument(
        "--dump",
        type=Path,
        help="write per-rung ranks and score distributions to this .npz, so the"
        " figures can be redrawn without running the ladder again",
    )
    args = ap.parse_args()

    prints = data.load(args.data)
    epochs = args.epochs
    if args.quick:
        fingers = np.unique(prints.finger)[::5]
        prints = prints.subset(np.isin(prints.finger, fingers))
        epochs = max(1, args.epochs // 10)

    results, meta = run(
        prints,
        epochs=epochs,
        seed=args.seed,
        rotate_probes=args.rotate_probes,
    )
    print(bench.table(results))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "fingers": int(prints.n_fingers),
                    "gallery": int(results[0].gallery_size),
                    "epochs": epochs,
                    "seed": args.seed,
                    "rows": [r.row() for r in results],
                    "cmc": {r.name: r.cmc(20).tolist() for r in results},
                    **meta,
                },
                indent=2,
            )
            + "\n",
        )

    if args.dump:
        # Kept small enough to commit: the repo caps added files at 500 kB and
        # the full impostor set is 199x198 doubles per rung. Ranks and genuine
        # scores are short and go in whole; the impostor scores only ever feed a
        # histogram, so a fixed random sample of them is the same picture.
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        payload = {"names": np.array([r.name for r in results])}
        for i, r in enumerate(results):
            impostor = r.impostor
            if len(impostor) > IMPOSTOR_SAMPLE:
                impostor = rng.choice(impostor, IMPOSTOR_SAMPLE, replace=False)
            payload[f"ranks_{i}"] = r.ranks.astype(np.int16)
            payload[f"genuine_{i}"] = r.genuine.astype(np.float32)
            payload[f"impostor_{i}"] = impostor.astype(np.float32)
        np.savez_compressed(args.dump, **payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
