"""Loading the print cache and splitting it into fingers to learn from and fingers to search.

The cache holds two impressions of each finger, taken from the two sets NIST
scanned. That structure is what makes an identification test possible: put one
impression of every test finger in the gallery, use the other as the probe, and
ask which gallery entry comes back first.

The split is by *finger*, not by image. A learned matcher that trained on one
impression of a finger and was then asked to recognise the other would be scored
on fingers it had already seen, which is not the question -- a phone enrols a
finger the model was never trained on. So the training fingers and the test
fingers are disjoint sets, and every rung is scored on the test half only, whether
it learned anything or not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path("data")


@dataclass(frozen=True)
class Prints:
    """The standardised scans and the labels that go with them."""

    images: np.ndarray  # (N, S, S) uint8
    finger: np.ndarray  # (N,) which finger, 0..F-1
    impression: np.ndarray  # (N,) 0 for NIST's set A, 1 for set B
    subject: np.ndarray  # (N,) which person
    position: np.ndarray  # (N,) ANSI/NIST finger position code
    quality: np.ndarray  # (N,) NIST's own quality grade, 0..5
    name: np.ndarray  # (N,) original filename

    def __len__(self) -> int:
        return len(self.images)

    @property
    def size(self) -> int:
        return self.images.shape[1]

    @property
    def n_fingers(self) -> int:
        return len(np.unique(self.finger))

    def subset(self, mask: np.ndarray) -> Prints:
        return Prints(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))


POSITION_NAMES = {
    2: "right index",
    3: "right middle",
    4: "right ring",
    5: "right little",
    7: "left index",
    8: "left middle",
    9: "left ring",
    10: "left little",
}


def load(path: Path | str = DATA / "prints.npz") -> Prints:
    """Read the cache written by src/fetch_data.py."""
    d = np.load(path)
    return Prints(
        images=d["images"],
        finger=d["finger"],
        impression=d["impression"],
        subject=d["subject"],
        position=d["position"],
        quality=d["quality"],
        name=d["name"],
    )


def manifest(path: Path | str = DATA / "manifest.json") -> dict:
    """Provenance for the cache: where it came from, when, and under what licence."""
    return json.loads(Path(path).read_text())


def split(prints: Prints, train_fraction: float = 0.5, seed: int = 7):
    """Divide the fingers in two. Returns (train, test) as `Prints`.

    Fingers are shuffled before splitting because the cache is ordered by subject
    and finger position, and cutting that order in half would put whole hands --
    and every left little finger -- on one side.
    """
    fingers = np.unique(prints.finger)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(fingers)
    cut = int(round(train_fraction * len(fingers)))
    train_ids, test_ids = shuffled[:cut], shuffled[cut:]
    return (
        prints.subset(np.isin(prints.finger, train_ids)),
        prints.subset(np.isin(prints.finger, test_ids)),
    )


def gallery_probe(prints: Prints):
    """Split one set of prints into the enrolled impression and the searched one.

    Returns `(gallery, probe, truth)`, where `truth[i]` is the gallery row holding
    the mate of probe `i`.
    """
    gallery = prints.subset(prints.impression == 0)
    probe = prints.subset(prints.impression == 1)
    order = np.argsort(gallery.finger)
    gallery = gallery.subset(order)
    truth = np.searchsorted(gallery.finger, probe.finger)
    assert np.array_equal(
        gallery.finger[truth], probe.finger
    ), "gallery is missing a mate"
    return gallery, probe, truth
