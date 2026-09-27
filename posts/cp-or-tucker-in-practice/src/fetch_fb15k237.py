r"""Fetch FB15k-237 once and keep only what the post reads: per-relation counts.

The three split files are 24 MB together, too large to commit for a table of 237
rows. This script downloads them, measures how often each relation's reversed
pair (o, s) also appears in training, and writes

- data/fb15k237_relations.csv: relation, training triples, reversed fraction
- data/fb15k237_manifest.json: split sizes, entity and relation counts, hashes

The copy is the one distributed with TuckER (Balazevic et al. 2019), which is the
Toutanova & Chen (2015) split. Some lines end in CRLF, so every field is stripped.

Usage:
    .venv-cp-tucker/bin/python posts/cp-or-tucker-in-practice/src/fetch_fb15k237.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.request import urlopen

OUT = Path(__file__).resolve().parent.parent / "data"
BASE = "https://raw.githubusercontent.com/ibalazevic/TuckER/master/data/FB15k-237"
SPLITS = ["train", "valid", "test"]


def fetch(split: str) -> tuple[list[tuple[str, str, str]], str]:
    raw = urlopen(f"{BASE}/{split}.txt", timeout=60).read()
    triples = []
    for line in raw.decode().splitlines():
        s, r, o = (field.strip() for field in line.split("\t"))
        triples.append((s, r, o))
    return triples, hashlib.sha256(raw).hexdigest()


def main() -> None:
    data, hashes = {}, {}
    for split in SPLITS:
        data[split], hashes[split] = fetch(split)

    everything = [t for split in SPLITS for t in data[split]]
    entities = {e for s, _, o in everything for e in (s, o)}
    relations = {r for _, r, _ in everything}

    pairs = defaultdict(set)
    for s, r, o in data["train"]:
        pairs[r].add((s, o))

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "fb15k237_relations.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["relation", "train_pairs", "reversed_fraction", "two_role_fraction"]
        )
        for r in sorted(pairs):
            seen = pairs[r]
            reversed_ = sum((o, s) in seen for s, o in seen if s != o)
            # A pair is exposed to a symmetric score when its object is also a
            # subject of r, or its subject also an object: only then does some
            # reversed pair compete with a true answer in a ranking query.
            subjects = {s for s, _ in seen}
            objects = {o for _, o in seen}
            two_role = sum(o in subjects or s in objects for s, o in seen)
            writer.writerow(
                [
                    r,
                    len(seen),
                    f"{reversed_ / len(seen):.4f}",
                    f"{two_role / len(seen):.4f}",
                ]
            )

    manifest = {
        "source": f"{BASE}/{{train,valid,test}}.txt",
        "split": "Toutanova & Chen (2015) FB15k-237, as distributed with TuckER",
        "entities": len(entities),
        "relations": len(relations),
        "triples": {split: len(data[split]) for split in SPLITS},
        "sha256": hashes,
        "fetched": date.today().isoformat(),
    }
    (OUT / "fb15k237_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
