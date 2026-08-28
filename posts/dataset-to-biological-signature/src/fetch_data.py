#!/usr/bin/env python3
"""Download and cache the two Golub sources this post reads.

Run by hand, not by Quarto: the render must work offline, so the outputs of
this script are committed under ../data/. Re-run only to refresh the cache.

Two sources, because neither one alone is enough:

1. Efron & Hastie's CASI copy (leukemia_big.csv). This is the matrix the post
   models on. It carries no gene identifiers and, per Hastie's own data page,
   "the genes in the big dataset have been transformed, with the exact
   transformation used lost in time."
2. The original Whitehead/MIT files. These carry the Affymetrix probe
   accessions and per-patient sample IDs, which is what lets the post name the
   genes it recovers and prove which CASI columns are Golub's held-out cohort.
   The Whitehead URL in the 1999 paper is long dead; this mirror is the
   commonly used copy of those files.
"""

import gzip
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
MIRROR = (
    "https://raw.githubusercontent.com/dharsandip/"
    "Classification_of_Cancer_by_Gene_Expression_Dataset/master/"
)

SOURCES = {
    "leukemia_big.csv.gz": "http://hastie.su.domains/CASI_files/DATA/leukemia_big.csv",
    "golub_train.csv.gz": MIRROR + "data_set_ALL_AML_train.csv",
    "golub_independent.csv.gz": MIRROR + "data_set_ALL_AML_independent.csv",
    "golub_labels.csv": MIRROR + "actual.csv",
}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        dest = DATA / name
        with urllib.request.urlopen(url, timeout=120) as resp:
            raw = resp.read()
        if name.endswith(".gz"):
            # mtime=0 so re-running produces a byte-identical file and does not
            # show up as a spurious diff.
            with gzip.GzipFile(dest, "wb", compresslevel=9, mtime=0) as fh:
                fh.write(raw)
        else:
            dest.write_bytes(raw)
        print(f"{name:28s} {len(raw):>9,} bytes -> {dest.stat().st_size:>9,} on disk")


if __name__ == "__main__":
    main()
