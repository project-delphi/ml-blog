"""Download the Metro Interstate Traffic Volume data (UCI 492) and write the raw file.

Run from the post directory with the series venv, naming a download directory
outside the repo (the zip is kept there, not committed):

    cd posts/two-cultures-breiman-2001
    ../../.venv-two-cultures/bin/python src/fetch_traffic.py ~/Downloads/traffic-raw

Source: <https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume>,
licence CC BY 4.0, accessed 2026-11-21. Hourly westbound I-94 volume at MN DoT ATR
station 301 (between Minneapolis and St Paul), October 2012 to September 2018, with
hourly weather and a holiday label.

Writes `data/raw/metro_traffic.csv.gz`: every source row and column, byte for byte
the CSV inside the zip, re-compressed with gzip mtime 0 so a rerun rebuilds the same
bytes. No cleaning happens here. Duplicate timestamps, the "None" holiday label, the
0 K temperatures, the 9831 mm rain hour and the ten-month stop are all left in and
handled, visibly, in `twocultures.traffic.load`.

A zip already in the download directory is reused; every file is checked against
the SHA-256 values pinned below, and a mismatch stops the run.
"""

import gzip
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import requests

if not Path("index.qmd").exists() or not Path("data/raw").is_dir():
    sys.exit("run from posts/two-cultures-breiman-2001 (it writes data/raw/)")

URL = (
    "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
)
ACCESSED = "2026-11-21"
MEMBER = "Metro_Interstate_Traffic_Volume.csv.gz"
SHA_ZIP = "b99aeabcbd6cc86f642da3a79d90883425798f58abd3b3302da2fa19dda73768"
SHA_MEMBER = "0b3679ac15173f79c6dc6c5ef8a0798d806fa5c5d7f05c84a5fa711bd1b05f07"
SHA_CSV = "749c90d720360a4215bb15345526073c079ba4cc95e3fa558796d083f85fce9e"
OUT = Path("data/raw/metro_traffic.csv.gz")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check(name, got, want):
    if got != want:
        sys.exit(f"{name}: sha256 {got}, expected {want}; the source changed, stopping")
    print(f"{name}  sha256 {got}  ok")


def main(dl: Path):
    dl.mkdir(parents=True, exist_ok=True)
    zpath = dl / "metro+interstate+traffic+volume.zip"
    if not zpath.exists():
        r = requests.get(URL, timeout=120)
        r.raise_for_status()
        zpath.write_bytes(r.content)
    zb = zpath.read_bytes()
    check(zpath.name, sha(zb), SHA_ZIP)
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        member = z.read(MEMBER)
    check(MEMBER, sha(member), SHA_MEMBER)
    csv = gzip.decompress(member)
    check("csv (decompressed)", sha(csv), SHA_CSV)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(csv)
    OUT.write_bytes(buf.getvalue())
    rows = csv.count(b"\n") - 1
    print(f"{OUT}  {len(buf.getvalue()):,} bytes  sha256 {sha(buf.getvalue())}")
    print(f"source {URL} (accessed {ACCESSED}); {rows:,} data rows")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).expanduser())
