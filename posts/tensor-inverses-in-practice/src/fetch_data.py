r"""Fetch and cache the two real datasets this post reads at render time.

Writes into posts/tensor-inverses-in-practice/data/, which .gitignore carves
out of the repo-wide `data/` rule so the cache is committed and a clone can
rebuild the post without network access.

Usage:
    .venv-tensor-factorizations/bin/python \\
        posts/tensor-inverses-in-practice/src/fetch_data.py
"""

from __future__ import annotations

import json
import urllib.parse
from datetime import date, datetime
from pathlib import Path

import numpy as np
import requests
from scipy.io import loadmat

OUT = Path(__file__).resolve().parent.parent / "data"

# Pavia University, ROSIS-03, 610 x 340 x 103. The canonical host
# (www.ehu.eus/ccwintco) refuses programmatic requests, so this pulls the same
# file from an ungated HuggingFace mirror. Provenance is unchanged: the scene
# was flown over Pavia by the DLR ROSIS-03 sensor and released for research by
# Prof. Paolo Gamba (University of Pavia).
PAVIA_URL = "https://huggingface.co/datasets/danaroth/pavia/resolve/main/PaviaU.mat"
PAVIA_KEY = "paviaU"
CROP_ROW, CROP_COL, CROP_SIZE = 150, 100, 128

# Chicago Crimes 2001-Present, City of Chicago open data portal (Socrata).
CHICAGO_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
CHICAGO_YEAR = 2023
CHICAGO_TYPES = [
    "THEFT",
    "BATTERY",
    "CRIMINAL DAMAGE",
    "ASSAULT",
    "MOTOR VEHICLE THEFT",
    "DECEPTIVE PRACTICE",
    "ROBBERY",
    "NARCOTICS",
]
CHICAGO_AREAS = 24  # community areas 1..24, the north and near-north side
PAGE = 50_000


def _download_resumable(url: str, dest: Path, attempts: int = 8) -> Path:
    """Stream a large file, resuming with a Range request if the socket drops."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        print(f"GET   {url} (attempt {attempt}, have {have / 1e6:.1f} MB)")
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if have and resp.status_code == 200:
                    have, mode = 0, "wb"  # server ignored Range; start over
                else:
                    resp.raise_for_status()
                    mode = "ab" if have else "wb"
                total = have + int(resp.headers.get("Content-Length", 0))
                with dest.open(mode) as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            if total and dest.stat().st_size >= total:
                print(f"      {dest.stat().st_size / 1e6:.1f} MB complete")
                return dest
        except requests.RequestException as err:
            print(f"      dropped: {err}")
    raise RuntimeError(f"could not download {url} after {attempts} attempts")


def fetch_pavia() -> Path:
    """Download the cube once, crop it, and keep only the crop."""
    out = OUT / "paviaU_crop.npz"
    if out.exists():
        print(f"kept  {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return out
    raw = OUT / "PaviaU.mat"  # gitignored scratch; only the crop is committed
    _download_resumable(PAVIA_URL, raw)
    cube = loadmat(raw)[PAVIA_KEY]
    print(f"      full cube {cube.shape} {cube.dtype}")
    r, c, n = CROP_ROW, CROP_COL, CROP_SIZE
    crop = cube[r : r + n, c : c + n, :].astype(np.float32)
    crop /= crop.max()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        cube=crop,
        crop_origin=np.array([r, c]),
        full_shape=np.array(cube.shape),
        source=np.array(PAVIA_URL),
        retrieved=np.array(date.today().isoformat()),
    )
    raw.unlink()
    print(f"wrote {out} {crop.shape} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def _socrata_page(offset: int) -> list[dict]:
    # community_area is a text column on this dataset, so a range comparison
    # is lexicographic: '3' <= '24' is false and areas 3-9 vanish. Enumerate
    # the values instead.
    areas = ",".join(f"'{i}'" for i in range(1, CHICAGO_AREAS + 1))
    where = (
        f"date >= '{CHICAGO_YEAR}-01-01T00:00:00'"
        f" AND date < '{CHICAGO_YEAR + 1}-01-01T00:00:00'"
        f" AND community_area IN ({areas})"
    )
    query = {
        "$select": "date,primary_type,community_area",
        "$where": where,
        "$limit": PAGE,
        "$offset": offset,
        "$order": "date",
    }
    url = f"{CHICAGO_URL}?{urllib.parse.urlencode(query)}"
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    return resp.json()


def fetch_chicago() -> Path:
    """Bin one year of reported incidents into weeks x area x offence type."""
    out = OUT / "chicago_counts.npz"
    if out.exists():
        print(f"kept  {out} ({out.stat().st_size / 1e3:.0f} kB)")
        return out
    type_index = {t: i for i, t in enumerate(CHICAGO_TYPES)}
    counts = np.zeros((52, CHICAGO_AREAS, len(CHICAGO_TYPES)), dtype=np.int32)
    offset, total = 0, 0
    while True:
        rows = _socrata_page(offset)
        if not rows:
            break
        print(f"GET   offset {offset:>7} -> {len(rows)} rows")
        for row in rows:
            kind = type_index.get(str(row.get("primary_type", "")))
            area = row.get("community_area")
            if kind is None or not area:
                continue
            stamp = datetime.fromisoformat(row["date"])
            week = min((stamp.timetuple().tm_yday - 1) // 7, 51)
            counts[week, int(area) - 1, kind] += 1
            total += 1
        offset += PAGE
        if len(rows) < PAGE:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        counts=counts,
        types=np.array(CHICAGO_TYPES),
        areas=np.arange(1, CHICAGO_AREAS + 1),
        year=np.array(CHICAGO_YEAR),
        n_incidents=np.array(total),
        source=np.array(CHICAGO_URL),
        retrieved=np.array(date.today().isoformat()),
    )
    print(f"wrote {out} {counts.shape}, {total} incidents binned")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fetch_pavia()
    fetch_chicago()
    manifest = {
        "pavia": {
            "url": PAVIA_URL,
            "crop": [CROP_ROW, CROP_COL, CROP_SIZE],
            "note": "ROSIS-03 over Pavia, Italy; released by P. Gamba, U. Pavia.",
        },
        "chicago": {
            "url": CHICAGO_URL,
            "year": CHICAGO_YEAR,
            "types": CHICAGO_TYPES,
            "areas": CHICAGO_AREAS,
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
