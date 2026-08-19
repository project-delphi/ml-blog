"""Fetch and cache the two downloaded datasets the post uses.

Two very different licences drive two very different caching rules.

The NIST StRD *Filip* file is US-Government public domain, 4 KB, and is
committed to ``assets/`` so the post renders with no network at all.

MovieLens 100k may **not** be redistributed (GroupLens' terms), so it lands in
``data/raw/``, which the repository gitignores. Only the derived RMSE curve is
committed, by ``export_widget_data.py``.

Usage:
    .venv-svd/bin/python posts/svd-rotate-stretch-rotate/src/data.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

POST: Final[Path] = Path(__file__).resolve().parent.parent
ASSETS: Final[Path] = POST / "assets"
RAW: Final[Path] = POST / "data" / "raw"

FILIP_URL: Final[str] = (
    "https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Filip.dat"
)
MOVIELENS_URL: Final[str] = (
    "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
)

# The certified file states its own layout in its header: data on lines 61-142,
# certified parameter estimates on lines 31-41. Both are 1-indexed.
_DATA_FIRST, _DATA_LAST = 61, 142
_PARAM_FIRST, _PARAM_LAST = 31, 41

DEGREE: Final[int] = 10


def _download(url: str, dest: Path) -> Path:
    """Fetch ``url`` to ``dest`` unless it is already there.

    Args:
        url: Source URL.
        dest: Destination path; parent directories are created.

    Returns:
        ``dest``.
    """
    if dest.exists():
        return dest
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def filip_path() -> Path:
    """Return the cached NIST Filip file, downloading it if missing."""
    return _download(FILIP_URL, ASSETS / "Filip.dat")


def load_filip() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse the Filip dataset and its certified coefficients.

    Returns:
        ``(x, y, beta_certified)`` -- 82 predictors, 82 responses, and the 11
        coefficients NIST certifies to 15 digits, ordered ``B0 ... B10``.
    """
    lines = filip_path().read_text().splitlines()

    xy = np.array(
        [
            [float(tok) for tok in line.split()]
            for line in lines[_DATA_FIRST - 1 : _DATA_LAST]
            if line.strip()
        ]
    )
    y, x = xy[:, 0], xy[:, 1]

    # "B8  -0.670191154593408E-01  0.142363763154724E-01" -- estimate is field 2.
    beta = np.array(
        [
            float(lines[i - 1].split()[1])
            for i in range(_PARAM_FIRST, _PARAM_LAST + 1)
        ]
    )
    return x, y, beta


def filip_design(x: np.ndarray) -> np.ndarray:
    """Build Filip's degree-10 polynomial design matrix.

    This is the matrix the post calls ill-conditioned: 82 rows, 11 columns,
    column *j* being ``x**j``. Nothing is rescaled, because the point is what
    the problem as-posed does to a solver.

    Args:
        x: The 82 predictor values.

    Returns:
        An ``(82, 11)`` design matrix.
    """
    return np.vander(x, DEGREE + 1, increasing=True)


def movielens_ratings() -> pd.DataFrame:
    """Download (once) and return the MovieLens 100k ratings.

    Returns:
        A frame with ``user``, ``item``, ``rating`` and ``timestamp`` columns,
        100,000 rows.
    """
    zip_path = _download(MOVIELENS_URL, RAW / "ml-100k.zip")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("ml-100k/u.data") as fh:
            return pd.read_csv(
                io.TextIOWrapper(fh, "utf-8"),
                sep="\t",
                names=["user", "item", "rating", "timestamp"],
            )


def main() -> int:
    """Report what is cached, fetching anything missing."""
    x, y, beta = load_filip()
    a = filip_design(x)
    print(f"Filip      {a.shape[0]} obs, design {a.shape}, cond {np.linalg.cond(a):.3e}")
    print(f"           certified B0 = {beta[0]:.11f}")
    ratings = movielens_ratings()
    print(
        f"MovieLens  {len(ratings):,} ratings, "
        f"{ratings.user.nunique()} users, {ratings.item.nunique()} items"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
