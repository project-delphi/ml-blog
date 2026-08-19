"""Write the committed JSON the browser widgets read.

Design rule, inherited from ``posts/bayesian-bootstrap``: ship *factors*, never
reconstructions. The rank slider has 100 positions; shipping 100 rendered images
would be 14 MB, while shipping the 100 singular triplets that generate them is
under half a megabyte and lets the browser rebuild any rank on demand.

The singular vectors are orthonormal, so every entry sits in [-1, 1] and a
single global scale quantises them to int16 with a relative error near 3e-5 --
far below what a display can show.

Usage:
    .venv-svd/bin/python posts/svd-rotate-stretch-rotate/src/export_widget_data.py
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image

import imagery as im
import movielens as ml

OUT: Final[Path] = Path(__file__).resolve().parent.parent / "widget-data"

# int16 full scale. Vectors are orthonormal so nothing reaches 1.0 in practice.
QUANT: Final[int] = 32767

BUDGET_BYTES: Final[int] = 1_024_000


def _b64_int16(values: np.ndarray) -> str:
    """Quantise a unit-norm array to int16 and base64-encode it.

    Args:
        values: Array with entries in ``[-1, 1]``.

    Returns:
        Base64 of the little-endian int16 buffer.
    """
    quantised = np.clip(np.round(values * QUANT), -QUANT, QUANT).astype("<i2")
    return base64.b64encode(quantised.tobytes()).decode("ascii")


def _png_data_uri(image: np.ndarray) -> tuple[str, int]:
    """Encode the image as a PNG data URI.

    The byte count comes back too: it is the post's honest reference point for
    what a real lossless codec achieves on the same picture.

    Args:
        image: Grayscale array on a 0-255 scale.

    Returns:
        ``(data_uri, png_byte_count)``.
    """
    buf = io.BytesIO()
    Image.fromarray(np.round(image).astype("uint8")).save(
        buf, format="PNG", optimize=True
    )
    raw = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii"), len(raw)


def export_image() -> dict[str, Any]:
    """Pack the photograph's leading singular triplets and their price list."""
    image = im.load_image()
    m, n = image.shape
    u, s, vt = im.svd(image)
    k_max = im.MAX_RANK

    ranks = np.arange(1, k_max + 1)
    table = im.quality_table(image, u, s, vt, ranks)
    uri, png_bytes = _png_data_uri(image)

    return {
        "m": int(m),
        "n": int(n),
        "maxRank": int(k_max),
        "quant": QUANT,
        # Column-major by rank: entry (i, j) of U lives at u[j * m + i].
        "u": _b64_int16(np.asfortranarray(u[:, :k_max]).ravel(order="F")),
        "v": _b64_int16(np.asfortranarray(vt[:k_max].T).ravel(order="F")),
        "sigma": [round(float(x), 4) for x in s[:k_max]],
        "psnr": [round(float(x), 3) for x in table["psnr"]],
        "energy": [round(float(x), 6) for x in table["energy"]],
        "fracF32": [round(float(x), 5) for x in table["frac_f32"]],
        "fracI16": [round(float(x), 5) for x in table["frac_i16"]],
        "rawBytes": im.original_bytes(m, n),
        "pngBytes": png_bytes,
        "elbow": im.elbow(s),
        "original": uri,
    }


def export_movielens() -> dict[str, Any]:
    """Pack the held-out RMSE curve. Tiny -- it is 19 numbers and a header."""
    out = ml.rmse_by_rank()
    out["rmse"] = [round(float(x), 5) for x in out["rmse"]]
    out["baseline_rmse"] = round(float(out["baseline_rmse"]), 5)
    out["best_rmse"] = round(float(out["best_rmse"]), 5)
    out["density"] = round(float(out["density"]), 5)
    return out


def main() -> int:
    """Write every payload and enforce the size budget."""
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {"image": export_image(), "movielens": export_movielens()}

    total = 0
    for name, payload in payloads.items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        size = path.stat().st_size
        total += size
        print(f"  {name:12s} {size / 1024:8.1f} KB")

    print(f"  {'total':12s} {total / 1024:8.1f} KB  (budget {BUDGET_BYTES / 1024:.0f} KB)")
    if total > BUDGET_BYTES:
        raise ValueError(f"widget data is {total / 1024:.0f} KB, over budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
