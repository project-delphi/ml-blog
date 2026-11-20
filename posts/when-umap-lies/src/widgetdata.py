"""Write precomputed data for the When UMAP Lies widgets as `widget-data/<name>.js`.

The widgets are static files on GitHub Pages and must also work from ``file://``,
where a ``fetch()`` of JSON fails, so each data file is a script that merges its
payload into one global::

    window.WK_DATA = Object.assign(window.WK_DATA || {}, {"<key>": {...}});

and a widget reads it with ``WK.data("<key>")``. Several data files can load on one
page without overwriting each other.

Embedding format ("uk-embed-1"), shared with ``UK.decode`` in ``../umapkit.js``
-------------------------------------------------------------------------------

An n-point 2-D embedding is stored as a JSON object::

    {"n": n, "bits": 8, "cx": cx, "cy": cy, "scale": s, "xy": "<base64>"}

- The centre ``(cx, cy)`` is the midpoint of the embedding's bounding box.
- One scale serves both axes, so the aspect ratio survives quantisation:
  ``s = max(width, height) / (2**bits - 1)``.
- Each coordinate is quantised as ``q = round((v - c) / s + (2**bits - 1) / 2)``,
  an unsigned integer in ``0 .. 2**bits - 1``. The longer axis spans the full range;
  the shorter one sits centred inside it.
- ``xy`` is base64 of the interleaved integers ``x0, y0, x1, y1, ...``: one byte each
  when ``bits`` is 8, two little-endian bytes each when it is 16.
- Decoding is ``v = (q - (2**bits - 1) / 2) * s + c``, within ``s / 2`` of the
  original on each axis.

Category labels are base64 of one uint8 per point (``encode_labels``), so at most 256
categories. Floats elsewhere in a payload are rounded to a few significant digits by
``compact`` before writing.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np

FORMAT = "uk-embed-1"


def encode_embedding(Y: np.ndarray, bits: int = 8) -> dict:
    """Quantise an (n, 2) embedding to the uk-embed-1 format described above."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError(f"expected an (n, 2) array, got {Y.shape}")
    if not np.isfinite(Y).all():
        raise ValueError("embedding has non-finite coordinates")
    if bits not in (8, 16):
        raise ValueError("bits must be 8 or 16")
    top = 2**bits - 1
    lo, hi = Y.min(0), Y.max(0)
    # Round the centre and scale first and quantise with the rounded values, so the
    # numbers written are exactly the ones the decoder uses.
    cx, cy = (_sig(c, 8) for c in (lo + hi) / 2)
    span = float((hi - lo).max())
    # A hair over span / top keeps the far edge inside the range after rounding.
    scale = _sig(span / top * (1 + 1e-7), 8) if span > 0 else 1
    q = np.rint((Y - [cx, cy]) / scale + top / 2)
    q = np.clip(q, 0, top).astype("<u1" if bits == 8 else "<u2")
    return {
        "n": int(Y.shape[0]),
        "bits": bits,
        "cx": cx,
        "cy": cy,
        "scale": scale,
        "xy": base64.b64encode(q.reshape(-1).tobytes()).decode("ascii"),
    }


def encode_labels(labels: np.ndarray) -> str:
    """Base64 of one uint8 per point."""
    labels = np.asarray(labels)
    if labels.min() < 0 or labels.max() > 255:
        raise ValueError("labels must lie in 0..255")
    return base64.b64encode(labels.astype(np.uint8).tobytes()).decode("ascii")


def encode_values(v: np.ndarray, bits: int = 8) -> dict:
    """A per-point value (for a continuous colouring) as {n, bits, lo, hi, q}.

    ``q`` is base64 of ``round((v - lo) / (hi - lo) * (2**bits - 1))``, one byte per
    point at 8 bits, two little-endian at 16; ``UK.decodeValues`` inverts it.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    if not np.isfinite(v).all():
        raise ValueError("values must be finite")
    if bits not in (8, 16):
        raise ValueError("bits must be 8 or 16")
    top = 2**bits - 1
    lo, hi = _sig(v.min(), 8), _sig(v.max(), 8)
    span = (hi - lo) or 1
    q = np.clip(np.rint((v - lo) / span * top), 0, top).astype(
        "<u1" if bits == 8 else "<u2"
    )
    return {
        "n": int(v.size),
        "bits": bits,
        "lo": lo,
        "hi": hi,
        "q": base64.b64encode(q.tobytes()).decode("ascii"),
    }


def _sig(x: float, digits: int) -> float | int | None:
    """Round to `digits` significant digits; NaN becomes None (JSON null)."""
    x = float(x)
    if math.isnan(x):
        return None
    if x == 0 or math.isinf(x):
        return x
    v = round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))
    return int(v) if v == int(v) and abs(v) < 1e15 else v


def compact(obj, digits: int = 4):
    """Recursively round every float (and numpy number/array) to `digits` sig. digits.

    Encoded embeddings carry their own precision and are left alone.
    """
    if isinstance(obj, dict):
        if ("xy" in obj and "scale" in obj) or ("q" in obj and "lo" in obj):
            return obj
        return {str(k): compact(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [compact(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return compact(obj.tolist(), digits)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _sig(obj, digits)
    return obj


def write_widget_data(path: Path, key: str, payload: dict, digits: int = 4) -> int:
    """Write `payload` under `key` as a WK_DATA script; returns the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {key: compact(payload, digits)},
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    text = (
        f"/* Generated by posts/when-umap-lies/src/export_widget_data.py; do not edit. */\n"
        f"window.WK_DATA = Object.assign(window.WK_DATA || {{}}, {body});\n"
    )
    path.write_text(text, encoding="utf-8")
    return path.stat().st_size
