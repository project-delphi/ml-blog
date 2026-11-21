"""Round-trip the widget embedding format, with a decoder written from its spec."""

import base64
import json

import numpy as np
import pytest
from widgetdata import (
    compact,
    encode_embedding,
    encode_labels,
    encode_values,
    write_widget_data,
)


def decode(e: dict) -> np.ndarray:
    """The uk-embed-1 decoder, as UK.decode in umapkit.js implements it."""
    raw = base64.b64decode(e["xy"])
    q = np.frombuffer(raw, dtype="<u1" if e["bits"] == 8 else "<u2").astype(float)
    top = 2 ** e["bits"] - 1
    q = q.reshape(e["n"], 2)
    return (q - top / 2) * e["scale"] + np.array([e["cx"], e["cy"]])


@pytest.mark.parametrize("bits", [8, 16])
def test_round_trip_within_one_step(bits):
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(2638, 2)) * [3.0, 0.5] + [10.0, -4.0]
    e = encode_embedding(Y, bits)
    Z = decode(json.loads(json.dumps(e)))
    assert Z.shape == Y.shape
    assert np.abs(Z - Y).max() <= e["scale"] * 0.5 + 1e-9


def test_aspect_ratio_is_kept():
    Y = np.c_[np.linspace(0, 10, 50), np.linspace(0, 1, 50)]
    e = encode_embedding(Y)
    q = np.frombuffer(base64.b64decode(e["xy"]), dtype=np.uint8).reshape(-1, 2)
    assert q[:, 0].min() == 0 and q[:, 0].max() == 255  # the long axis fills the range
    assert q[:, 1].max() - q[:, 1].min() in (25, 26)  # the short one keeps its tenth


def test_degenerate_embedding():
    e = encode_embedding(np.zeros((5, 2)))
    assert np.allclose(decode(e), 0.0, atol=e["scale"])


def test_labels_and_compact(tmp_path):
    labels = np.array([0, 7, 3, 3])
    assert list(base64.b64decode(encode_labels(labels))) == [0, 7, 3, 3]
    assert compact({"a": 0.123456, "b": [np.float32(2.0), np.nan]}) == {
        "a": 0.1235,
        "b": [2, None],
    }
    size = write_widget_data(tmp_path / "x.js", "k", {"v": 1.23456})
    text = (tmp_path / "x.js").read_text()
    assert size == len(text.encode())
    assert (
        'window.WK_DATA = Object.assign(window.WK_DATA || {}, {"k":{"v":1.235}});'
        in text
    )


def test_values_round_trip():
    """UK.decodeValues: v = lo + (hi - lo) * q / (2**bits - 1)."""
    v = np.random.default_rng(1).uniform(-3, 40, 1000)
    e = encode_values(v)
    q = np.frombuffer(base64.b64decode(e["q"]), dtype=np.uint8).astype(float)
    back = e["lo"] + (e["hi"] - e["lo"]) * q / 255
    assert np.abs(back - v).max() <= (e["hi"] - e["lo"]) / 255 / 2 + 1e-6
