"""The numbers each part's reproduction-status rows quote, shared with Part 5.

Parts 1-4 each end with a hidden cell that calls `export(...)`, which writes
`cache/status.json` in that post's directory every time the post executes.
Part 5's series table reads the four files with `load(slug)`, so it never names
another part's versioned cache file or copies one of its results by hand. A
part that re-runs an experiment refreshes its own file; Part 5 then needs a
re-render of its own, since Quarto's freeze hashes only its `index.qmd`
(ENVIRONMENTS.md).
"""

import json
from pathlib import Path

FILE = Path("cache") / "status.json"


def _plain(v):
    """Convert a NumPy scalar to a Python number, so json can write it."""
    return v.item() if hasattr(v, "item") else v


def export(**values) -> None:
    """Write this post's status values to cache/status.json."""
    FILE.parent.mkdir(exist_ok=True)
    out = {k: _plain(v) for k, v in values.items()}
    FILE.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


def load(slug: str) -> dict:
    """Read a sibling post's status values (called from another post's directory)."""
    return json.loads((Path("..") / slug / FILE).read_text())
