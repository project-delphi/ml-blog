"""Tests for scripts/freeze_realign.py: the splice must be exact or refuse."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import freeze_realign as fr  # noqa: E402

FRONT = '---\ntitle: "Old title"\njupyter: eigen-blog\n---\n\n'
# Quarto stores the frontmatter verbatim, then one more blank line than the
# source has, then the body.
FRONT_REC = FRONT + "\n"
# CELL_RE ends a cell at its closing fence, so the newline after it is prose.
CELL_A = "```{python}\nimport numpy as np\nprint(np.arange(3))\n```"
CELL_B = "```{python}\n#| echo: false\nprint(1 + 1)\n```"
MERMAID = "```{mermaid}\nflowchart LR\n  A --> B\n```\n"
OUT_A = (
    "::: {#fig-a .cell execution_count=1}\n"
    "``` {.python .cell-code}\nimport numpy as np\nprint(np.arange(3))\n```\n\n"
    "::: {.cell-output .cell-output-stdout}\n```\n[0 1 2]\n```\n:::\n:::\n"
)
OUT_B = (
    "::: {.cell execution_count=2}\n"
    "::: {.cell-output .cell-output-stdout}\n```\n2\n```\n:::\n:::\n"
)
DEFAULT = ("one\n", "\ntwo\n", "\nthree\n")


def source(p1: str, p2: str, p3: str) -> str:
    """Source with two cells; p2 and p3 begin with the newline after a fence."""
    return FRONT + p1 + CELL_A + p2 + CELL_B + p3


def stored(p1: str, p2: str, p3: str) -> str:
    """The markdown the way Quarto stores it: cells replaced by output divs."""
    return FRONT_REC + p1 + OUT_A + p2 + OUT_B + p3


def frozen(src: str, p1: str, p2: str, p3: str) -> dict:
    return {
        "hash": hashlib.md5(src.encode()).hexdigest(),
        "result": {"engine": "jupyter", "markdown": stored(p1, p2, p3)},
    }


def realign_ok(old: str, new: str, parts=DEFAULT) -> dict:
    return fr.realign(old, new, frozen(old, *parts))


def test_split_separates_prose_from_compute_cells():
    prose, cells = fr.split(source("one\n", "\ntwo\n" + MERMAID, "\nthree\n"))
    assert cells == [CELL_A, CELL_B]
    assert prose == [FRONT + "one\n", "\ntwo\n" + MERMAID, "\nthree\n"]


def test_prose_edit_is_spliced_and_hash_updated():
    old = source(*DEFAULT)
    new = source("one, revised\n", "\ntwo\n", "\nthree, with a closer.\nDone.\n")
    rec = realign_ok(old, new)
    assert rec["hash"] == hashlib.md5(new.encode()).hexdigest()
    assert rec["result"]["markdown"] == stored(
        "one, revised\n", "\ntwo\n", "\nthree, with a closer.\nDone.\n"
    )


def test_frontmatter_edit_keeps_quartos_blank_line_run():
    old = source(*DEFAULT)
    new = old.replace('title: "Old title"', 'title: "New title"')
    rec = realign_ok(old, new)
    assert rec["result"]["markdown"] == stored(*DEFAULT).replace(
        'title: "Old title"', 'title: "New title"'
    )


def test_opening_prose_edit_after_frontmatter():
    old = source(*DEFAULT)
    new = source("one, again\n", "\ntwo\n", "\nthree\n")
    rec = realign_ok(old, new)
    assert rec["result"]["markdown"] == stored("one, again\n", "\ntwo\n", "\nthree\n")


def test_mermaid_edit_counts_as_prose():
    parts = ("one\n", "\ntwo\n" + MERMAID, "\nthree\n")
    old = source(*parts)
    new = old.replace("A --> B", "A --> C")
    rec = realign_ok(old, new, parts)
    assert "A --> C" in rec["result"]["markdown"]


def test_prose_added_between_adjacent_cells_lands_after_the_output():
    parts = ("one\n", "\n", "\nthree\n")
    old = source(*parts)
    new = source("one\n", "\nnow some prose\n", "\nthree\n")
    rec = realign_ok(old, new, parts)
    assert rec["result"]["markdown"] == stored(
        "one\n", "\nnow some prose\n", "\nthree\n"
    )


def test_output_containing_div_markers_inside_a_fence_is_skipped_whole():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    rec = frozen(old, *DEFAULT)
    tricky = OUT_A.replace("[0 1 2]", ":::\n::: {.x}\n[0 1 2]")
    rec["result"]["markdown"] = stored(*DEFAULT).replace(OUT_A, tricky)
    out = fr.realign(old, new, rec)
    assert out["result"]["markdown"] == stored(
        "one!\n", "\ntwo\n", "\nthree\n"
    ).replace(OUT_A, tricky)


def test_changed_cell_is_refused():
    old = source(*DEFAULT)
    new = old.replace("np.arange(3)", "np.arange(4)")
    with pytest.raises(fr.RealignError, match="executable cell 1 changed"):
        fr.realign(old, new, frozen(old, *DEFAULT))


def test_added_cell_is_refused():
    old = source(*DEFAULT)
    new = old + "\n" + CELL_A
    with pytest.raises(fr.RealignError, match="cell count changed"):
        fr.realign(old, new, frozen(old, *DEFAULT))


def test_prose_rewritten_by_quarto_is_refused():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    rec = frozen(old, *DEFAULT)
    rec["result"]["markdown"] = rec["result"]["markdown"].replace("\ntwo\n", "\nTWO\n")
    with pytest.raises(
        fr.RealignError, match="segment 2 is not in the frozen markdown"
    ):
        fr.realign(old, new, rec)


def test_rewritten_frontmatter_is_refused():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    rec = frozen(old, *DEFAULT)
    rec["result"]["markdown"] = rec["result"]["markdown"].replace(
        "eigen-blog", "python3"
    )
    with pytest.raises(fr.RealignError, match="frontmatter is not in the frozen"):
        fr.realign(old, new, rec)


def test_output_not_where_expected_is_refused():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    rec = frozen(old, *DEFAULT)
    rec["result"]["markdown"] = rec["result"]["markdown"].replace(
        "::: {#fig-a .cell execution_count=1}", "text"
    )
    with pytest.raises(fr.RealignError, match="expected cell 1's output"):
        fr.realign(old, new, rec)


def test_trailing_content_in_record_is_refused():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    rec = frozen(old, *DEFAULT)
    rec["result"]["markdown"] += "extra\n"
    with pytest.raises(fr.RealignError, match="after the last prose segment"):
        fr.realign(old, new, rec)


def test_input_record_is_not_mutated():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    rec = frozen(old, *DEFAULT)
    before = rec["hash"], rec["result"]["markdown"]
    fr.realign(old, new, rec)
    assert (rec["hash"], rec["result"]["markdown"]) == before
