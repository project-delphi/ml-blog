"""Tests for scripts/freeze_realign.py: the splice must be exact or refuse."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import freeze_realign as fr  # noqa: E402

FRONT = '---\ntitle: "Old title"\njupyter: eigen-blog\n---\n\n'
# CELL_RE ends a cell at its closing fence, so the newline after it is prose.
CELL_A = "```{python}\nimport numpy as np\nprint(np.arange(3))\n```"
CELL_B = "```{python}\n#| echo: false\nprint(1 + 1)\n```"
CELL_LABELLED = "```{python}\n#| label: fig-b\nprint(1 + 1)\n```"
CELL_HIDDEN = "```{python}\n#| include: false\nimport numpy as np\n```"
CELL_QUIET = "```{python}\n#| label: setup\n#| echo: false\nimport numpy as np\n```"
CELL_NOEVAL = "```{python}\n#| eval: false\nprint('never')\n```"
CELL_ASIS = (
    "```{python}\n#| echo: false\n#| output: asis\nprint('<script>1</script>')\n```"
)
MERMAID = "```{mermaid}\nflowchart LR\n  A --> B\n```\n"
OUT_A = (
    "::: {.cell execution_count=1}\n"
    "``` {.python .cell-code}\nimport numpy as np\nprint(np.arange(3))\n```\n\n"
    "::: {.cell-output .cell-output-stdout}\n```\n[0 1 2]\n```\n:::\n:::\n"
)
OUT_B = (
    "::: {.cell execution_count=2}\n"
    "::: {.cell-output .cell-output-stdout}\n```\n2\n```\n:::\n:::\n"
)
OUT_B_LABELLED = OUT_B.replace("::: {.cell ", "::: {#cell-fig-b .cell ")
DEFAULT = ("one\n", "\ntwo\n", "\nthree\n")


def norm(text: str) -> str:
    """Collapse newline runs: Quarto's padding is not part of the content."""
    return re.sub(r"\n+", "\n", text)


def source(p1: str, p2: str, p3: str, cells=(CELL_A, CELL_B)) -> str:
    """Source with two cells; p2 and p3 begin with the newline after a fence."""
    return FRONT + p1 + cells[0] + p2 + cells[1] + p3


def stored(p1: str, p2: str, p3: str, outs=(OUT_A, OUT_B)) -> str:
    """The markdown the way Quarto stores it: cells replaced by output divs,
    and a blank line added after the frontmatter and before every div."""
    return FRONT + "\n" + p1 + "\n" + outs[0] + p2 + "\n" + outs[1] + p3


def frozen(src: str, markdown: str) -> dict:
    return {
        "hash": hashlib.md5(src.encode()).hexdigest(),
        "result": {"engine": "jupyter", "markdown": markdown},
    }


def realign_ok(old: str, new: str, parts=DEFAULT, outs=(OUT_A, OUT_B)) -> dict:
    return fr.realign(old, new, frozen(old, stored(*parts, outs=outs)))


def test_split_separates_prose_from_compute_cells():
    prose, cells = fr.split(source("one\n", "\ntwo\n" + MERMAID, "\nthree\n"))
    assert cells == [CELL_A, CELL_B]
    assert prose == [FRONT + "one\n", "\ntwo\n" + MERMAID, "\nthree\n"]


def test_identity_realign_is_byte_for_byte():
    old = source(*DEFAULT)
    rec = realign_ok(old, old)
    assert rec["result"]["markdown"] == stored(*DEFAULT)


def test_prose_edit_is_spliced_and_hash_updated():
    old = source(*DEFAULT)
    new = source("one, revised\n", "\ntwo\n", "\nthree, with a closer.\nDone.\n")
    rec = realign_ok(old, new)
    assert rec["hash"] == hashlib.md5(new.encode()).hexdigest()
    assert norm(rec["result"]["markdown"]) == norm(
        stored("one, revised\n", "\ntwo\n", "\nthree, with a closer.\nDone.\n")
    )
    # Unchanged segments keep the record's own bytes, padding included.
    assert "\n" + OUT_B + "\nthree" in rec["result"]["markdown"]


def test_frontmatter_edit_counts_as_prose():
    old = source(*DEFAULT)
    new = old.replace('title: "Old title"', 'title: "New title"')
    rec = realign_ok(old, new)
    assert norm(rec["result"]["markdown"]) == norm(
        stored(*DEFAULT).replace('title: "Old title"', 'title: "New title"')
    )


def test_changed_segment_gets_fences_padded():
    parts = ("one\n" + MERMAID + "after\n", "\ntwo\n", "\nthree\n")
    old = source(*parts)
    new = old.replace("after\n", "after, edited\n")
    rec = realign_ok(old, new, parts)
    assert "one\n\n" + MERMAID + "\nafter, edited\n" in rec["result"]["markdown"]


def test_padded_fence_in_record_is_tolerated():
    parts = ("one\n" + MERMAID + "after\n", "\ntwo\n", "\nthree\n")
    old = source(*parts)
    new = old.replace("three", "three, edited")
    padded = stored(*parts).replace("one\n" + MERMAID, "one\n\n" + MERMAID + "\n")
    rec = fr.realign(old, new, frozen(old, padded))
    assert rec["result"]["markdown"].startswith(
        FRONT + "\n" + "one\n\n" + MERMAID + "\nafter\n"
    )
    assert rec["result"]["markdown"].endswith("three, edited\n")


def test_changed_segment_keeps_a_blank_line_before_the_next_div():
    # Source with no blank line between the heading and the fence.
    parts = ("### Heading\n", "\ntwo\n", "\nthree\n")
    old = source(*parts)
    new = old.replace("### Heading", "### New heading")
    rec = realign_ok(old, new, parts)
    assert "### New heading\n\n" + OUT_A in rec["result"]["markdown"]


def test_prose_added_between_adjacent_cells_lands_after_the_output():
    parts = ("one\n", "\n", "\nthree\n")
    old = source(*parts)
    new = source("one\n", "\nnow some prose\n", "\nthree\n")
    rec = realign_ok(old, new, parts)
    assert norm(rec["result"]["markdown"]) == norm(
        stored("one\n", "\nnow some prose\n", "\nthree\n")
    )


def test_labelled_cell_matches_its_prefixed_div_id():
    old = source(*DEFAULT, cells=(CELL_A, CELL_LABELLED))
    new = old.replace("three", "three, edited")
    rec = fr.realign(
        old, new, frozen(old, stored(*DEFAULT, outs=(OUT_A, OUT_B_LABELLED)))
    )
    assert rec["result"]["markdown"].endswith(OUT_B_LABELLED + "\nthree, edited\n")


def test_hidden_cell_before_labelled_cell_does_not_claim_its_div():
    # A quiet setup cell followed directly by a labelled cell: the record has
    # one div, and it belongs to the second cell.
    old = source("one\n", "\n", "\nthree\n", cells=(CELL_QUIET, CELL_LABELLED))
    new = old.replace("three", "three, edited")
    markdown = FRONT + "\n" + "one\n" + "\n" + OUT_B_LABELLED + "\nthree\n"
    rec = fr.realign(old, new, frozen(old, markdown))
    assert rec["result"]["markdown"].endswith(OUT_B_LABELLED + "\nthree, edited\n")


def test_hidden_cell_leaves_no_trace_and_newlines_collapse():
    old = source(*DEFAULT, cells=(CELL_HIDDEN, CELL_B))
    new = old.replace("three", "three, edited")
    # Quarto folds the blank lines around the vanished cell into one run.
    markdown = FRONT + "\n" + "one\n\n" + "two\n" + "\n" + OUT_B + "\nthree\n"
    rec = fr.realign(old, new, frozen(old, markdown))
    assert rec["result"]["markdown"].endswith(OUT_B + "\nthree, edited\n")


def test_quiet_setup_cell_leaves_no_trace_when_next_prose_is_right_there():
    old = source(*DEFAULT, cells=(CELL_QUIET, CELL_B))
    new = old.replace("three", "three, edited")
    markdown = FRONT + "\n" + "one\n\n" + "two\n" + "\n" + OUT_B + "\nthree\n"
    rec = fr.realign(old, new, frozen(old, markdown))
    assert rec["result"]["markdown"].endswith(OUT_B + "\nthree, edited\n")


def test_no_eval_cell_is_stored_as_its_fence():
    old = source(*DEFAULT, cells=(CELL_NOEVAL, CELL_B))
    new = old.replace("three", "three, edited")
    # The div-line count is the cell's index, so an unrun cell still counts.
    markdown = (
        FRONT
        + "\n"
        + "one\n\n"
        + CELL_NOEVAL
        + "\n\ntwo\n"
        + "\n"
        + OUT_B
        + "\nthree\n"
    )
    rec = fr.realign(old, new, frozen(old, markdown))
    assert rec["result"]["markdown"].endswith(OUT_B + "\nthree, edited\n")
    assert CELL_NOEVAL in rec["result"]["markdown"]


def test_unlabelled_hidden_cell_does_not_claim_a_labelled_div():
    # Prose added between a hidden cell and the figure after it must land
    # before that figure, not after it.
    old = source("one\n", "\n", "\nthree\n", cells=(CELL_HIDDEN, CELL_LABELLED))
    new = source(
        "one\n", "\nNEW PROSE HERE\n", "\nthree\n", cells=(CELL_HIDDEN, CELL_LABELLED)
    )
    markdown = FRONT + "\n" + "one\n" + "\n" + OUT_B_LABELLED + "\nthree\n"
    rec = fr.realign(old, new, frozen(old, markdown))
    assert norm(rec["result"]["markdown"]) == norm(
        FRONT + "one\n" + "NEW PROSE HERE\n" + OUT_B_LABELLED + "\nthree\n"
    )


def test_unlabelled_hidden_cell_does_not_claim_the_next_unlabelled_div():
    # Same shape with no labels at all: the execution count gives it away.
    old = source("one\n", "\n", "\nthree\n", cells=(CELL_HIDDEN, CELL_B))
    new = source(
        "one\n", "\nNEW PROSE HERE\n", "\nthree\n", cells=(CELL_HIDDEN, CELL_B)
    )
    markdown = FRONT + "\n" + "one\n" + "\n" + OUT_B + "\nthree\n"
    rec = fr.realign(old, new, frozen(old, markdown))
    assert norm(rec["result"]["markdown"]) == norm(
        FRONT + "one\n" + "NEW PROSE HERE\n" + OUT_B + "\nthree\n"
    )


def test_mixed_case_label_keeps_its_case():
    cell = CELL_LABELLED.replace("fig-b", "fig-MyPlot")
    out = OUT_B_LABELLED.replace("cell-fig-b", "cell-fig-MyPlot")
    old = source(*DEFAULT, cells=(CELL_A, cell))
    new = old.replace("three", "three, edited")
    rec = fr.realign(old, new, frozen(old, stored(*DEFAULT, outs=(OUT_A, out))))
    assert rec["result"]["markdown"].endswith(out + "\nthree, edited\n")


def test_asis_cell_is_bridged_to_the_next_prose():
    old = source(*DEFAULT, cells=(CELL_ASIS, CELL_B))
    new = old.replace("three", "three, edited")
    markdown = (
        FRONT
        + "\n"
        + "one\n"
        + "<script>1</script>\n"
        + "\ntwo\n"
        + "\n"
        + OUT_B
        + "\nthree\n"
    )
    rec = fr.realign(old, new, frozen(old, markdown))
    assert "<script>1</script>" in rec["result"]["markdown"]
    assert rec["result"]["markdown"].endswith(OUT_B + "\nthree, edited\n")


def test_asis_cell_followed_directly_by_a_cell_runs_to_its_div():
    old = source("one\n", "\n\n", "\nthree\n", cells=(CELL_ASIS, CELL_LABELLED))
    new = old.replace("three", "three, edited")
    markdown = (
        FRONT + "\n" + "one\n" + "<script>1</script>\n\n" + OUT_B_LABELLED + "\nthree\n"
    )
    rec = fr.realign(old, new, frozen(old, markdown))
    assert rec["result"]["markdown"] == (
        FRONT
        + "\n"
        + "one\n"
        + "<script>1</script>\n\n"
        + OUT_B_LABELLED
        + "\nthree, edited\n"
    )


def test_asis_cell_last_in_post_takes_the_rest():
    old = source("one\n", "\ntwo\n", "\n", cells=(CELL_A, CELL_ASIS))
    new = old.replace("two", "two, edited")
    markdown = (
        FRONT + "\n" + "one\n" + "\n" + OUT_A + "\ntwo\n" + "<script>1</script>\n"
    )
    rec = fr.realign(old, new, frozen(old, markdown))
    assert rec["result"]["markdown"].endswith("two, edited\n<script>1</script>\n")


def test_inline_code_values_are_carried_into_edited_prose():
    parts = ("one\n", '\nThe mean is `{python} f"{m:.2f}"` here.\n', "\nthree\n")
    old = source(*parts)
    new = old.replace("here.", "here, edited.")
    markdown = stored("one\n", "\nThe mean is 2\\.50 here.\n", "\nthree\n")
    rec = fr.realign(old, new, frozen(old, markdown))
    assert "The mean is 2\\.50 here, edited.\n" in rec["result"]["markdown"]


def test_knitr_inline_code_is_treated_like_python_inline():
    parts = ("one\n", "\nThe ratio is `r fmt(x, 4)` here.\n", "\nthree\n")
    old = source(*parts)
    new = old.replace("here.", "here, edited.")
    markdown = stored("one\n", "\nThe ratio is 1.0667 here.\n", "\nthree\n")
    rec = fr.realign(old, new, frozen(old, markdown))
    assert "The ratio is 1.0667 here, edited.\n" in rec["result"]["markdown"]


def test_changed_inline_code_is_refused():
    parts = ("one\n", '\nThe mean is `{python} f"{m:.2f}"` here.\n', "\nthree\n")
    old = source(*parts)
    new = old.replace('f"{m:.2f}"', 'f"{m:.3f}"')
    markdown = stored("one\n", "\nThe mean is 2\\.50 here.\n", "\nthree\n")
    with pytest.raises(fr.RealignError, match="inline"):
        fr.realign(old, new, frozen(old, markdown))


def test_mermaid_edit_counts_as_prose():
    parts = ("one\n", "\ntwo\n" + MERMAID, "\nthree\n")
    old = source(*parts)
    new = old.replace("A --> B", "A --> C")
    rec = realign_ok(old, new, parts)
    assert "A --> C" in rec["result"]["markdown"]


def test_output_containing_div_markers_inside_a_fence_is_skipped_whole():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    tricky = OUT_A.replace("[0 1 2]", ":::\n::: {.x}\n[0 1 2]")
    rec = fr.realign(old, new, frozen(old, stored(*DEFAULT, outs=(tricky, OUT_B))))
    assert tricky in rec["result"]["markdown"]
    assert norm(rec["result"]["markdown"]) == norm(
        stored("one!\n", "\ntwo\n", "\nthree\n", outs=(tricky, OUT_B))
    )


def test_changed_cell_is_refused():
    old = source(*DEFAULT)
    new = old.replace("np.arange(3)", "np.arange(4)")
    with pytest.raises(fr.RealignError, match="executable cell 1 changed"):
        fr.realign(old, new, frozen(old, stored(*DEFAULT)))


def test_added_cell_is_refused():
    old = source(*DEFAULT)
    new = old + "\n" + CELL_A
    with pytest.raises(fr.RealignError, match="cell count changed"):
        fr.realign(old, new, frozen(old, stored(*DEFAULT)))


def test_prose_rewritten_by_quarto_is_refused():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    markdown = stored(*DEFAULT).replace("\ntwo\n", "\nTWO\n")
    with pytest.raises(
        fr.RealignError, match="segment 2 is not in the frozen markdown"
    ):
        fr.realign(old, new, frozen(old, markdown))


def test_rewritten_frontmatter_is_refused():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    markdown = stored(*DEFAULT).replace("eigen-blog", "python3")
    with pytest.raises(fr.RealignError, match="segment 1 is not in the frozen"):
        fr.realign(old, new, frozen(old, markdown))


def test_unlocatable_cell_is_refused():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    markdown = stored(*DEFAULT).replace("::: {.cell execution_count=1}", "text")
    with pytest.raises(fr.RealignError, match="cannot tell where cell 1"):
        fr.realign(old, new, frozen(old, markdown))


def test_trailing_content_in_record_is_refused():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    with pytest.raises(fr.RealignError, match="after the last prose segment"):
        fr.realign(old, new, frozen(old, stored(*DEFAULT) + "extra\n"))


def test_input_record_is_not_mutated():
    old = source(*DEFAULT)
    new = source("one!\n", "\ntwo\n", "\nthree\n")
    rec = frozen(old, stored(*DEFAULT))
    before = rec["hash"], rec["result"]["markdown"]
    fr.realign(old, new, rec)
    assert (rec["hash"], rec["result"]["markdown"]) == before


def test_pad_blocks_pads_fences_and_divs_only_where_needed():
    text = "a\n```python\nx\n```\nb\n::: {.callout-note}\nc\n:::\nd\n"
    assert fr.pad_blocks(text) == (
        "a\n\n```python\nx\n```\n\nb\n\n::: {.callout-note}\nc\n:::\n\nd\n"
    )
    already = "a\n\n```python\nx\n```\n\nb\n"
    assert fr.pad_blocks(already) == already
