#!/usr/bin/env python3
"""Accept a prose-only edit to a freeze-backed post without re-executing it.

Quarto keys a post's frozen output on an md5 of index.qmd, so any edit -- one
word of prose -- marks the record stale and the next project render tries to
execute the post. For the 63 freeze-backed posts that means building the
post's real .venv first, even when no code cell changed.

Rewriting just the stored hash is NOT enough, and the advice to do so is a
trap: the record also stores the whole document as markdown, prose and
frontmatter included, with each code cell replaced by its output. A render
that hits the freeze uses that stored markdown wholesale, so a hash-only
realign publishes the *old* prose under a valid-looking record.

This script therefore does both. It finds the revision of index.qmd the
record was built from, splits old and new source into prose segments around
the executable cells, and splices each new prose segment into the stored
markdown where the old one sits. The walk is positional, never a free text
search: each cell's rendering is stepped over by parsing its `::: {.cell}`
div (matched by label when the cell has one), its kept fence (`eval: false`),
or nothing (`include: false`, or a setup cell that printed nothing, accepted
only when the prose that follows sits right there); raw `output: asis` cells
are bridged to the next prose or the next cell's div. Newline runs may be
longer in the record than in the source, because Quarto pads fences and divs
with blank lines, and inline `{python}` expressions are matched as wildcards
whose stored values are put back into the new prose. An identity realign
reproduces 60 of the 61 valid records byte for byte; the odd one out is a
knitr post with inline `r` code, which is also LEGACY_NO_ENV.

It refuses, and says why, when:

- any executable cell body or inline `{python}` expression differs (that is
  a real re-execution);
- an old prose segment is not found in the stored markdown where the
  previous cell's rendering ends (Quarto rewrote it, so the splice would be
  a guess);
- the post is in LEGACY_NO_ENV (its record is the only copy of what it
  computes; do not touch it by machine).

After a successful realign the docs/ page is still the old one -- run the
project render, which now reuses the frozen outputs and re-renders the
prose. Stdlib only, like check_posts.py, so it runs bare on any clone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_posts as cp  # noqa: E402

MAX_HISTORY = 200

# The div Quarto wraps each executed cell in: `::: {.cell ...}` or, when the
# cell has a label, `::: {#label .cell ...}`.
CELL_DIV_RE = re.compile(r"::: \{[^}\n]*\.cell[\s}]")
CELL_OPT_RE = re.compile(r"^#\|\s*([\w.-]+):\s*(.*?)\s*$")
# Inline code the engine evaluates in prose: `{python} expr`.
INLINE_RE = re.compile(r"`\{python\}[^`\n]*`")
DIV_ID_RE = re.compile(r"::: \{#([^\s}]+)")
EXEC_COUNT_RE = re.compile(r"execution_count=(\d+)")


class RealignError(Exception):
    """A reason the record cannot be realigned safely."""


def split(text: str) -> tuple[list[str], list[str]]:
    """Split source into prose segments and the executable cells between them.

    Only cells in a compute language count as cells; a ```{mermaid} block is
    rendered by a pandoc filter, not executed, so it travels with the prose.
    Returns (prose, cells) with len(prose) == len(cells) + 1.
    """
    prose, cells = [], []
    pos = 0
    for m in cp.CELL_RE.finditer(text):
        if m.group(1) not in cp.COMPUTE_LANGS:
            continue
        prose.append(text[pos : m.start()])
        cells.append(m.group(0))
        pos = m.end()
    prose.append(text[pos:])
    return prose, cells


def mismatch_context(expected: str, found: str, width: int = 60) -> str:
    """Show where two texts first differ, for the refusal message."""
    n = next(
        (i for i, (a, b) in enumerate(zip(expected, found)) if a != b),
        min(len(expected), len(found)),
    )
    lo = max(0, n - width)
    return (
        f"  first difference at offset {n}:\n"
        f"  source : {expected[lo : n + width]!r}\n"
        f"  record : {found[lo : n + width]!r}"
    )


def tolerant(text: str, after_cell: bool = False) -> re.Pattern:
    """Match `text` exactly, up to Quarto's blank-line padding and inline code.

    Quarto's engine pads every fenced block and div with a blank line when it
    stores the markdown, so a newline run in the record may be longer than the
    source's. After a cell that rendered nothing the run may also have been
    absorbed by the previous segment, so a leading run is optional there.
    Inline `{python}` expressions are stored as their values, so each one
    becomes a capture group; `realign` puts the captured values back.
    """

    def literal(chunk: str, lead_optional: bool) -> str:
        parts = re.split(r"\n+", chunk)
        pat = r"\n+".join(map(re.escape, parts))
        if lead_optional and parts and parts[0] == "" and len(parts) > 1:
            pat = r"\n*" + pat[len(r"\n+") :]
        return pat

    pieces = INLINE_RE.split(text)
    pattern = literal(pieces[0], after_cell)
    for chunk in pieces[1:]:
        pattern += r"(.*?)" + literal(chunk, False)
    return re.compile(pattern, re.S)


def with_inline_values(new_p: str, old_p: str, values: tuple[str, ...]) -> str:
    """Put the values Quarto stored for `old_p`'s inline code into `new_p`."""
    old_inline = INLINE_RE.findall(old_p)
    new_inline = INLINE_RE.findall(new_p)
    if old_inline != new_inline:
        raise RealignError(
            "an inline `{python}` expression changed, which needs a re-execution."
        )
    if not values:
        return new_p
    it = iter(values)
    return INLINE_RE.sub(lambda _m: next(it), new_p)


def pad_blocks(text: str) -> str:
    """Put a blank line before an opening fence or div and after a closing one.

    Pandoc needs fenced blocks separated from surrounding text; this is the
    padding Quarto adds when it stores the markdown, applied to new prose.
    """
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    pad_next = False
    for line in lines:
        s = line.strip()
        is_fence = s.startswith("```")
        opening = (is_fence and not in_fence) or (
            not in_fence and (s.startswith("::: {") or s.startswith(":::{"))
        )
        closing_div = not in_fence and s == ":::"
        if (opening or pad_next) and s and out and out[-1].strip():
            out.append("")
        pad_next = False
        out.append(line)
        if is_fence:
            if in_fence:
                pad_next = True
            in_fence = not in_fence
        elif closing_div:
            pad_next = True
    return "\n".join(out)


def cell_options(cell: str) -> dict[str, str]:
    """Read the leading `#| key: value` options of a cell."""
    opts = {}
    for line in cell.split("\n")[1:]:
        m = CELL_OPT_RE.match(line)
        if not m:
            break
        value = m.group(2).strip("\"'")
        # Labels are case-sensitive ids; the switches are compared lowercase.
        opts[m.group(1)] = value if m.group(1) == "label" else value.lower()
    return opts


def cell_end(
    markdown: str,
    pos: int,
    cell: str,
    k: int,
    next_prose: str,
    is_last: bool,
    next_cell: str | None = None,
    expected_count: int | None = None,
) -> int:
    """Return the index just past cell `k`'s rendering in the stored markdown.

    An executed cell becomes a `::: {.cell}` div. A cell that keeps its fence
    (`eval: false`) is stored as that fence. A cell that shows nothing
    (`include: false`, or `echo: false` with `output: false`) leaves no trace.
    Raw output (`output: asis`) has no wrapper, so its extent is found from
    the prose that follows it.
    """
    opts = cell_options(cell)
    label = opts.get("label")
    # Quarto pads a div with a blank line; step over it before looking.
    while pos < len(markdown) and markdown[pos] == "\n":
        pos += 1
    if CELL_DIV_RE.match(markdown, pos):
        # The div at hand may belong to a *later* cell when this one rendered
        # nothing. A labelled cell renders as `::: {#cell-label .cell}`, so an
        # id that is not this cell's label gives it away; so does an
        # `execution_count` other than the one this cell would have received.
        div_id = DIV_ID_RE.match(markdown, pos)
        line_end = markdown.find("\n", pos)
        count = EXEC_COUNT_RE.search(markdown, pos, line_end if line_end > 0 else None)
        # An unlabelled cell gets a random hex id, so only a `cell-` id (the
        # form a label produces) proves the div is someone else's.
        ours = True
        if div_id and label and div_id.group(1) not in (label, f"cell-{label}"):
            ours = False
        if div_id and not label and div_id.group(1).startswith("cell-"):
            ours = False
        if (
            count
            and expected_count is not None
            and int(count.group(1)) != expected_count
        ):
            ours = False
        if not ours:
            return pos
        return cell_output_end(markdown, pos, k)
    if m := tolerant(cell).match(markdown, pos):
        return m.end()
    anchor = next_prose.rstrip("\n") if is_last else next_prose
    # A cell whose options hide the code and whose run printed nothing (a
    # setup cell) leaves no trace: accept that only when the prose that
    # follows it sits right here, which is unambiguous.
    if anchor.strip() and tolerant(anchor, after_cell=True).match(markdown, pos):
        return pos
    hidden = opts.get("include") == "false" or (
        opts.get("echo") == "false" and opts.get("output") == "false"
    )
    if hidden:
        return pos
    if opts.get("output") == "asis":
        if anchor.strip():
            if m := tolerant(anchor, after_cell=True).search(markdown, pos):
                return m.start()
        elif is_last:
            return len(markdown.rstrip("\n"))
        elif next_cell is not None:
            # Adjacent cells: the raw output runs up to the next cell's div.
            next_label = cell_options(next_cell).get("label")
            div_re = (
                re.compile(r"::: \{#(?:cell-)?" + re.escape(next_label) + r"[\s}]")
                if next_label
                else CELL_DIV_RE
            )
            if m := div_re.search(markdown, pos):
                return m.start()
    raise RealignError(
        f"cannot tell where cell {k} (options {opts or 'none'}) rendered: at "
        f"offset {pos} the frozen markdown has {markdown[pos : pos + 80]!r}; "
        "re-render instead."
    )


def cell_output_end(markdown: str, pos: int, n: int) -> int:
    """Return the index just past the `::: {.cell}` div that starts at `pos`.

    Quarto renders every executed cell as a fenced div, nested one level for
    each output. Divs are counted line by line, ignoring lines inside a code
    fence, so printed output containing `:::` cannot close the cell early.
    """
    if not CELL_DIV_RE.match(markdown, pos):
        raise RealignError(
            f"expected cell {n}'s output at offset {pos} of the frozen markdown "
            f"and found {markdown[pos : pos + 80]!r}; re-render instead."
        )
    depth = 0
    in_fence = False
    i = pos
    while i < len(markdown):
        nl = markdown.find("\n", i)
        end = len(markdown) if nl < 0 else nl + 1
        line = markdown[i:end].strip()
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            if line.startswith("::: {") or line.startswith("::: ."):
                depth += 1
            elif line == ":::":
                depth -= 1
                if depth == 0:
                    return end
        i = end
    raise RealignError(f"cell {n}'s output div never closes; re-render instead.")


def realign(old_src: str, new_src: str, record: dict) -> dict:
    """Return a copy of `record` that matches `new_src`, or raise RealignError."""
    old_prose, old_cells = split(old_src)
    new_prose, new_cells = split(new_src)
    if len(old_cells) != len(new_cells):
        raise RealignError(
            f"cell count changed ({len(old_cells)} -> {len(new_cells)}); "
            "that is a re-execution, not a prose edit."
        )
    for i, (a, b) in enumerate(zip(old_cells, new_cells)):
        if a != b:
            raise RealignError(
                f"executable cell {i + 1} changed; re-render with the post's venv."
            )
    # Walk the stored markdown positionally, never by free search: a short
    # prose segment such as a lone newline would match inside a cell's output.
    # The stored markdown is P0 O0 P1 O1 ... Pn; each old prose segment must
    # sit exactly where the previous cell's rendering ends, allowing only for
    # the blank lines Quarto pads fences and divs with.
    markdown = record["result"]["markdown"]
    out = []
    pos = 0
    n = len(old_cells)
    # The `execution_count` on a cell's div line is the cell's index among the
    # source's cells in the same language, `eval: false` ones included (the
    # kernel's own counter sits on the inner output div); a `{r}` block in a
    # Python post is kept as a fence and does not count.
    seen: dict[str, int] = {}
    index = []
    for cell in old_cells:
        lang = cell[3:].split("}", 1)[0].strip("{")
        seen[lang] = seen.get(lang, 0) + 1
        index.append(seen[lang])
    for i, (old_p, new_p) in enumerate(zip(old_prose, new_prose)):
        if i > 0:
            end = cell_end(
                markdown,
                pos,
                old_cells[i - 1],
                i,
                old_p,
                i == n,
                old_cells[i] if i < n else None,
                index[i - 1],
            )
            out.append(markdown[pos:end])
            pos = end
        # The record may end without the source's final newline.
        anchor = old_p.rstrip("\n") if i == n else old_p
        m = tolerant(anchor, after_cell=i > 0).match(markdown, pos)
        if not m:
            raise RealignError(
                f"prose segment {i + 1} is not in the frozen markdown where cell "
                f"{i}'s output ends; Quarto rewrote it, so re-render instead.\n"
                + mismatch_context(old_p, markdown[pos : pos + len(old_p)])
            )
        # An unchanged segment keeps the record's bytes; a changed one gets the
        # values of its inline code and the blank-line padding Quarto would
        # have given it.
        if old_p == new_p:
            out.append(m.group(0))
        else:
            out.append(pad_blocks(with_inline_values(new_p, anchor, m.groups())))
        pos = m.end()
    if markdown[pos:].strip():
        raise RealignError(
            "the frozen markdown has content after the last prose segment; "
            "re-render instead."
        )
    # A changed final segment carries its own trailing newlines.
    if old_prose[-1] == new_prose[-1]:
        out.append(markdown[pos:])
    new_record = json.loads(json.dumps(record))
    new_record["result"]["markdown"] = "".join(out)
    new_record["hash"] = hashlib.md5(new_src.encode("utf-8")).hexdigest()
    return new_record


def git_show(rev: str, path: Path) -> str | None:
    """Return the file at `rev`, or None if it did not exist there."""
    # Bytes, not text: the hash Quarto and check_posts compare is over the raw
    # file, so newline translation here would make a realigned hash never match.
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path.as_posix()}"],
        cwd=cp.ROOT,
        capture_output=True,
    )
    return proc.stdout.decode("utf-8") if proc.returncode == 0 else None


def source_for_hash(path: Path, wanted: str) -> str | None:
    """Walk the file's history for the revision whose md5 the record stores."""
    proc = subprocess.run(
        [
            "git",
            "log",
            f"--max-count={MAX_HISTORY}",
            "--format=%H",
            "--",
            path.as_posix(),
        ],
        cwd=cp.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for rev in proc.stdout.split():
        text = git_show(rev, path)
        if text is not None and hashlib.md5(text.encode("utf-8")).hexdigest() == wanted:
            return text
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slug")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the realign would succeed, without writing",
    )
    args = parser.parse_args(argv)
    slug = args.slug
    if slug in cp.LEGACY_NO_ENV:
        print(f"{slug}: is in LEGACY_NO_ENV; its record is not touched by machine.")
        return 1
    source = cp.POSTS / slug / "index.qmd"
    record_path = cp.FREEZE / slug / "index" / "execute-results" / "html.json"
    if not source.exists():
        print(f"{slug}: no posts/{slug}/index.qmd")
        return 1
    if not record_path.exists():
        print(f"{slug}: has no _freeze/ record, so there is nothing to realign.")
        return 1
    record = json.loads(record_path.read_text())
    new_src = source.read_bytes().decode("utf-8")
    new_hash = hashlib.md5(new_src.encode("utf-8")).hexdigest()
    if record["hash"] == new_hash:
        print(f"{slug}: frozen record already matches the source.")
        return 0
    old_src = source_for_hash(source.relative_to(cp.ROOT), record["hash"])
    if old_src is None:
        print(
            f"{slug}: no revision in the last {MAX_HISTORY} commits of index.qmd "
            f"matches the record's hash {record['hash'][:8]}; re-render instead."
        )
        return 1
    try:
        new_record = realign(old_src, new_src, record)
    except RealignError as e:
        print(f"{slug}: refusing to realign: {e}")
        return 1
    if args.check:
        print(f"{slug}: prose-only change; realign would succeed.")
        return 0
    record_path.write_text(json.dumps(new_record, indent=2, ensure_ascii=False) + "\n")
    print(
        f"{slug}: realigned {record['hash'][:8]} -> {new_hash[:8]}. "
        "Now run the project render and commit docs/ with the source."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
