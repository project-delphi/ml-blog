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
markdown where the old one sits. It refuses, and says why, when:

- any executable cell body differs (that is a real re-execution);
- an old prose segment is not found verbatim in the stored markdown where
  the previous cell's output ends (Quarto rewrote it, so the splice would be
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
    # Walk the stored markdown positionally, never by search: a short prose
    # segment such as a lone newline would match inside a cell's output. The
    # stored markdown is P0 O0 P1 O1 ... Pn, each Oi a `::: {.cell}` div, so
    # each old prose segment must sit exactly where the previous output ends.
    markdown = record["result"]["markdown"]
    out = []
    pos = 0
    for i, (old_p, new_p) in enumerate(zip(old_prose, new_prose)):
        if i > 0:
            end = cell_output_end(markdown, pos, i)
            out.append(markdown[pos:end])
            pos = end
        elif (fm := cp.FRONTMATTER_RE.match(old_p)) and cp.FRONTMATTER_RE.match(new_p):
            # Quarto re-emits the frontmatter verbatim but follows it with one
            # more blank line than the source has. Anchor on the frontmatter,
            # keep the record's own newline run, then match the body.
            old_front, old_body = old_p[: fm.end()], old_p[fm.end() :].lstrip("\n")
            new_fm = cp.FRONTMATTER_RE.match(new_p)
            new_front, new_body = (
                new_p[: new_fm.end()],
                new_p[new_fm.end() :].lstrip("\n"),
            )
            if not markdown.startswith(old_front, pos):
                raise RealignError(
                    "the frontmatter is not in the frozen markdown verbatim; "
                    "re-render instead.\n"
                    + mismatch_context(old_front, markdown[pos : pos + len(old_front)])
                )
            pos += len(old_front)
            gap_end = pos
            while gap_end < len(markdown) and markdown[gap_end] == "\n":
                gap_end += 1
            out.append(new_front)
            out.append(markdown[pos:gap_end])
            pos = gap_end
            old_p, new_p = old_body, new_body
        if not markdown.startswith(old_p, pos):
            raise RealignError(
                f"prose segment {i + 1} is not in the frozen markdown verbatim; "
                "Quarto rewrote it, so re-render instead.\n"
                + mismatch_context(old_p, markdown[pos : pos + len(old_p)])
            )
        out.append(new_p)
        pos += len(old_p)
    if markdown[pos:].strip():
        raise RealignError(
            "the frozen markdown has content after the last prose segment; "
            "re-render instead."
        )
    out.append(markdown[pos:])
    new_record = json.loads(json.dumps(record))
    new_record["result"]["markdown"] = "".join(out)
    new_record["hash"] = hashlib.md5(new_src.encode("utf-8")).hexdigest()
    return new_record


def git_show(rev: str, path: Path) -> str | None:
    """Return the file at `rev`, or None if it did not exist there."""
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path.as_posix()}"],
        cwd=cp.ROOT,
        capture_output=True,
        text=True,
    )
    return proc.stdout if proc.returncode == 0 else None


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
    new_src = source.read_text()
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
