#!/usr/bin/env python3
"""Answer bounded questions about rendered output under docs/.

`docs/` is 109 MB across 524 files and `docs/search.json` is 1.9 MB, so
`.claude/settings.json` denies the Read tool on it: one careless read ends a
session. Every question actually asked of `docs/` is a counting question, and
this script answers those without ever returning bulk content. Output is capped
at MAX_LINES lines and every excerpt at MAX_EXCERPT characters, so no
invocation can flood a caller's context however the pattern is written.

Read-only: it opens files for reading and writes nothing.

    docs_query.py count PATTERN PATH...        matches per file
    docs_query.py files PATTERN GLOB           which files match
    docs_query.py exists PATH...               presence and size
    docs_query.py excerpt PATTERN PATH [-n 3]  up to n short matched strings
    docs_query.py post SLUG                    is the post live everywhere
    docs_query.py widget SLUG                  is the served bundle current

Stdlib only, so it runs against any interpreter without installing anything.
"""

from __future__ import annotations

import argparse
import glob as globlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
POSTS = ROOT / "posts"

MAX_LINES = 60
MAX_EXCERPT = 100
# Read in chunks so a 1.9 MB search.json never lands in memory whole. A match
# longer than OVERLAP could still be split across two reads; nothing this is
# asked about comes close.
CHUNK = 1 << 20
OVERLAP = 256


def emit(lines: list[str]) -> None:
    """Print at most MAX_LINES lines, saying plainly when more were dropped."""
    for line in lines[:MAX_LINES]:
        print(line)
    if len(lines) > MAX_LINES:
        print(f"... {len(lines) - MAX_LINES} more line(s) suppressed")


def compile_pattern(args) -> re.Pattern[str]:
    """The pattern, literal under -F. A bad regex is a message, not a traceback."""
    raw = re.escape(args.pattern) if getattr(args, "fixed", False) else args.pattern
    try:
        return re.compile(raw)
    except re.error as exc:
        raise SystemExit(
            f"bad pattern {args.pattern!r}: {exc}. Pass -F to match it literally."
        )


def resolve(path: str) -> Path:
    """Accept a repo-relative or docs-relative path; refuse to leave the repo."""
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT):
        raise SystemExit(f"refusing a path outside the repo: {path}")
    return p


def count_in(path: Path, pattern: re.Pattern[str]) -> int:
    """Count matches without holding the file in memory.

    Each read is prefixed with the previous window's last OVERLAP characters so
    a match straddling a chunk edge is still seen. Anything ending inside that
    prefix was already counted last time round, so only matches reaching past
    it are added -- counting the whole buffer would double every match that
    happened to land in the overlap.
    """
    total, tail = 0, ""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            buf = tail + chunk
            seen = len(tail)
            total += sum(1 for m in pattern.finditer(buf) if m.end() > seen)
            tail = buf[-OVERLAP:]
    return total


def iter_matches(path: Path, pattern: re.Pattern[str], limit: int) -> list[str]:
    """Up to `limit` matched strings, truncated, without reading the whole file."""
    found: list[str] = []
    tail = ""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while len(found) < limit:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            buf = tail + chunk
            seen = len(tail)
            for m in pattern.finditer(buf):
                if m.end() <= seen:
                    continue
                found.append(m.group(0)[:MAX_EXCERPT])
                if len(found) == limit:
                    break
            tail = buf[-OVERLAP:]
    return found


def cmd_count(args) -> int:
    pat = compile_pattern(args)
    out = []
    for raw in args.paths:
        p = resolve(raw)
        out.append(f"{count_in(p, pat) if p.is_file() else 'missing'} {raw}")
    emit(out)
    return 0


def cmd_files(args) -> int:
    pat = compile_pattern(args)
    hits = []
    for raw in sorted(globlib.glob(args.glob, root_dir=ROOT, recursive=True)):
        # a glob can escape the repo with ..; resolve() refuses those
        path = resolve(raw)
        if path.is_file() and count_in(path, pat):
            hits.append(raw)
    emit(hits or ["(no file matches)"])
    print(f"{len(hits)} file(s) match")
    return 0


def cmd_exists(args) -> int:
    out = []
    for raw in args.paths:
        p = resolve(raw)
        if p.is_file():
            out.append(f"yes {p.stat().st_size} {raw}")
        elif p.is_dir():
            out.append(f"dir {len(list(p.iterdir()))} entries {raw}")
        else:
            out.append(f"no - {raw}")
    emit(out)
    return 0


def cmd_excerpt(args) -> int:
    pat = compile_pattern(args)
    p = resolve(args.path)
    if not p.is_file():
        print(f"missing {args.path}")
        return 1
    found = iter_matches(p, pat, args.n)
    emit(found or ["(no match)"])
    return 0


def cmd_post(args) -> int:
    """Is the post live on its own page, in the listing, and in search?"""
    slug = args.slug
    page = DOCS / "posts" / slug / "index.html"
    rows = [f"page: {'yes' if page.is_file() else 'NO'} docs/posts/{slug}/index.html"]
    for name in ("listings.json", "search.json"):
        f = DOCS / name
        n = count_in(f, re.compile(re.escape(f"posts/{slug}/"))) if f.is_file() else 0
        rows.append(f"{name}: {'yes' if n else 'NO'} ({n} reference(s))")
    emit(rows)
    return 0 if all("NO" not in r for r in rows) else 1


def source_markers(text: str, want: int = 3) -> list[str]:
    """Distinctive lines of a bundle, spread through it, to search a page for."""
    lines = [ln.strip() for ln in text.splitlines()]
    usable = [ln for ln in lines if len(ln) >= 40]
    if not usable:
        return []
    step = max(1, len(usable) // (want + 1))
    return [usable[min(len(usable) - 1, step * (i + 1))][:80] for i in range(want)]


def cmd_widget(args) -> int:
    """Is the bundle the page serves the one in the source sidecar?

    Quarto hashes index.qmd alone, so editing widgets.js leaves _freeze/ valid
    and a project render can keep serving the old bundle with no warning. Two
    shapes exist and the evidence differs. A kit post publishes the sidecar as
    a resource, so the published file can be compared byte for byte and the
    page only has to reference it. An older post prints the bundle into an
    inline <script> from a Python cell, so there is no file to compare and the
    only evidence is whether the current source's own lines are in the page.
    Counting a mount id would prove nothing either way: the mount div lives in
    index.qmd and is there whatever the bundle's age.
    """
    slug = args.slug
    src = POSTS / slug / "widgets.js"
    page = DOCS / "posts" / slug / "index.html"
    if not src.is_file():
        print(f"no source sidecar: posts/{slug}/widgets.js")
        return 1
    if not page.is_file():
        print(f"no rendered page: docs/posts/{slug}/index.html")
        return 1

    text = src.read_text(encoding="utf-8", errors="replace")
    served = DOCS / "posts" / slug / "widgets.js"
    rows: list[str] = []
    stale = False

    if served.is_file():
        same = served.read_bytes() == src.read_bytes()
        rows.append(
            f"published sidecar: {'identical' if same else 'DIFFERS'} from source"
        )
        stale = stale or not same
        linked = count_in(page, re.compile(r'src="[^"]*widgets\.js"'))
        rows.append(f"page loads it: {'yes' if linked else 'NO'}")
        stale = stale or not linked
    else:
        rows.append("inline bundle: no published sidecar, checking the page text")
        markers = source_markers(text)
        if not markers:
            rows.append("no usable marker in the source; CANNOT VERIFY")
            emit(rows)
            return 1
        hits = 0
        for m in markers:
            # Some print cells rewrite `</` to `<\/` on the way into the page
            # and some do not, so a marker counts as present in either form.
            forms = {m, m.replace("</", "<\\/")}
            found = any(count_in(page, re.compile(re.escape(f))) for f in forms)
            hits += 1 if found else 0
            rows.append(f"marker {'found' if found else 'MISSING'}: {m[:48]}")
        stale = stale or hits < len(markers)
        rows.append(f"{hits}/{len(markers)} markers present")

    rows.append(f"verdict: {'STALE' if stale else 'current'}")
    emit(rows)
    return 1 if stale else 0


def main(argv: list[str] | None = None) -> int:
    fixed = argparse.ArgumentParser(add_help=False)
    fixed.add_argument(
        "-F",
        "--fixed",
        action="store_true",
        default=argparse.SUPPRESS,
        help="literal pattern, not regex",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], parents=[fixed])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("count", parents=[fixed])
    c.add_argument("pattern")
    c.add_argument("paths", nargs="+")
    c.set_defaults(fn=cmd_count)

    f = sub.add_parser("files", parents=[fixed])
    f.add_argument("pattern")
    f.add_argument("glob")
    f.set_defaults(fn=cmd_files)

    e = sub.add_parser("exists")
    e.add_argument("paths", nargs="+")
    e.set_defaults(fn=cmd_exists)

    x = sub.add_parser("excerpt", parents=[fixed])
    x.add_argument("pattern")
    x.add_argument("path")
    x.add_argument("-n", type=int, default=3)
    x.set_defaults(fn=cmd_excerpt)

    p = sub.add_parser("post")
    p.add_argument("slug")
    p.set_defaults(fn=cmd_post)

    w = sub.add_parser("widget")
    w.add_argument("slug")
    w.set_defaults(fn=cmd_widget)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
