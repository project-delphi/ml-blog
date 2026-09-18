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
# Read in chunks so a 1.9 MB search.json never lands in memory whole.
CHUNK = 1 << 20


def emit(lines: list[str]) -> None:
    """Print at most MAX_LINES lines, saying plainly when more were dropped."""
    for line in lines[:MAX_LINES]:
        print(line)
    if len(lines) > MAX_LINES:
        print(f"... {len(lines) - MAX_LINES} more line(s) suppressed")


def resolve(path: str) -> Path:
    """Accept a repo-relative or docs-relative path; refuse to leave the repo."""
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT):
        raise SystemExit(f"refusing a path outside the repo: {path}")
    return p


def count_in(path: Path, pattern: re.Pattern[str]) -> int:
    """Count matches without holding the file in memory."""
    total, tail = 0, ""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            buf = tail + chunk
            total += len(pattern.findall(buf))
            # keep an overlap so a match spanning a chunk edge is not lost
            tail = buf[-256:]
    return total


def cmd_count(args) -> int:
    pat = re.compile(re.escape(args.pattern) if args.fixed else args.pattern)
    out = []
    for raw in args.paths:
        p = resolve(raw)
        out.append(f"{count_in(p, pat) if p.is_file() else 'missing'} {raw}")
    emit(out)
    return 0


def cmd_files(args) -> int:
    pat = re.compile(re.escape(args.pattern) if args.fixed else args.pattern)
    hits = [
        raw
        for raw in sorted(globlib.glob(args.glob, root_dir=ROOT, recursive=True))
        if (ROOT / raw).is_file() and count_in(ROOT / raw, pat)
    ]
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
    pat = re.compile(re.escape(args.pattern) if args.fixed else args.pattern)
    p = resolve(args.path)
    if not p.is_file():
        print(f"missing {args.path}")
        return 1
    text = p.read_text(encoding="utf-8", errors="replace")
    found = [m.group(0)[:MAX_EXCERPT] for m in pat.finditer(text)][: args.n]
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


def cmd_widget(args) -> int:
    """Is the bundle the page serves the one in the source sidecar?

    Quarto hashes index.qmd alone, so editing widgets.js leaves _freeze/ valid
    and a project render can keep serving the old bundle with no warning. The
    check is whether a marker taken from the current source appears in the
    rendered page.
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

    served = DOCS / "posts" / slug / "widgets.js"
    rows = []
    if served.is_file():
        same = served.read_bytes() == src.read_bytes()
        rows.append(
            f"published sidecar: {'identical' if same else 'DIFFERS'} from source"
        )
    else:
        rows.append("published sidecar: absent (inlined bundle, or not a resource)")

    # A marker the source defines and the page must therefore contain.
    text = src.read_text(encoding="utf-8", errors="replace")
    markers = re.findall(r'WK\.mount\("([^"]+)"', text) or re.findall(
        r'id="([a-z0-9-]*widget[a-z0-9-]*)"', text
    )
    for m in markers[:3]:
        n = count_in(page, re.compile(re.escape(m)))
        rows.append(f"mount {m}: {n} occurrence(s) in the page")
    if not markers:
        rows.append("no mount id found in the source; cannot verify freshness")
    emit(rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "-F", "--fixed", action="store_true", help="literal pattern, not regex"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("count")
    c.add_argument("pattern")
    c.add_argument("paths", nargs="+")
    c.set_defaults(fn=cmd_count)

    f = sub.add_parser("files")
    f.add_argument("pattern")
    f.add_argument("glob")
    f.set_defaults(fn=cmd_files)

    e = sub.add_parser("exists")
    e.add_argument("paths", nargs="+")
    e.set_defaults(fn=cmd_exists)

    x = sub.add_parser("excerpt")
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
