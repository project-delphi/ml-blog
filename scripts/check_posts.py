#!/usr/bin/env python3
"""Guard the two invariants that keep this blog renderable off one laptop.

1. A post that executes code must pin a dedicated kernel and commit the
   versions behind it, so its environment can be rebuilt from the repo.
2. Every post's frozen output must match its current source, because a
   project render re-executes anything whose source hash has drifted --
   and for the legacy posts there is no environment left to execute in.
3. Every kernel a post pins must be registered by `make kernels-stub`.
   Quarto resolves kernelspecs while *indexing* the project, before it
   consults _freeze/, so one missing name fails the entire render on a
   fresh clone -- while passing silently on the laptop that happens to
   have the real kernel installed.
4. Every post must appear in docs/listings.json, which only a *project*
   render writes. A single-document render publishes the post's own page
   and never touches the site index, so the post is live at its URL and
   invisible from the home page.

Invariant 2 is the one that actually bit: the cover-image commits edited
11 post sources without re-rendering, silently marking their frozen output
stale. Nothing noticed until a full render was attempted months later.
Invariant 4 bit the same way: claude-api-python-sdk shipped from a
single-document render and was missing from the listing for as long as
nobody scrolled looking for it.

Stdlib only, so it runs against any interpreter without installing anything.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSTS = ROOT / "posts"
FREEZE = ROOT / "_freeze" / "posts"
LISTINGS = ROOT / "docs" / "listings.json"
MAKEFILE = ROOT / "Makefile"

# The kernel list inside the kernels-stub recipe: `@for k in a b \<newline> c; do`
STUB_LIST_RE = re.compile(r"for k in\s+(.*?);\s*do", re.S)

CELL_RE = re.compile(r"^```\{(\w+)\}\n(.*?)^```", re.S | re.M)
EVAL_FALSE_RE = re.compile(r"#\|\s*eval:\s*false")
KERNEL_RE = re.compile(r"^jupyter:\s*(\S+)", re.M)

# Languages that need a kernel and therefore an environment. Diagram blocks
# (mermaid, dot) are handled by Quarto's own filters -- langgraph-vs-llamaindex
# is all `eval: false` Python plus four mermaid diagrams, and needs no venv.
COMPUTE_LANGS = {"python", "r", "julia"}

# Posts predating the venv-per-post convention: no dedicated kernel, no
# requirements.txt, and dependencies (diffusers, setfit, peft, transformers)
# whose environments no longer exist anywhere. They are reproducible only
# through committed _freeze/. Grandfathered so this check is green today and
# fires only on NEW drift -- a checker that is red on arrival gets ignored.
#
# Shrink this list, never grow it: if you edit one of these posts, build it a
# venv + kernel + requirements.txt per CLAUDE.md and delete its line. Editing
# the source invalidates its frozen output, so there is no third option.
LEGACY_NO_ENV = {
    "data-types",
    "features-importance-after-clustering",
    "poor-persons-bayesian",
    "post-with-code",
    "working-with-quarto",
}

# Frozen output that is knowingly stale. Both are prose posts with no code
# cells at all, so a re-execution runs nothing and the staleness is inert.
STALE_FREEZE_OK = {"pinecone-vs-weaviate", "pyspark"}


def executes_code(text: str) -> bool:
    """Report whether the post has a cell Quarto would actually run."""
    cells = [b for lang, b in CELL_RE.findall(text) if lang in COMPUTE_LANGS]
    return any(not EVAL_FALSE_RE.search(c) for c in cells)


def check_environment(slug: str, text: str, post_dir: Path) -> list[str]:
    """Require a pinned kernel and committed versions for a code post."""
    if slug in LEGACY_NO_ENV:
        return []
    problems = []
    kernel = KERNEL_RE.search(text)
    if not kernel:
        problems.append(
            "executes code but declares no `jupyter:` kernel, so it resolves to "
            "system Python. Give it a .venv-<slug> and a named kernel (CLAUDE.md).",
        )
    elif kernel.group(1) == "python3":
        problems.append(
            "pins the shared `python3` kernel rather than a dedicated one, so its "
            "dependencies are whatever happens to be installed.",
        )
    if not (post_dir / "requirements.txt").exists():
        problems.append(
            "executes code but has no requirements.txt, so a re-render on drifted "
            "dependencies silently changes published output. Freeze it with "
            "`uv pip freeze --python .venv-<slug>/bin/python > requirements.txt`.",
        )
    return problems


def check_freeze(slug: str, source: Path, executes: bool) -> list[str]:
    """Match frozen output against its source; drift means re-execution."""
    record = FREEZE / slug / "index" / "execute-results" / "html.json"
    if not record.exists():
        # For a legacy post the frozen output is the *only* reproducible copy
        # of what it computes -- there is no environment left to regenerate it
        # from. Losing the record is the unrecoverable case, so say so loudly.
        if slug in LEGACY_NO_ENV:
            return [
                "has no _freeze/ record, and no environment to rebuild one from. "
                "Restore it from git history: a project render will otherwise try "
                "to execute this post and fail.",
            ]
        return []
    if slug in STALE_FREEZE_OK:
        # Exemption is honoured only while its premise holds. If code cells get
        # added later, the staleness stops being inert and must be reported.
        if executes:
            return [
                "is exempted in STALE_FREEZE_OK on the grounds that it has no code "
                "cells, but it now executes code. Re-render it and drop the exemption.",
            ]
        return []
    stored = json.loads(record.read_text())["hash"]
    actual = hashlib.md5(source.read_bytes()).hexdigest()
    if stored == actual:
        return []
    return [
        f"frozen output is stale ({stored[:8]} != {actual[:8]}): the source changed "
        "without a re-render, so a project render will try to re-execute this post. "
        "Re-render it, or realign the hash if only prose changed.",
    ]


def listed_slugs() -> set[str] | None:
    """Read the slugs the built site actually lists, or None if unbuilt."""
    if not LISTINGS.exists():
        # A fresh clone has no docs/ yet. Nothing to compare against, and a
        # checker that is red on arrival gets ignored.
        return None
    slugs = set()
    for listing in json.loads(LISTINGS.read_text()):
        for item in listing.get("items", []):
            parts = item.strip("/").split("/")
            if len(parts) >= 2 and parts[0] == "posts":
                slugs.add(parts[1])
    return slugs


def check_listed(slugs: list[str]) -> dict[str, list[str]]:
    """Require every post to appear in the site index a project render writes."""
    listed = listed_slugs()
    if listed is None:
        return {}
    return {
        f"posts/{slug}/": [
            "is missing from docs/listings.json, so it is published at its own "
            "URL but absent from the home page and site search. Only a project "
            "render writes the site index -- a single-document `quarto render "
            "posts/<slug>/index.qmd` never touches it. Run "
            '`QUARTO_PYTHON="$(pwd)/.venv/bin/python" quarto render .` and '
            "commit docs/.",
        ]
        for slug in slugs
        if slug not in listed
    }


def stubbed_kernels() -> set[str]:
    """Read the kernel names `make kernels-stub` registers."""
    match = STUB_LIST_RE.search(MAKEFILE.read_text())
    if not match:
        return set()
    return set(match.group(1).replace("\\", "").split())


def check_kernel_stubs(pinned: dict[str, str]) -> list[str]:
    """Require every pinned kernel to be one `make kernels-stub` registers."""
    stubbed = stubbed_kernels()
    if not stubbed:
        return ["could not parse the kernel list out of the kernels-stub recipe."]
    return [
        f"`{kernel}` is pinned by posts/{slug}/index.qmd but is not in the "
        "kernels-stub list, so a fresh clone cannot render *any* page. Add it "
        "to the Makefile."
        for slug, kernel in sorted(pinned.items())
        if kernel not in stubbed
    ]


def main() -> int:
    failures: dict[str, list[str]] = {}
    pinned: dict[str, str] = {}
    slugs: list[str] = []
    for post_dir in sorted(p for p in POSTS.iterdir() if p.is_dir()):
        source = post_dir / "index.qmd"
        if source.exists() or (post_dir / "index.ipynb").exists():
            slugs.append(post_dir.name)
        if not source.exists():
            continue  # .ipynb posts carry their own stored outputs
        text = source.read_text(errors="ignore")
        executes = executes_code(text)
        kernel = KERNEL_RE.search(text)
        if kernel:
            pinned[post_dir.name] = kernel.group(1)
        problems = check_freeze(post_dir.name, source, executes)
        if executes:
            problems += check_environment(post_dir.name, text, post_dir)
        if problems:
            failures[f"posts/{post_dir.name}/index.qmd"] = problems

    failures.update(check_listed(slugs))

    stub_problems = check_kernel_stubs(pinned)
    if stub_problems:
        failures["Makefile (kernels-stub)"] = stub_problems

    if not failures:
        print(f"check-posts: OK ({len(LEGACY_NO_ENV)} legacy posts grandfathered)")
        return 0

    for where, problems in failures.items():
        print(f"\n{where}")
        for p in problems:
            print(f"  - {p}")
    print(f"\ncheck-posts: {len(failures)} file(s) need attention")
    return 1


if __name__ == "__main__":
    sys.exit(main())
