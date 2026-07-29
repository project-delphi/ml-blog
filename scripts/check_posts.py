#!/usr/bin/env python3
"""Guard the two invariants that keep this blog renderable off one laptop.

1. A post that executes code must pin a dedicated kernel and commit the
   versions behind it, so its environment can be rebuilt from the repo.
2. Every post's frozen output must match its current source, because a
   project render re-executes anything whose source hash has drifted --
   and for the legacy posts there is no environment left to execute in.

Invariant 2 is the one that actually bit: the cover-image commits edited
11 post sources without re-rendering, silently marking their frozen output
stale. Nothing noticed until a full render was attempted months later.

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
    "classifying_text_chunks",
    "data-types",
    "datasets",
    "diffusers",
    "evaluation-metrics",
    "features-importance-after-clustering",
    "hub",
    "hugging-face-evaluate-library",
    "langchain",
    "optimum",
    "peft",
    "poor-persons-bayesian",
    "post-with-code",
    "setfit",
    "soft-vs-hard-prompts",
    "tokenizers",
    "transformers-library",
    "working-with-quarto",
}

# Frozen output that is knowingly stale. Both are prose posts with no code
# cells at all, so a re-execution runs nothing and the staleness is inert.
STALE_FREEZE_OK = {"pinecone-vs-weaviate", "pyspark"}


def executes_code(text: str) -> bool:
    """True if the post has at least one cell Quarto would actually run."""
    cells = [b for lang, b in CELL_RE.findall(text) if lang in COMPUTE_LANGS]
    return any(not EVAL_FALSE_RE.search(c) for c in cells)


def check_environment(slug: str, text: str, post_dir: Path) -> list[str]:
    """A code-executing post must pin a kernel and commit its versions."""
    if slug in LEGACY_NO_ENV:
        return []
    problems = []
    kernel = KERNEL_RE.search(text)
    if not kernel:
        problems.append(
            "executes code but declares no `jupyter:` kernel, so it resolves to "
            "system Python. Give it a .venv-<slug> and a named kernel (CLAUDE.md)."
        )
    elif kernel.group(1) == "python3":
        problems.append(
            "pins the shared `python3` kernel rather than a dedicated one, so its "
            "dependencies are whatever happens to be installed."
        )
    if not (post_dir / "requirements.txt").exists():
        problems.append(
            "executes code but has no requirements.txt, so a re-render on drifted "
            "dependencies silently changes published output. Freeze it with "
            "`uv pip freeze --python .venv-<slug>/bin/python > requirements.txt`."
        )
    return problems


def check_freeze(slug: str, source: Path) -> list[str]:
    """Frozen output is keyed on an md5 of the source; drift means re-execution."""
    record = FREEZE / slug / "index" / "execute-results" / "html.json"
    if not record.exists() or slug in STALE_FREEZE_OK:
        return []
    stored = json.loads(record.read_text())["hash"]
    actual = hashlib.md5(source.read_bytes()).hexdigest()
    if stored == actual:
        return []
    return [
        f"frozen output is stale ({stored[:8]} != {actual[:8]}): the source changed "
        "without a re-render, so a project render will try to re-execute this post. "
        "Re-render it, or realign the hash if only prose changed."
    ]


def main() -> int:
    failures: dict[str, list[str]] = {}
    for post_dir in sorted(p for p in POSTS.iterdir() if p.is_dir()):
        source = post_dir / "index.qmd"
        if not source.exists():
            continue  # .ipynb posts carry their own stored outputs
        text = source.read_text(errors="ignore")
        problems = check_freeze(post_dir.name, source)
        if executes_code(text):
            problems += check_environment(post_dir.name, text, post_dir)
        if problems:
            failures[post_dir.name] = problems

    if not failures:
        print(f"check-posts: OK ({len(LEGACY_NO_ENV)} legacy posts grandfathered)")
        return 0

    for slug, problems in failures.items():
        print(f"\nposts/{slug}/index.qmd")
        for p in problems:
            print(f"  - {p}")
    print(f"\ncheck-posts: {len(failures)} post(s) need attention")
    return 1


if __name__ == "__main__":
    sys.exit(main())
