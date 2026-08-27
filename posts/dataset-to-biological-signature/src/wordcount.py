#!/usr/bin/env python3
"""Count the post's prose, and nothing else.

The brief for this post caps prose, not total page length, so code blocks,
their output, figure captions, the references list, YAML front matter and
mermaid diagrams are all excluded. Stdlib only, so it runs on any interpreter.

Usage:
    python3 posts/dataset-to-biological-signature/src/wordcount.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

POST = Path(__file__).resolve().parent.parent / "index.qmd"


def prose(text: str) -> list[str]:
    """Strip everything that is not body prose and return the remaining lines."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)  # front matter
    text = re.sub(r"^```.*?^```[ \t]*$", "", text, flags=re.S | re.M)  # fenced blocks
    text = re.sub(r"^::: .*?^:::[ \t]*$", "", text, flags=re.S | re.M)  # divs
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)  # display maths
    text = text.split("## References")[0]  # references live at the end

    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", "|", ":", "<", "[^")):
            continue  # headings, images, tables, attributes, footnote defs
        out.append(stripped)
    return out


def main() -> int:
    lines = prose(POST.read_text())
    words = sum(len(re.findall(r"[A-Za-z0-9][\w'’/-]*", line)) for line in lines)
    print(f"{POST.name}: {words} words of prose ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
