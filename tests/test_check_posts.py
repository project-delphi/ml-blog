"""Tests for scripts/check_posts.py."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_posts as cp  # noqa: E402


def test_executes_code_ignores_eval_false_and_diagrams():
    assert cp.executes_code("```{python}\nprint(1)\n```\n")
    assert not cp.executes_code("```{python}\n#| eval: false\nprint(1)\n```\n")
    assert not cp.executes_code("```{mermaid}\nflowchart LR\n```\n")
    assert cp.executes_code("```{r}\nx <- 1\n```\n")


def test_check_freeze_reports_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "FREEZE", tmp_path / "_freeze")
    src = tmp_path / "index.qmd"
    src.write_text("---\ntitle: t\n---\nhello\n")
    rec = tmp_path / "_freeze" / "slug" / "index" / "execute-results" / "html.json"
    rec.parent.mkdir(parents=True)
    rec.write_text(json.dumps({"hash": hashlib.md5(src.read_bytes()).hexdigest()}))
    assert cp.check_freeze("slug", src, executes=True) == []
    src.write_text("---\ntitle: t\n---\nhello, edited\n")
    problems = cp.check_freeze("slug", src, executes=True)
    assert len(problems) == 1 and "freeze-realign" in problems[0]


def test_check_freeze_missing_record_is_clean_unless_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "FREEZE", tmp_path / "_freeze")
    src = tmp_path / "index.qmd"
    src.write_text("x")
    assert cp.check_freeze("not-legacy", src, executes=False) == []
    assert cp.check_freeze("data-types", src, executes=True)


def test_categories_parses_inline_and_block_lists():
    inline = '---\ntitle: t\ncategories: [Machine Learning, "Statistics"]\n---\n'
    block = "---\ntitle: t\ncategories:\n  - Python\n  - 'NLP'\ntags: [a]\n---\n"
    assert cp.categories(inline) == ["Machine Learning", "Statistics"]
    assert cp.categories(block) == ["Python", "NLP"]
    assert cp.categories("---\ntitle: t\n---\n") == []


def test_check_categories_flags_unknown_and_missing():
    canon = {"Machine Learning", "Python"}
    assert cp.check_categories(["Machine Learning"], canon) == []
    assert cp.check_categories([], canon) == ["has no categories."]
    problems = cp.check_categories(["machine learning", "Docker"], canon)
    assert any("`machine learning`" in p and "Machine Learning" in p for p in problems)
    assert any("`Docker`" in p for p in problems)


def _fixture_repo(tmp_path, category):
    """One post, one stubbed kernel, one listing entry: enough for main()."""
    post = tmp_path / "posts" / "slug"
    post.mkdir(parents=True)
    (post / "index.qmd").write_text(
        f'---\ntitle: t\ndate: "2026-01-01"\ncategories: [{category}]\n---\nprose\n'
    )
    canon = tmp_path / "categories.txt"
    canon.write_text("Machine Learning\n")
    listings = tmp_path / "listings.json"
    listings.write_text(json.dumps([{"items": ["/posts/slug/index.html"]}]))
    return post, canon, listings


def test_categories_check_runs_by_default_and_no_categories_skips_it(
    tmp_path, monkeypatch
):
    _, canon, listings = _fixture_repo(tmp_path, "Statistics")
    monkeypatch.setattr(cp, "POSTS", tmp_path / "posts")
    monkeypatch.setattr(cp, "FREEZE", tmp_path / "_freeze")
    monkeypatch.setattr(cp, "CATEGORIES", canon)
    monkeypatch.setattr(cp, "LISTINGS", listings)
    monkeypatch.setattr(cp, "MAKEFILE", tmp_path / "Makefile")
    (tmp_path / "Makefile").write_text(
        "kernels-stub: install\n\t@for k in blog-base \\\n\t          ; do :; done\n"
    )
    # `Statistics` is absent from this fixture's canonical list, so the default
    # run must fail and only `--no-categories` may pass.
    assert cp.main([]) != 0
    assert cp.main(["--no-categories"]) == 0
