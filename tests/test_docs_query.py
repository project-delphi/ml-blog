"""Tests for scripts/docs_query.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import docs_query as dq  # noqa: E402


def _repo(tmp_path, monkeypatch):
    """A miniature docs/ and posts/ tree, standing in for the real ones."""
    docs, posts = tmp_path / "docs", tmp_path / "posts"
    (docs / "posts" / "slug").mkdir(parents=True)
    (posts / "slug").mkdir(parents=True)
    monkeypatch.setattr(dq, "ROOT", tmp_path)
    monkeypatch.setattr(dq, "DOCS", docs)
    monkeypatch.setattr(dq, "POSTS", posts)
    return docs, posts


def test_count_and_exists(tmp_path, monkeypatch, capsys):
    docs, _ = _repo(tmp_path, monkeypatch)
    (docs / "posts" / "slug" / "index.html").write_text("aXbXc")
    assert dq.main(["-F", "count", "X", "docs/posts/slug/index.html"]) == 0
    assert capsys.readouterr().out.startswith("2 ")
    assert dq.main(["exists", "docs/posts/slug/index.html", "docs/gone.html"]) == 0
    out = capsys.readouterr().out
    assert "yes 5 " in out and "no - docs/gone.html" in out


def test_count_spans_a_chunk_boundary(tmp_path, monkeypatch, capsys):
    """A match straddling the read window must still be counted once."""
    docs, _ = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(dq, "CHUNK", 64)
    body = "y" * 60 + "NEEDLE" + "y" * 60
    (docs / "posts" / "slug" / "index.html").write_text(body)
    dq.main(["-F", "count", "NEEDLE", "docs/posts/slug/index.html"])
    assert capsys.readouterr().out.startswith("1 ")


def test_output_is_capped(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dq, "MAX_LINES", 3)
    dq.emit([f"line {i}" for i in range(10)])
    out = capsys.readouterr().out
    assert out.count("\n") == 4 and "7 more line(s) suppressed" in out


def test_excerpt_is_truncated(tmp_path, monkeypatch, capsys):
    docs, _ = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(dq, "MAX_EXCERPT", 10)
    (docs / "posts" / "slug" / "index.html").write_text("z" * 500)
    dq.main(["excerpt", "z+", "docs/posts/slug/index.html"])
    assert capsys.readouterr().out.strip() == "z" * 10


def test_post_reports_a_missing_listing(tmp_path, monkeypatch, capsys):
    docs, _ = _repo(tmp_path, monkeypatch)
    (docs / "posts" / "slug" / "index.html").write_text("<html>")
    (docs / "listings.json").write_text(
        json.dumps([{"items": ["/posts/slug/index.html"]}])
    )
    (docs / "search.json").write_text("[]")  # the post is absent from search
    assert dq.main(["post", "slug"]) == 1
    out = capsys.readouterr().out
    assert "listings.json: yes" in out and "search.json: NO" in out


def test_widget_spots_a_stale_bundle(tmp_path, monkeypatch, capsys):
    docs, posts = _repo(tmp_path, monkeypatch)
    (posts / "slug" / "widgets.js").write_text(
        'WK.mount("widget-demo", function () {});'
    )
    (docs / "posts" / "slug" / "widgets.js").write_text("stale bundle")
    (docs / "posts" / "slug" / "index.html").write_text("<html>no mount here</html>")
    dq.main(["widget", "slug"])
    out = capsys.readouterr().out
    assert "DIFFERS" in out
    assert "mount widget-demo: 0 occurrence(s)" in out


def test_refuses_a_path_outside_the_repo(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    try:
        dq.main(["exists", "../../../etc/hosts"])
    except SystemExit as exc:
        assert "outside the repo" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a refusal")
