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


BUNDLE = "\n".join(
    f"const marker{i} = 'a distinctive line of bundle text';" for i in range(40)
)


def test_widget_spots_a_stale_published_sidecar(tmp_path, monkeypatch, capsys):
    docs, posts = _repo(tmp_path, monkeypatch)
    (posts / "slug" / "widgets.js").write_text(BUNDLE)
    (docs / "posts" / "slug" / "widgets.js").write_text("stale bundle")
    (docs / "posts" / "slug" / "index.html").write_text("<html>nothing of it</html>")
    assert dq.main(["widget", "slug"]) == 1
    out = capsys.readouterr().out
    assert "DIFFERS" in out and "STALE" in out


def test_widget_spots_a_stale_inline_bundle(tmp_path, monkeypatch, capsys):
    """The seven pre-kit posts publish no sidecar, so the page text is the only evidence."""
    docs, posts = _repo(tmp_path, monkeypatch)
    (posts / "slug" / "widgets.js").write_text(BUNDLE)
    (docs / "posts" / "slug" / "index.html").write_text(
        "<script>an older bundle</script>"
    )
    assert dq.main(["widget", "slug"]) == 1
    out = capsys.readouterr().out
    assert "inline bundle" in out and "STALE" in out and "MISSING" in out


def test_widget_passes_when_the_inline_bundle_is_current(tmp_path, monkeypatch, capsys):
    docs, posts = _repo(tmp_path, monkeypatch)
    (posts / "slug" / "widgets.js").write_text(BUNDLE)
    (docs / "posts" / "slug" / "index.html").write_text(f"<script>{BUNDLE}</script>")
    assert dq.main(["widget", "slug"]) == 0
    assert "current" in capsys.readouterr().out


def test_widget_marker_survives_the_inline_escape(tmp_path, monkeypatch, capsys):
    """The print cell rewrites `</` to `<\\/`, so a marker must be escaped to match."""
    docs, posts = _repo(tmp_path, monkeypatch)
    body = "\n".join(
        f"const closing{i} = '</div> and more text to pad the line';" for i in range(40)
    )
    (posts / "slug" / "widgets.js").write_text(body)
    (docs / "posts" / "slug" / "index.html").write_text(
        "<script>" + body.replace("</", "<\\/") + "</script>"
    )
    assert dq.main(["widget", "slug"]) == 0


def test_count_does_not_double_count_the_overlap(tmp_path, monkeypatch, capsys):
    """A match inside the carried-over window was already counted last round."""
    docs, _ = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(dq, "CHUNK", 1000)
    monkeypatch.setattr(dq, "OVERLAP", 256)
    p = docs / "posts" / "slug" / "index.html"
    p.write_text("a" * 900 + "NEEDLE" + "b" * 3000)
    dq.main(["-F", "count", "NEEDLE", "docs/posts/slug/index.html"])
    assert capsys.readouterr().out.startswith("1 ")
    p.write_text(("NEEDLE" + "z" * 94) * 50)
    dq.main(["-F", "count", "NEEDLE", "docs/posts/slug/index.html"])
    assert capsys.readouterr().out.startswith("50 ")


def test_fixed_flag_after_the_subcommand(tmp_path, monkeypatch, capsys):
    docs, _ = _repo(tmp_path, monkeypatch)
    (docs / "posts" / "slug" / "index.html").write_text("a.b")
    dq.main(["count", "-F", ".", "docs/posts/slug/index.html"])
    assert capsys.readouterr().out.startswith("1 ")  # literal dot, not regex


def test_files_refuses_a_glob_that_escapes(tmp_path, monkeypatch):
    """A glob is not a licence to leave the repo; the guard aborts loudly."""
    _repo(tmp_path, monkeypatch)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    try:
        dq.main(["-F", "files", "secret", "../outside.txt"])
    except SystemExit as exc:
        assert "outside the repo" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a refusal")


def test_refuses_a_path_outside_the_repo(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    try:
        dq.main(["exists", "../../../etc/hosts"])
    except SystemExit as exc:
        assert "outside the repo" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a refusal")


def test_widget_marker_matches_an_unescaped_page(tmp_path, monkeypatch):
    """Not every print cell rewrites `</`; a raw page must still verify."""
    docs, posts = _repo(tmp_path, monkeypatch)
    body = "\n".join(
        f"const closing{i} = '</strong> and more text to pad the line';"
        for i in range(40)
    )
    (posts / "slug" / "widgets.js").write_text(body)
    (docs / "posts" / "slug" / "index.html").write_text(f"<script>{body}</script>")
    assert dq.main(["widget", "slug"]) == 0


def test_widget_flags_a_kit_post_the_page_never_loads(tmp_path, monkeypatch, capsys):
    """An identical published sidecar is no use if the page does not link it."""
    docs, posts = _repo(tmp_path, monkeypatch)
    (posts / "slug" / "widgets.js").write_text(BUNDLE)
    (docs / "posts" / "slug" / "widgets.js").write_text(BUNDLE)
    (docs / "posts" / "slug" / "index.html").write_text("<html>no script tag</html>")
    assert dq.main(["widget", "slug"]) == 1
    out = capsys.readouterr().out
    assert "identical" in out and "page loads it: NO" in out and "STALE" in out


def test_fixed_flag_before_the_subcommand(tmp_path, monkeypatch, capsys):
    """A subparser default must not clobber -F given ahead of the subcommand."""
    docs, _ = _repo(tmp_path, monkeypatch)
    (docs / "posts" / "slug" / "index.html").write_text("a(b")
    # an invalid regex, so this only passes if the pattern is treated literally
    dq.main(["-F", "count", "a(b", "docs/posts/slug/index.html"])
    assert capsys.readouterr().out.startswith("1 ")


def test_bad_regex_is_a_message_not_a_traceback(tmp_path, monkeypatch):
    docs, _ = _repo(tmp_path, monkeypatch)
    (docs / "posts" / "slug" / "index.html").write_text("x")
    try:
        dq.main(["count", "a(b", "docs/posts/slug/index.html"])
    except SystemExit as exc:
        assert "bad pattern" in str(exc) and "-F" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a refusal")
