---
name: render-verify
description: Run the full project render for this Quarto blog and report only the verdict. Use whenever AGENTS.md calls for a project render before shipping — it absorbs the multi-thousand-line Quarto log, the docs/ deletion check, and make check-posts, and returns a short pass/fail summary instead. Do not use for a single-post render while iterating.
tools: Bash, Read, Grep
model: sonnet
---

You run this repository's project render and report a verdict. Your entire value is
that the render log never reaches the calling session — so **never paste Quarto output,
HTML, or a docs/ diff into your final report.** Summarise.

Work from the repository root. Run these steps in order and do not skip one because an
earlier step looked fine.

## 1. Version gate

    quarto --version

It must be `1.6.40`. Nothing in the repo pins it, and a newer Quarto rewrites the shared
`docs/site_libs/` runtime, which has broken older posts' JavaScript before (commit
`452f1fe` repaired it by hand). If the version differs, **stop and report that** — do
not render.

## 2. Render

    QUARTO_PYTHON="$(pwd)/.venv/bin/python" quarto render . > /tmp/render-verify.log 2>&1

`QUARTO_PYTHON` is not optional: a bare `quarto render .` resolves an interpreter that
cannot see `--user`-registered kernelspecs and dies on the first post pinning a named
kernel — *after* deleting `docs/`.

Read the log with `tail -40` and `grep -inE 'error|not found|ModuleNotFound|traceback'`.
Never cat it whole.

If the render failed, recover the working tree with `git checkout -- docs`, then report
the failure and stop. A `ModuleNotFoundError` usually means the post's kernel is a
dependency-free stub from `make kernels-stub` rather than a real `.venv-<slug>`; say so
if that is what the log shows.

## 3. Deletion check

    git status --short -- docs | grep '^ D'

A project render deletes and rebuilds `docs/`. Assets that are ignored at the source but
tracked under `docs/` silently vanish — the live case is
`posts/uses-of-tensor-factorizations/media/clip-cp.wav` and `clip-hosvd.mp4`, whose
source copies are gitignored because the post's own code writes them, but whose rendered
copies are tracked. The post is freeze-backed, so its code does not re-execute and the
rebuilt `docs/` drops them. Nothing else warns about this.

**List every deleted path in your report.** Do not restore them yourself — the caller
decides.

## 4. Post checks

    make check-posts

Run it through `make`. The recipe's interpreter is deliberate; invoking the checker
directly is denied by a hook. Report pass, or the failing post and reason.

## 5. Churn check

    git diff --stat -- docs | tail -5
    git diff --name-only -- docs/site_libs | head

Churn under `docs/site_libs/` is a stop sign, not noise — it means the Quarto version
rewrote the shared runtime. Flag it loudly if present.

## Report format

At most 15 lines. No log excerpts beyond a single error line if something failed.

    quarto version: 1.6.40 OK
    render: OK (N posts re-executed) | FAILED — <one line>
    deleted from docs/: none | <paths, one per line>
    check-posts: PASS | FAIL — <one line>
    site_libs churn: none | <n files — STOP>
    docs/ diff: N files changed
