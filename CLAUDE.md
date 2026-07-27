# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal ML/data blog ("Synthetic Musings") built with [Quarto](https://quarto.org/) and published to GitHub Pages at https://project-delphi.github.io/ml-blog/. Each post is a self-contained `.qmd` or `.ipynb` file under `posts/<slug>/`; the rendered static site lives in `docs/` and is served from `main` (there is no CI workflow — `docs/` must be rendered locally and committed *in the same commit/PR* as the source change, otherwise the published site drifts from the source).

## Commands

- Render one post (safe, targeted): `quarto render posts/<slug>/index.qmd` (or `index.ipynb`). This also refreshes the homepage listing (`index.qmd` → `docs/index.html`) but leaves every other already-built page untouched.
- Render the whole site: `quarto render .` or `make quatro`. **This deletes and rebuilds the entire `docs/` output directory**, re-executing *every* post's code — including posts that need kernels/dependencies (R, PySpark, HuggingFace downloads, etc.) that may not be set up in the current environment. Don't run this for a single-post change; only do it deliberately.
- Preview: `quarto preview` (or `make preview`). Previewing the whole project also indexes/executes every post the first time, so prefer `quarto preview posts/<slug>/index.qmd` or serve the already-built `docs/` folder statically (e.g. `python -m http.server` from `docs/`) when you just need to eyeball one post.
- `make install`: `uv sync` the base dev/lint toolchain into `.venv`, creating the venv if absent (this does *not* include per-post ML dependencies, see below). It installs from the committed root `uv.lock` rather than re-resolving, so every clone gets identical versions, and it *prunes* anything not in the lock — don't hand-install extras into `.venv` and expect them to survive. `make venv` still exists for a bare venv, but `make install` doesn't need it. `make lock` regenerates `uv.lock` from `pyproject.toml` without touching `.venv`, for reviewing a dependency change before installing it. `pyproject.toml` sets `requires-python = ">=3.11"`; lowering it re-forks the lock across interpreter versions, so raise it rather than lower it.
- Lint/format tooling (black, ruff, mypy, pyupgrade, commitizen, codespell) is configured in `.pre-commit-config.yaml` and `pyproject.toml` (`[tool.ruff]`, `[tool.pydoclint]`, `[tool.codespell]`) but hooks aren't installed by default — run manually with `pre-commit run --all-files` if needed.

## Workflow

**Never commit to `main`.** Every change — new post, edit, fix, even a one-line typo — goes through: feature branch → commit (source *and* the re-rendered `docs/` output together) → push → open a PR with a real description of what changed and why → PR review → merge. `main` only ever advances via a merged PR. The `ship-pr` skill automates this loop.

This is enforced: `.claude/settings.json` registers a `PreToolUse` hook on `Bash` running `.claude/hooks/block-main-commit.sh`, which denies any command reaching `git commit` while HEAD is on `main`/`master`. Creating a branch first passes (`git switch -c rk/foo && git commit …`); switching onto `main` is not an escape hatch. Run `.claude/hooks/test-block-main-commit.sh` after touching the hook — it asserts the full allow/block matrix. To override deliberately, commit from your own terminal or disable the hook via `/hooks`.

**Hand back a localhost preview link whenever a unit of work is complete** — a clickable URL, not just "done":

- `quarto preview posts/<slug>/index.qmd`, reporting the URL it prints (typically `http://localhost:<port>/`) — prefer this single-post form, since a whole-project preview indexes and executes every post; or
- serve the already-built output: `python -m http.server 8000 --directory docs` → `http://localhost:8000/posts/<slug>/index.html`.

**Kill every running preview server once the change is merged**, so stale previews don't linger on their ports:

```bash
pkill -f "quarto.*preview"        # quarto preview servers
# Static docs servers. Matching argv misses `cd docs && python -m http.server`
# (no "docs" in its command line), so match on cwd — which also spares servers
# belonging to other projects. Run from the repo root.
pgrep -f "http\.server" | while read -r pid; do
  cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')
  case "$cwd" in "$PWD"|"$PWD"/*) kill "$pid";; esac
done
```

Confirm nothing survives — `pgrep -fl "quarto preview"` prints nothing, and `lsof -nP -iTCP -sTCP:LISTEN | grep -i python` lists no server rooted in this repo — and say so in the wrap-up.

## Writing conventions

**Give every post a spine.** A post is one argument, not a pile of sections. Name the core idea in a sentence before writing; every section advances it, and the reader always knows where they stand relative to it.

- **Open on the core idea and the stakes** — the question, why it's non-obvious, what changes once the reader knows. No "in this post we will…" preamble.
- **Headings state claims, not topics** — `## Efron's bootstrap is a weighted bootstrap in disguise`, not `## Background`, so the ToC reconstructs the argument.
- **Each section earns the next.** If two could swap without damage, merge or cut one. `###` is for steps within one idea, not new ideas.
- **Stitch every seam.** A section's first sentence links back to the previous result or the core idea; its last names the unresolved thing the next section answers — the gap, not the mechanics ("next, some code"). Same for code and figures: a sentence before saying what it will show, one after saying what happened. If an opening sentence reads identically with the previous section deleted, the seam isn't stitched.
- **Close by returning to the core idea** — restate the opening claim now that it's earned, what it buys in practice, and where it stops holding. Not a summary of sections.
- **Caveats inline**, as their own short section where the objection occurs (`## Caveat: the uniform is the posterior, not the prior`) — not a "Limitations" bin at the end.

**Always situate the data.** Any post that uses, plots, or even mentions a dataset must tell the reader where it came from before doing anything with it. Cover, in prose (not a bullet checklist bolted on):

- **Provenance** — what the dataset actually is, and a link/citation to the source.
- **Collector and motive** — who gathered it, under what program or institution, and why they went to the trouble. Instrument, survey, scrape, simulation: say which.
- **Objective** — what question is being asked *of this data in this post*, and what the target/label means in the real world.
- **Downstream impact** — what a decision made from this analysis would actually affect (a diagnosis, a loan, a forecast, a research conclusion), and what it costs to be wrong.
- **Why this method** — what specific property of *this* data (sample size, noise, class imbalance, heavy tails, hierarchy, missingness, small-n uncertainty) makes the technique in the post the right tool, and which quantity we care about it improves.

Synthetic data is not exempt: state that it is synthetic, give the generating process, and explain what real-world situation it is standing in for and why simulating is preferable to a real dataset here.

## Architecture

**Posts are independent, dependency-isolated documents.** `pyproject.toml` carries only dev/lint tooling — no numpy/sklearn/torch. Each code-heavy post gets its own gitignored `.venv-<slug>` at the repo root (e.g. `.venv-tda`, `.venv-tda-svm`), registered as a named Jupyter kernel, and pins execution to it via `jupyter: <kernel-name>` in frontmatter. **Never install into or execute with the system Python** — a bare `python`/`pip` is the system one and will pollute it (or fail on externally-managed environments). Always target the venv explicitly rather than relying on activation:

```bash
uv venv .venv-<slug>
uv pip install --python .venv-<slug>/bin/python <post deps> ipykernel jupyter nbclient nbformat pyyaml
.venv-<slug>/bin/python -m ipykernel install --user --name <kernel-name>
QUARTO_PYTHON="$(pwd)/.venv-<slug>/bin/python" quarto render posts/<slug>/index.qmd
uv pip freeze --python .venv-<slug>/bin/python > posts/<slug>/requirements.txt
```

Each line above fixes a real failure. `ipykernel` alone is not enough — without Quarto's execution stack (`jupyter nbclient nbformat pyyaml`) a render dies with `ModuleNotFoundError: No module named 'yaml'` from Quarto's `jupyter.py` shim. And `quarto render` finds its kernel through whatever Python it defaults to, which usually cannot see a `--user`-registered kernel: without `QUARTO_PYTHON` it fails with `ERROR: Jupyter kernel '<name>' not found. Known kernels: python3`. If a render picks up the wrong environment, `jupyter kernelspec list` and the kernel's `argv[0]` say which interpreter it actually resolved to.

That last line is not optional: every post with a dedicated venv carries its exact versions in `posts/<slug>/requirements.txt`, because `_freeze/` is gitignored while `docs/` is committed — without it a re-render on drifted dependencies silently changes published output. Venvs predating uv still work, driven by `.venv-<slug>/bin/python -m pip` (a `uv venv` has no `pip`); recreate one in place from its lockfile to migrate it — kernel specs store an absolute path, so reusing the directory name keeps the registration valid.

Posts that only *display* code (all cells `#| eval: false` — e.g. `posts/langgraph-vs-llamaindex`) need no dedicated venv; they pin `jupyter: blog-base`, a shared kernel over the base `.venv` (`make install && make kernel`). Quarto still needs *some* working kernel to structurally process `{python}` cells even when nothing runs, so register it once on a fresh clone.

**Frontmatter conventions** (see any existing post for a template):
```yaml
title: "..."
author: "Ravi Kalia"
date: "YYYY-MM-DD"
categories: [Some, Categories]
image: "./cover.png"
tags: [some, tags]
jupyter: <kernel-name>       # only for code-heavy posts pinned to a dedicated venv
format:
  html:
    toc: true
    code-fold: true
```
The body conventionally opens with `![Title](./cover.png)` echoing the frontmatter `image`. `.ipynb` posts embed this same YAML in a raw first cell instead of a `.qmd` header.

**Cover images**: every post directory should have `./cover.png`. For posts without a natural content-derived cover, the house style is a solid `#4A3AA7` purple card with a translucent rounded category-badge pill top-left (e.g. "ML THEORY & MATH"), a white triple-ring logomark, and bold centered title text (see `posts/topological-data-analysis-clustering/cover.png`). The site favicon (`favicon.png` at repo root, declared via `_quarto.yml`'s `website.favicon`) reuses the same triple-ring mark.

**`posts/_metadata.yml`** applies `freeze: auto` (cache computed output so re-rendering an unchanged post is a no-op) and `title-block-banner: true` to every post.

**`_quarto.yml`** sets `output-dir: docs` and excludes `notes/` from rendering (`render: ["*.qmd", "*.ipynb", "!notes/"]`) — that's a scratch/drafts area, not published content.
