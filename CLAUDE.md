# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal ML/data blog ("Synthetic Musings") built with [Quarto](https://quarto.org/) and published to GitHub Pages at https://project-delphi.github.io/ml-blog/. Each post is a self-contained `.qmd` or `.ipynb` file under `posts/<slug>/`; the rendered static site lives in `docs/` and is served from `main` (there is no CI workflow — `docs/` must be rendered locally and committed *in the same commit/PR* as the source change, otherwise the published site drifts from the source).

## Commands

- Render one post (safe, targeted): `quarto render posts/<slug>/index.qmd` (or `index.ipynb`). This also refreshes the homepage listing (`index.qmd` → `docs/index.html`) but leaves every other already-built page untouched.
- Render the whole site: `quarto render .` or `make quatro`. **This deletes and rebuilds the entire `docs/` output directory**, re-executing *every* post's code — including posts that need kernels/dependencies (R, PySpark, HuggingFace downloads, etc.) that may not be set up in the current environment. Don't run this for a single-post change; only do it deliberately.
- Preview: `quarto preview` (or `make preview`). Previewing the whole project also indexes/executes every post the first time, so prefer `quarto preview posts/<slug>/index.qmd` or serve the already-built `docs/` folder statically (e.g. `python -m http.server` from `docs/`) when you just need to eyeball one post.
- `make venv` / `make install`: create `.venv` and `pip install .` (the base dev/lint toolchain from `pyproject.toml` — this does *not* include per-post ML dependencies, see below).
- Lint/format tooling (black, ruff, mypy, pyupgrade, commitizen, codespell) is configured in `.pre-commit-config.yaml` and `pyproject.toml` (`[tool.ruff]`, `[tool.pydoclint]`, `[tool.codespell]`) but hooks aren't installed by default — run manually with `pre-commit run --all-files` if needed.

## Workflow

**Never commit to `main`.** Every change — new post, edit, fix, even a one-line typo — goes through: feature branch → commit (source *and* the re-rendered `docs/` output together) → push → open a PR with a real description of what changed and why → PR review → merge. `main` only ever advances via a merged PR. The `ship-pr` skill automates this loop.

This is enforced, not just documented: `.claude/settings.json` registers a `PreToolUse` hook on `Bash` that runs `.claude/hooks/block-main-commit.sh`, which denies any command reaching `git commit` while HEAD is on `main`/`master`. A command that *creates* a branch first passes (`git switch -c rk/foo && git commit …`); a bare `git switch main && git commit` does not — switching onto `main` is not an escape hatch. Switching to an *existing* branch and committing is also blocked while on `main`, which is deliberate: it fails safe, and the deny message says what to do. Run `.claude/hooks/test-block-main-commit.sh` after touching the hook — it asserts the full allow/block matrix. To override deliberately, commit from your own terminal, or disable the hook via `/hooks`.

**Hand back a localhost preview link whenever a unit of work is complete.** Don't just say "done" — give a clickable URL the change can be eyeballed at, e.g.:

- `quarto preview posts/<slug>/index.qmd` and report the URL it prints (typically `http://localhost:<port>/`), or
- serve the already-built output: `python -m http.server 8000 --directory docs` → `http://localhost:8000/posts/<slug>/index.html`.

Prefer the single-post form; a whole-project preview indexes and executes every post.

**Kill every running Quarto preview server once the change is shipped and merged.** After the PR merges, tear the servers down so stale previews don't linger on their ports:

```bash
pkill -f "quarto.*preview"        # quarto preview servers
pkill -f "http.server .*docs"     # any static docs server started for previewing
```

Confirm nothing survives (`pgrep -fl "quarto preview"` should print nothing) and say so in the wrap-up.

## Writing conventions

**Always situate the data.** Any post that uses, plots, or even mentions a dataset must tell the reader where it came from before doing anything with it. Cover, in prose (not a bullet checklist bolted on):

- **Provenance** — what the dataset actually is, and a link/citation to the source.
- **Collector and motive** — who gathered it, under what program or institution, and why they went to the trouble. Instrument, survey, scrape, simulation: say which.
- **Objective** — what question is being asked *of this data in this post*, and what the target/label means in the real world.
- **Downstream impact** — what a decision made from this analysis would actually affect (a diagnosis, a loan, a forecast, a research conclusion), and what it costs to be wrong.
- **Why this method** — what specific property of *this* data (sample size, noise, class imbalance, heavy tails, hierarchy, missingness, small-n uncertainty) makes the technique in the post the right tool, and which quantity we care about it improves.

Synthetic data is not exempt: state that it is synthetic, give the generating process, and explain what real-world situation it is standing in for and why simulating is preferable to a real dataset here.

## Architecture

**Posts are independent, dependency-isolated documents.** `pyproject.toml`'s dependencies are only dev/lint tooling (black, ruff, mypy, jupyter, etc.) — no numpy/sklearn/torch/etc. Any post with real compute dependencies gets its own dedicated virtualenv at the repo root (gitignored, e.g. `.venv-tda`, `.venv-tda-svm`), registered as a **named Jupyter kernel** (`python -m ipykernel install --user --name <kernel-name>`), and the post's frontmatter pins execution to it via `jupyter: <kernel-name>`. When adding a new code-heavy post: create `.venv-<slug>`, `pip install` only what that post needs **plus Quarto's own execution stack** (`jupyter nbclient nbformat pyyaml` — `ipykernel` alone is not enough; without them a render dies with `ModuleNotFoundError: No module named 'yaml'` from Quarto's `jupyter.py` shim), register the kernel, set `jupyter: <kernel-name>` in frontmatter, and render/execute through that venv specifically.

**Never install into or execute with the system Python.** Every `pip install`, `ipykernel install`, and render must go through a venv — either activate it (`source .venv-<slug>/bin/activate`) or, more robustly in non-interactive shells, invoke the venv's interpreter directly: `.venv-<slug>/bin/python -m pip install ...` and `.venv-<slug>/bin/python -m ipykernel install --user --name <kernel-name>`. A bare `python`/`pip` at the shell prompt is the system Python and will pollute it (or fail on externally-managed environments); the registered kernel must point at the venv's interpreter, not the system one — verify with `jupyter kernelspec list` and check the kernel's `kernel.json` `argv` path if a render picks up the wrong environment.

**Rendering a code-heavy post from a non-interactive shell:** `quarto render` discovers its kernel through whatever Python it finds by default, which usually does *not* see a `--user`-registered kernel — the render then fails with `ERROR: Jupyter kernel '<name>' not found. Known kernels: python3`. Point Quarto at the post's venv explicitly: `QUARTO_PYTHON="$(pwd)/.venv-<slug>/bin/python" quarto render posts/<slug>/index.qmd` (equivalently, activate the venv first). This is why the per-post venv must carry the full Jupyter execution stack noted above, not just `ipykernel`.

For posts that only *display* code (all cells `#| eval: false`, nothing actually executes — e.g. `posts/langgraph-vs-llamaindex`), a dedicated venv is unnecessary; they pin `jupyter: blog-base`, a shared kernel over the base `.venv` (`make venv && make install`, then `python -m ipykernel install --user --name blog-base`). Quarto still needs *some* working kernel to structurally process `{python}` cells even when nothing runs, so register `blog-base` once before rendering this kind of post on a fresh clone.

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
