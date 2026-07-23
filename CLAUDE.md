# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal ML/data blog ("Synthetic Musings") built with [Quarto](https://quarto.org/) and published to GitHub Pages at https://project-delphi.github.io/ml-blog/. Each post is a self-contained `.qmd` or `.ipynb` file under `posts/<slug>/`; the rendered static site lives in `docs/` and is committed directly to `main` (there is no CI workflow — `docs/` must be rendered and committed locally as part of any content change).

## Commands

- Render one post (safe, targeted): `quarto render posts/<slug>/index.qmd` (or `index.ipynb`). This also refreshes the homepage listing (`index.qmd` → `docs/index.html`) but leaves every other already-built page untouched.
- Render the whole site: `quarto render .` or `make quatro`. **This deletes and rebuilds the entire `docs/` output directory**, re-executing *every* post's code — including posts that need kernels/dependencies (R, PySpark, HuggingFace downloads, etc.) that may not be set up in the current environment. Don't run this for a single-post change; only do it deliberately.
- Preview: `quarto preview` (or `make preview`). Previewing the whole project also indexes/executes every post the first time, so prefer `quarto preview posts/<slug>/index.qmd` or serve the already-built `docs/` folder statically (e.g. `python -m http.server` from `docs/`) when you just need to eyeball one post.
- `make venv` / `make install`: create `.venv` and `pip install .` (the base dev/lint toolchain from `pyproject.toml` — this does *not* include per-post ML dependencies, see below).
- Lint/format tooling (black, ruff, mypy, pyupgrade, commitizen, codespell) is configured in `.pre-commit-config.yaml` and `pyproject.toml` (`[tool.ruff]`, `[tool.pydoclint]`, `[tool.codespell]`) but hooks aren't installed by default — run manually with `pre-commit run --all-files` if needed.

## Architecture

**Posts are independent, dependency-isolated documents.** `pyproject.toml`'s dependencies are only dev/lint tooling (black, ruff, mypy, jupyter, etc.) — no numpy/sklearn/torch/etc. Any post with real compute dependencies gets its own dedicated virtualenv at the repo root (gitignored, e.g. `.venv-tda`, `.venv-tda-svm`), registered as a **named Jupyter kernel** (`python -m ipykernel install --user --name <kernel-name>`), and the post's frontmatter pins execution to it via `jupyter: <kernel-name>`. When adding a new code-heavy post: create `.venv-<slug>`, `pip install` only what that post needs, register the kernel, set `jupyter: <kernel-name>` in frontmatter, and render/execute through that venv specifically.

**Never install into or execute with the system Python.** Every `pip install`, `ipykernel install`, and render must go through a venv — either activate it (`source .venv-<slug>/bin/activate`) or, more robustly in non-interactive shells, invoke the venv's interpreter directly: `.venv-<slug>/bin/python -m pip install ...` and `.venv-<slug>/bin/python -m ipykernel install --user --name <kernel-name>`. A bare `python`/`pip` at the shell prompt is the system Python and will pollute it (or fail on externally-managed environments); the registered kernel must point at the venv's interpreter, not the system one — verify with `jupyter kernelspec list` and check the kernel's `kernel.json` `argv` path if a render picks up the wrong environment.

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
