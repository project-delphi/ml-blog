# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal ML/data blog ("Synthetic Musings") built with [Quarto](https://quarto.org/) and published to GitHub Pages at https://project-delphi.github.io/ml-blog/. Each post is a self-contained `.qmd` or `.ipynb` under `posts/<slug>/`; the rendered site lives in `docs/` and is served from `main`. There is no CI workflow — `docs/` must be rendered locally and committed *in the same PR* as the source change, or the published site drifts from the source.

## Commands

- **Render one post**: `quarto render posts/<slug>/index.qmd`. Narrow blast radius, but it **always executes that post's code** — `freeze` is honoured only on a *project* render. Needs the post's real venv.
- **Render the whole site**: `QUARTO_PYTHON="$(pwd)/.venv/bin/python" quarto render .` (`make quatro` runs the bare form). Deletes and rebuilds `docs/`, but *respects* `freeze: auto` — it re-executes only posts whose **source md5 changed** since their `_freeze/` record was written, which today is none. The right tool for anything site-wide: nav, theme, `_quarto.yml`, a stale `search.json`.
  - `QUARTO_PYTHON` is not optional. A bare `quarto render .` resolves a Python that cannot see `--user`-registered kernelspecs and dies on the first post pinning a named kernel — after it has already deleted `docs/`. Recover with `git checkout -- docs`.
  - Kernelspecs resolve while Quarto *indexes* the project, before it consults `_freeze/`. A missing kernel fails the whole render, frozen output or not. On a fresh clone run `make kernels-stub` first.
- **Preview**: prefer `quarto preview posts/<slug>/index.qmd` — a whole-project preview indexes every post. Or serve the built output: `python -m http.server 8000 --directory docs`.
- **Checks** (there is no test suite): `make check-posts` runs `scripts/check_posts.py` — stdlib-only, so it works on any interpreter — verifying that code posts pin a kernel + `requirements.txt`, that every pinned kernel appears in `make kernels-stub`, and that no post's frozen output has drifted from its source. Run it before any full render. `.claude/hooks/test-block-main-commit.sh` asserts the commit hook's allow/block matrix; run it after touching that hook.
- **`make install`**: `uv sync` the base dev/lint toolchain into `.venv` (no per-post ML deps). Installs from the committed `uv.lock` rather than re-resolving, and *prunes* anything not in the lock — hand-installed extras will not survive. `make lock` regenerates the lock without touching `.venv`. `requires-python = ">=3.11"`; raise it rather than lower it, since lowering re-forks the lock across interpreter versions.
- **Lint**: black, ruff, mypy, pyupgrade, commitizen, codespell are configured in `.pre-commit-config.yaml` and `pyproject.toml`, but hooks aren't installed — run `pre-commit run --all-files` manually.
- `.ipynb` posts are never executed by Quarto; their stored cell outputs are used as-is, which is why none of the 8 have a `_freeze/` record.

## Workflow

**Never commit to `main`.** Every change — new post, edit, one-line typo — goes: feature branch → commit (source *and* re-rendered `docs/` together) → push → PR with a real description → review → merge. The `ship-pr` skill automates this.

This is enforced. `.claude/settings.json` registers a `PreToolUse` hook on `Bash` running `.claude/hooks/block-main-commit.sh`, which denies any command reaching `git commit` while HEAD is on `main`/`master`. Branching first passes; switching onto `main` is not an escape hatch. To override deliberately, commit from your own terminal or disable the hook via `/hooks`.

**Hand back a localhost preview link whenever a unit of work is complete** — a clickable URL, not just "done": the URL `quarto preview posts/<slug>/index.qmd` prints, or `http://localhost:8000/posts/<slug>/index.html` from the static server above.

**Put that link last.** It goes at the very end of the reply, after the summary, caveats and any list of what was skipped — the final thing on screen, on its own line, so it is never buried mid-message and never needs scrolling back to find. One link per reply; if several posts changed, link the one the work was about and mention the others by slug.

Once the change is merged the preview server is gone (see below), so there is no localhost link left to give. On that final reply, end with the published URL instead — `https://project-delphi.github.io/ml-blog/posts/<slug>/` — after the confirmation that no server survived. Say that it goes live once Pages rebuilds: the deploy is asynchronous, so that URL 404s for a minute or two after the merge commit lands.

**Kill every preview server once the change is merged**, then confirm nothing survives (`pgrep -fl "quarto preview"` silent; no Python listener rooted in this repo) and say so in the wrap-up:

```bash
pkill -f "quarto.*preview"
# Match static servers on cwd, not argv: `cd docs && python -m http.server` has
# no "docs" in its command line, and cwd-matching also spares other projects.
pgrep -f "http\.server" | while read -r pid; do
  cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')
  case "$cwd" in "$PWD"|"$PWD"/*) kill "$pid";; esac
done
```

## Writing conventions

For blog posts, read and follow `notes/blog-style.md` before writing.

**Give every post a spine.** A post is one argument, not a pile of sections. Name the core idea in a sentence before writing; every section advances it.

- **Open on the core idea and the stakes** — the question, why it's non-obvious, what changes once the reader knows. No "in this post we will…" preamble.
- **Headings state claims, not topics** — `## Efron's bootstrap is a weighted bootstrap in disguise`, not `## Background`, so the ToC reconstructs the argument.
- **Each section earns the next.** If two could swap without damage, merge or cut one. `###` is for steps within one idea, not new ideas.
- **Stitch every seam.** A section's first sentence links back to the previous result or the core idea; its last names the unresolved thing the next section answers — the gap, not the mechanics ("next, some code"). Same for code and figures: a sentence before saying what it will show, one after saying what happened. If an opening sentence reads identically with the previous section deleted, the seam isn't stitched.
- **Close by returning to the core idea** — restate the opening claim now that it's earned, what it buys, where it stops holding. Not a summary of sections.
- **Caveats inline**, as a short section where the objection occurs (`## Caveat: the uniform is the posterior, not the prior`) — not a "Limitations" bin at the end.

**Always situate the data.** Any post that uses, plots, or even mentions a dataset must say where it came from before doing anything with it — in prose, not a checklist bolted on:

- **Provenance** — what the dataset is, with a link or citation.
- **Collector and motive** — who gathered it, under what program, and why they went to the trouble. Instrument, survey, scrape, simulation: say which.
- **Objective** — what is being asked *of this data in this post*, and what the target means in the real world.
- **Downstream impact** — what a decision made from this analysis would affect (a diagnosis, a loan, a forecast, a research conclusion), and what being wrong costs.
- **Why this method** — what property of *this* data (sample size, noise, class imbalance, heavy tails, hierarchy, missingness, small-n uncertainty) makes the post's technique the right tool, and which quantity it improves.

Synthetic data is not exempt: say it is synthetic, give the generating process, and explain what real situation it stands in for and why simulating beats a real dataset here.

## Architecture

### Three tiers of post

Posts are dependency-isolated: `pyproject.toml` carries only dev/lint tooling, no numpy/sklearn/torch.

1. **Executes at render** — its own gitignored `.venv-<slug>` at the repo root, a named Jupyter kernel, `jupyter: <kernel-name>` in frontmatter, and a committed `posts/<slug>/requirements.txt`. 27 posts. Venv and kernel names are historical abbreviations and often do **not** match the slug (`convex-optimization-interior-point-methods` → `.venv-ipm`/`ipm-blog`; `sir-training-vs-calibration` → `.venv-sir`/`sir-blog`) — read the post's `jupyter:` field rather than guessing.
2. **Displays code only** — all cells `#| eval: false`. Pins `jupyter: blog-base`, a shared kernel over the base `.venv` (`make install && make kernel`). Four posts, all Claude-API ones: `claude-api-eval-pipeline`, `langgraph-vs-llamaindex`, `messages-api-streaming`, `structured-json-with-claude`. Quarto still needs *some* working kernel to structurally process `{python}` cells even when nothing runs. (`mermaid` and other diagram blocks go through Quarto's own filters and need no kernel.)
3. **Assets generated ahead of the render** — see § `posts/<slug>/src/` below.

### Building a post venv

**Never install into or execute with the system Python** — a bare `python`/`pip` is the system one and will pollute it (or fail on externally-managed environments). Target the venv explicitly rather than relying on activation:

```bash
uv venv .venv-<slug>
uv pip install --python .venv-<slug>/bin/python <post deps> ipykernel jupyter nbclient nbformat pyyaml
.venv-<slug>/bin/python -m ipykernel install --user --name <kernel-name>
QUARTO_PYTHON="$(pwd)/.venv-<slug>/bin/python" quarto render posts/<slug>/index.qmd
uv pip freeze --python .venv-<slug>/bin/python > posts/<slug>/requirements.txt
```

Every line fixes a real failure. `ipykernel` alone is not enough — without Quarto's execution stack (`jupyter nbclient nbformat pyyaml`) a render dies with `ModuleNotFoundError: No module named 'yaml'` from Quarto's `jupyter.py` shim. Without `QUARTO_PYTHON`, `quarto render` resolves a Python that cannot see a `--user`-registered kernel and fails with `Jupyter kernel '<name>' not found. Known kernels: python3`. If a render picks up the wrong environment, `jupyter kernelspec list` and the kernel's `argv[0]` say which interpreter it actually resolved to.

The freeze is not optional: `requirements.txt` is what lets a re-render be checked against the versions that produced the published output. **Also add the new kernel name to `kernels-stub` in the `Makefile`** — a name missing there fails a project render on any fresh clone, since kernelspecs resolve before `_freeze/` is consulted. `make check-posts` enforces this.

Venvs predating uv still work, driven by `.venv-<slug>/bin/python -m pip` (a `uv venv` has no `pip`); recreate one in place from its lockfile to migrate — kernel specs store an absolute path, so reusing the directory name keeps the registration valid.

### The Hugging Face posts share two venvs, not thirteen

Thirteen posts (`datasets`, `hub`, `tokenizers`, `peft`, `transformers-library`, `hugging-face-evaluate-library`, `optimum`, `setfit`, `diffusers`, `classifying_text_chunks`, `langchain`, `evaluation-metrics`, `soft-vs-hard-prompts`) share ~90% of their dependency closure, most of it `torch`. Each venv is ~1.2 GB and `uv` copies rather than hardlinks here, so thirteen would be ~15 GB of near-identical wheels. They pin one of two shared kernels:

- **`huggingface-blog`** over `.venv-huggingface` — `transformers` 5.x. Eleven posts.
- **`huggingface-t4-blog`** over `.venv-huggingface-t4` — `transformers` 4.57.x. Two posts, `optimum` and `setfit`, only because those packages cap it. `optimum-onnx` declares `transformers<4.58.0`; `setfit` declares no upper bound but imports `transformers.training_args.default_logdir`, which 5.x removed — so it *resolves* against 5.x and then fails at import. A resolver-only check misses that; import the packages before trusting a resolve.

`requirements.txt` records a freeze, not intent, so the package sets live here:

```bash
uv venv .venv-huggingface --python 3.12
uv pip install --python .venv-huggingface/bin/python \
  torch transformers datasets tokenizers huggingface_hub evaluate peft \
  sentence-transformers diffusers accelerate safetensors \
  rouge_score sacrebleu nltk absl-py bert_score scikit-learn numpy pandas matplotlib Pillow \
  torchvision opencv-python-headless \
  pypdf langchain langchain-core langchain-text-splitters langchain-huggingface \
  ipykernel jupyter nbclient nbformat pyyaml
.venv-huggingface/bin/python -m ipykernel install --user --name huggingface-blog

uv venv .venv-huggingface-t4 --python 3.12
uv pip install --python .venv-huggingface-t4/bin/python \
  "optimum-onnx[onnxruntime]" setfit transformers torch datasets evaluate \
  sentence-transformers scikit-learn sentencepiece \
  ipykernel jupyter nbclient nbformat pyyaml
.venv-huggingface-t4/bin/python -m ipykernel install --user --name huggingface-t4-blog
```

`setfit` is deliberately absent from the first set: it resolves fine there and then breaks the kernel at import.

The trade: for these thirteen, `requirements.txt` is a copy of a shared freeze, so bumping a dependency for one post changes the versions recorded for every post on that kernel. `_freeze/` hashes source only, so nothing silently re-executes — but committed versions and the versions that produced a post's output can drift apart. **Re-render every post on a kernel when you bump that kernel's venv.**

### `posts/<slug>/src/` — assets built outside the render

Fourteen posts carry a `src/` directory of scripts that run *ahead of* the render, with their output committed. Most hold a `make_cover.py`; some go further — `metagenomics/src/make_figs.py`, `open-source-tts-history/src/make_audio.py` (which writes the committed `audio/*.mp3`), and `bayesian-bootstrap/src/` with a full module set (`bootstrap.py`, `data.py`, `mnist_experiment.py`, `export_widget_data.py`).

These scripts are **not** executed by Quarto and are not covered by `_freeze/` or `make check-posts`. Regenerating an asset means running the script yourself and committing the result. They may also carry their own dependency pin (`open-source-tts-history/src/requirements-audio.txt`) and their own venv with no kernel and no post pinning it — `.venv-kokoro` exists solely to run `make_audio.py`, which is why it looks orphaned next to the kernel-backed venvs.

### Why `_freeze/` is committed

`_freeze/` is **tracked**, not ignored. Five posts still predate the venv-per-post convention — `data-types`, `features-importance-after-clustering`, `poor-persons-bayesian`, `post-with-code`, `working-with-quarto` — with no pinned kernel, no pinned versions, and dependencies pinned to nothing. Their committed `_freeze/` record is the only reproducible copy of what they compute. It is also what would let the site render in CI with no ML dependencies installed.

`scripts/check_posts.py` grandfathers exactly those five in `LEGACY_NO_ENV`. **Shrink that set, never grow it.** A second exemption, `STALE_FREEZE_OK` (`pinecone-vs-weaviate`, `pyspark`), covers posts whose frozen output is knowingly stale but inert because they have no code cells at all; the checker cancels the exemption automatically if code cells later appear.

**Editing a legacy post breaks it.** Quarto keys frozen output on an md5 of the source, so *any* edit — even a one-word prose fix — invalidates the record and makes the next project render try to execute a post that cannot execute. If you touch one, build it a venv + kernel + `requirements.txt` first and delete its `LEGACY_NO_ENV` entry. This is not hypothetical: the cover-image commits edited 11 sources without re-rendering and left the freeze cache silently stale for months.

### One post calls a live API

`llm-agents-from-first-principles` runs an LLM agent against Groq's free tier, so re-executing it needs `GROQ_API_KEY` — its `.venv-llm-agents` holds only Quarto's execution stack, since `agent.py` is stdlib-only. A *project* render never re-executes it, so a keyless clone renders the site fine; only a single-document render needs the key. Its captured transcripts are one sample from a stochastic policy, not a reproducible fixed point: re-rendering legitimately changes published output even at `temperature=0`. Don't re-render it to "refresh" anything, and never hand-edit the transcripts.

### Frontmatter and assets

```yaml
title: "..."
author: "Ravi Kalia"
date: "YYYY-MM-DD"
categories: [Some, Categories]
image: "./cover.png"
tags: [some, tags]
jupyter: <kernel-name>       # tier 1 and tier 2 posts only
format:
  html:
    toc: true
    code-fold: true
```

The body conventionally opens with `![Title](./cover.png)` echoing the frontmatter `image`. `.ipynb` posts embed this same YAML in a raw first cell.

**Cover images**: every *new* post gets a `./cover.png` (older posts predate the convention and many have none — not a defect to go fix). Without a natural content-derived cover, the house style is a solid `#4A3AA7` purple card with a translucent rounded category-badge pill top-left (e.g. "ML THEORY & MATH"), a white triple-ring logomark, and bold centered title text — see `posts/topological-data-analysis-clustering/cover.png`, and the `make_cover.py` scripts under `posts/*/src/`. The site favicon (`favicon.png`, declared via `_quarto.yml`'s `website.favicon`) reuses the triple-ring mark.

**`posts/_metadata.yml`** applies `freeze: auto` and `title-block-banner: true` to every post. **`_quarto.yml`** sets `output-dir: docs` and excludes `notes/` from rendering — that's a scratch area, not published content.
