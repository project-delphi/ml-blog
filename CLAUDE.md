# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal ML/data blog ("Synthetic Musings") built with [Quarto](https://quarto.org/) and published to GitHub Pages at https://project-delphi.github.io/ml-blog/. Each post is a self-contained `.qmd` or `.ipynb` under `posts/<slug>/`; the rendered site lives in `docs/` and is served from `main`. There is no CI workflow — `docs/` must be rendered locally and committed *in the same PR* as the source change, or the published site drifts from the source.

The product here is prose. The tooling below exists to get prose onto the web with its numbers intact; it is not the point of the repo.

## Write to the post's register

`STYLE.md` owns the prose rules — read it in full before writing or editing a post. It has two registers. Pick one and stay in it.

- **Register A — personal / old** (dated before 2025-08-28, plus the personal essays listed in `STYLE.md`): claim headings, narrative spine, 2–4 sentence warm-up before mechanics. Personal essays must not drift into guide voice.
- **Register B — recent technical** (technical posts dated 2025-08-28 or later): dry, numbered, list-first documentation. Topic headings, `number-sections: true`, lists over prose, one- or two-sentence paragraphs, no narrative closer.

Do not restyle a Register A post into Register B unless the author asked for that sweep. Do not restyle a Register B post back into claim-heading narrative.

### Register A (summary)

Write so a smart reader who has never met the topic follows every sentence on the first pass.

- Explain the idea before you name it.
- Warm up before the mechanics; never open a section with code, a bullet list, or a table.
- Everyday words over Latinate ones; active voice; one idea per paragraph.
- Headings state claims so the ToC reconstructs the argument. Each section's first sentence links back; its last names the gap the next section fills. Close by returning to the opening claim. Caveats go inline where the objection occurs.

`STYLE.md` carries two calibration examples (cross-validation, the bootstrap). Read the tone off those rather than guessing at it.

### Register B (summary)

Optimize for rapid transfer of technical information. Do not try to make the text engaging.

- Start with the definition or procedure. Cut scene-setting and conversational transitions.
- Topic headings (`## Eigendecomposition`), numbered by Quarto — do not put `1.` in the heading text.
- Default to bullets or numbered lists for multiple concepts, steps, or trade-offs.
- Remaining paragraphs: one or two sentences. Isolate code, commands, and display math in fenced blocks.
- End when the information is delivered.
- **References last (both registers).** If the post cites papers, books, docs, datasets, or other posts as sources, end with `## References`: a short bulleted list of those sources only. Do not invent citations. Skip the section when there are none. Inline links may stay; the end list is the bibliography.

### Two failure modes no check catches

- **An artifact that teaches the opposite of the prose.** A widget, worked example, or figure can be internally correct and still contradict the sentence pointing at it. If the post tells the reader to drag a slider to 12 and watch a band rise, drive it to 12 and confirm the band rises. Every automated check passes either way, because they are all per-artifact.
- **A heading that contradicts its own section.** In Register A, read the headings alone as claims. In Register B, read them as a numbered outline and check each label matches the section.

### Never invent first-person detail

Personal essays land through specifics, and I do not have the author's. Write the scene in the second person or as a hypothetical someone ("picture being twelve in a house where the rules move without notice") — that carries the concreteness an essay needs while claiming nothing about his life. The same pull produces invented citations: the active-voice rule wants a subject, and the nearest one is often a source the post does not have. Ask for the real specifics instead; it improves the essay more than anything invented, and a fabricated memory published under someone's name is unfixable after the fact.

### Always situate the data

Any post that uses, plots, or even mentions a dataset must say where it came from before doing anything with it — in prose, not a checklist bolted on:

- **Provenance** — what the dataset is, with a link or citation.
- **Collector and motive** — who gathered it, under what program, and why they went to the trouble. Instrument, survey, scrape, simulation: say which.
- **Objective** — what is being asked *of this data in this post*, and what the target means in the real world.
- **Downstream impact** — what a decision made from this analysis would affect (a diagnosis, a loan, a forecast, a research conclusion), and what being wrong costs.
- **Why this method** — what property of *this* data (sample size, noise, class imbalance, heavy tails, hierarchy, missingness, small-n uncertainty) makes the post's technique the right tool, and which quantity it improves.

Synthetic data is not exempt: say it is synthetic, give the generating process, and explain what real situation it stands in for and why simulating beats a real dataset here.

### References at the end (both registers)

If a post cites papers, books, docs, datasets, or other posts as sources, end with `## References`: a short bulleted list of those sources only. Do not invent citations. Skip the section when there are none. Inline links may stay; the end list is the bibliography. In Register A this heading is an allowed topic label after the narrative close.

## Workflow

**Never commit to `main`.** Every change — new post, edit, one-line typo — goes: feature branch → commit (source *and* re-rendered `docs/` together) → push → PR with a real description → review → merge. The `ship-pr` skill automates this.

This is enforced. `.claude/settings.json` registers a `PreToolUse` hook on `Bash` running `.claude/hooks/block-main-commit.sh`, which denies a command that reaches `git commit` while HEAD is on `main`/`master`. It anchors on `commit` being the actual git subcommand, so compound forms (`git add -A && git commit …`) are caught while commands that merely mention the word (`git help commit`, `grep -r commit`) pass. It does match on the raw command text, though, so a heredoc that writes `git commit` into a file is denied too — write such files with the Write tool. Branching first passes; switching onto `main` is not an escape hatch. To override deliberately, work from your own terminal or disable the hook via `/hooks`.

**Hand back a localhost preview link whenever a unit of work is complete** — a clickable URL, not just "done": the URL `quarto preview posts/<slug>/index.qmd` prints, or `http://localhost:8000/posts/<slug>/index.html` from the static server below.

**Put that link last.** It goes at the very end of the reply, after the summary, caveats and any list of what was skipped — the final thing on screen, on its own line, so it is never buried mid-message. One link per reply; if several posts changed, link the one the work was about and mention the others by slug.

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

With the server gone there is no localhost link left to give, so that final reply ends with the published URL instead — `https://project-delphi.github.io/ml-blog/posts/<slug>/` — after the confirmation that no server survived. Say that it goes live once Pages rebuilds: the deploy is asynchronous, so that URL 404s for a minute or two after the merge commit lands.

## Commands

- **Render one post**: `quarto render posts/<slug>/index.qmd`. Narrow blast radius, but it **always executes that post's code** — `freeze` is honoured only on a *project* render. Needs the post's real venv.
- **Render the whole site**: `QUARTO_PYTHON="$(pwd)/.venv/bin/python" quarto render .` (`make quatro` runs the bare form). Deletes and rebuilds `docs/`, but *respects* `freeze: auto` — it re-executes only posts whose **source md5 changed** since their `_freeze/` record was written, which today is none. The right tool for anything site-wide: nav, theme, `_quarto.yml`, a stale `search.json`.
  - `QUARTO_PYTHON` is not optional. A bare `quarto render .` resolves a Python that cannot see `--user`-registered kernelspecs and dies on the first post pinning a named kernel — after it has already deleted `docs/`. Recover with `git checkout -- docs`.
  - Kernelspecs resolve while Quarto *indexes* the project, before it consults `_freeze/`. A missing kernel fails the whole render, frozen output or not. On a fresh clone run `make kernels-stub` first.
- **Preview**: prefer `quarto preview posts/<slug>/index.qmd` — a whole-project preview indexes every post. Or serve the built output: `python -m http.server 8000 --directory docs`.
- **Checks** (there is no test suite): `make check-posts` runs `scripts/check_posts.py` — stdlib-only, so it works on any interpreter — verifying that code posts pin a kernel + `requirements.txt`, that every pinned kernel appears in `make kernels-stub`, and that no post's frozen output has drifted from its source. Run it before any full render. `.claude/hooks/test-block-main-commit.sh` asserts the commit hook's allow/block matrix; run it after touching that hook.
- **`make install`**: `uv sync` the base dev/lint toolchain into `.venv` (no per-post ML deps). Installs from the committed `uv.lock` rather than re-resolving, and *prunes* anything not in the lock — hand-installed extras will not survive. `make lock` regenerates the lock without touching `.venv`. `requires-python = ">=3.11"`; raise it rather than lower it, since lowering re-forks the lock across interpreter versions.
- **Lint**: black, ruff, mypy, pyupgrade, commitizen, codespell are configured in `.pre-commit-config.yaml` and `pyproject.toml`, but hooks aren't installed — run them manually, scoped to the files you touched: `pre-commit run --files <paths>`. **Not `--all-files`**: the repo carries years of pre-existing lint debt, so that rewrites ~235 unrelated files into your diff.

## Architecture

### Three tiers of post

Posts are dependency-isolated: `pyproject.toml` carries only dev/lint tooling, no numpy/sklearn/torch.

1. **Executes at render** — its own gitignored `.venv-<slug>` at the repo root, a named Jupyter kernel, `jupyter: <kernel-name>` in frontmatter, and a committed `posts/<slug>/requirements.txt`. 37 posts (count them with `ls posts/*/requirements.txt`; this number goes stale). Venv and kernel names are historical abbreviations and often do **not** match the slug (`convex-optimization-interior-point-methods` → `.venv-ipm`/`ipm-blog`; `sir-training-vs-calibration` → `.venv-sir`/`sir-blog`) — read the post's `jupyter:` field rather than guessing.
2. **Displays code only** — all cells `#| eval: false`. Pins `jupyter: blog-base`, a shared kernel over the base `.venv` (`make install && make kernel`). Four posts, all Claude-API ones: `claude-api-eval-pipeline`, `langgraph-vs-llamaindex`, `messages-api-streaming`, `structured-json-with-claude`. Quarto still needs *some* working kernel to structurally process `{python}` cells even when nothing runs. (`mermaid` and other diagram blocks go through Quarto's own filters and need no kernel.)
3. **Assets generated ahead of the render** — see § `posts/<slug>/src/` below.

`.ipynb` posts are never executed by Quarto; their stored cell outputs are used as-is, which is why none of the 8 have a `_freeze/` record.

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

Some posts carry a `src/` directory of scripts that run *ahead of* the render, with their output committed — `metagenomics/src/make_figs.py` and `voice-ai-architectures-2026/src/make_figs.py` (figures), `open-source-tts-history/src/make_audio.py` and `llm-agent-memory/src/make_audio.py` (which write the committed `audio/*.mp3`), `kite-shape-analysis/src/` (`prepare_photos.py`, `digitize_landmarks.py`), and full module sets under `bayesian-bootstrap/src/`, `svd-rotate-stretch-rotate/src/` and `explainability-localization/src/`. A `src/export_widget_data.py` writes the JSON behind a post's interactive widget (`bayesian-bootstrap`, `svd-rotate-stretch-rotate`, `correlation-concordance-discordance`). Covers are not drawn here: `scripts/make_cover.py` writes `cover.png` from the map in `scripts/cover_sources.yml`.

These scripts are **not** executed by Quarto and are not covered by `_freeze/` or `make check-posts`. Regenerating an asset means running the script yourself and committing the result. They may also carry their own dependency pin (`open-source-tts-history/src/requirements-audio.txt`) and their own venv with no kernel and no post pinning it — `.venv-kokoro` exists solely to run `make_audio.py`, which is why it looks orphaned next to the kernel-backed venvs.

### What `.gitignore` swallows

Two rules bite when adding a post:

- **`data/` is ignored everywhere.** A post that reads a data file at render time needs that path un-ignored, or a clone cannot rebuild it. Two directories are carved out already — `posts/volcano-plots/data/` and `posts/dataset-to-biological-signature/data/` — so copy their two-line pattern (un-ignore the directory *and* `/**`; git will not descend into an excluded directory to find a negated file inside it). Both are also exempted from the whitespace and large-file pre-commit hooks, since they cache upstream bytes verbatim. Note the carve-out covers the *source* copy only: Quarto also copies the directory into `docs/`, where the same `data/` rule swallows it again, so every render regenerates a few MB that is never committed and the published path 404s. Nothing links to it, so this is cosmetic — but don't be surprised by it.
- **Training artefacts are ignored on purpose**: `checkpoints/`, `posts/**/checkpoints/`, `posts/**/results/`, and Quarto's `posts/**/index_files/` and `index_cache/`. A fine-tuning post drops hundreds of MB there relative to the render's cwd. The published copies live in `docs/` and `_freeze/`.

### Why `_freeze/` is committed

`_freeze/` is **tracked**, not ignored. Five posts still predate the venv-per-post convention — `data-types`, `features-importance-after-clustering`, `poor-persons-bayesian`, `post-with-code`, `working-with-quarto` — with no pinned kernel, no pinned versions, and dependencies pinned to nothing. Their committed `_freeze/` record is the only reproducible copy of what they compute. It is also what would let the site render in CI with no ML dependencies installed.

`scripts/check_posts.py` grandfathers exactly those five in `LEGACY_NO_ENV`. **Shrink that set, never grow it.** A second exemption, `STALE_FREEZE_OK` (`pinecone-vs-weaviate`, `pyspark`), covers posts whose frozen output is knowingly stale but inert because they have no code cells at all; the checker cancels the exemption automatically if code cells later appear.

**Editing a legacy post breaks it.** Quarto keys frozen output on an md5 of the source, so *any* edit — even a one-word prose fix — invalidates the record and makes the next project render try to execute a post that cannot execute. If you touch one, build it a venv + kernel + `requirements.txt` first and delete its `LEGACY_NO_ENV` entry. This is not hypothetical: the cover-image commits edited 11 sources without re-rendering and left the freeze cache silently stale for months. `STYLE.md` therefore does not license a style rewrite of those five.

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

**Cover images**: every post gets a `./cover.png`. Decide the source with these conditionals, in order:

1. **IF** the post already writes `cover.png` during render (today: `poor-persons-bayesian`, which `ggsave`s figure 2) **THEN** leave that file alone. Do not overwrite it with the shared script.
2. **ELSE IF** the post contains an in-post raster visual that is not `cover.png` itself — a committed plot or photo, or a freeze figure under `_freeze/posts/<slug>/index/figure-html/` — **THEN** copy the figure that carries the post's main claim into `./cover.png`. Prefer that claim-bearing figure over "the first image in the folder".
   - Example: `brian-ripley-rousseeuw-prize` → `ripley-oxford-announcement.png` (the Oxford press photo already in the post).
   - Example: `volcano-plots` → `_freeze/posts/volcano-plots/index/figure-html/fig-airway-output-1.png` (the GSE52778 volcano, `fig-airway`).
3. **ELSE IF** the post has no internal raster (mermaid inlined into HTML does not count) **THEN** search [Wikimedia Commons](https://commons.wikimedia.org/) for a relevant image whose `LicenseShortName` is public domain, CC0, CC BY, CC BY-SA, or Apache 2.0 (software logos) — never CC BY-NC, and never a hotlinked Google result. Commit it as `./cover.png` and record the Commons title, page URL, and licence in `./cover-source.txt` (same shape as `posts/poor-persons-bayesian/images/attribution.txt`).
4. **ELSE** stop. Do **not** draw a solid `#4A3AA7` purple title card, category pill, triple-ring logomark, or centred title text.

Fit the chosen raster to 1200×630 with the shared generator; do not paste a title overlay on top:

```bash
uv run --with pillow --with pyyaml python scripts/make_cover.py \
  posts/<slug> --source <repo-relative-path>
# or, for a Commons file:
uv run --with pillow --with pyyaml python scripts/make_cover.py \
  posts/<slug> --commons "File:Some_image.jpg"
# or rebuild everything the map already knows:
uv run --with pillow --with pyyaml python scripts/make_cover.py --all
```

By default the raster is scaled to fill 1200×630 and centre-cropped, which is right for a wide figure. A **portrait** source loses its subject to that crop — the Ripley photo came out with the top of his head cut off — so pin it with `--fit contain`, which scales the source to fit whole and pads the remainder with the colour of the edges the pad abuts.

Record the choice in `scripts/cover_sources.yml` (`source:`, `commons:`, or `skip: true`, plus `fit: contain` where it applies) so a later `--all` can reproduce it. A recorded `fit:` is also what a bare `--source`/`--commons` run for that slug defaults to, so regenerating one cover the usual way will not silently re-crop it; pass `--fit` only to override. Do not add a per-post `src/make_cover.py`. Do not edit a freeze-backed `index.qmd` merely to point `image:` at a new file — that invalidates `_freeze/` (see `scripts/check_posts.py`).

The site favicon (`favicon.png`, declared via `_quarto.yml`'s `website.favicon`) still uses the triple-ring mark; that is independent of post covers.

**`posts/_metadata.yml`** applies `freeze: auto` and `title-block-banner: true` to every post. **`_quarto.yml`** sets `output-dir: docs` and excludes `notes/` from rendering — that's a scratch area, not published content.
