# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal ML/data blog ("Synthetic Musings") built with [Quarto](https://quarto.org/) and published to GitHub Pages at https://project-delphi.github.io/ml-blog/. Each post is a self-contained `.qmd` or `.ipynb` file under `posts/<slug>/`; the rendered static site lives in `docs/` and is served from `main` (there is no CI workflow — `docs/` must be rendered locally and committed *in the same commit/PR* as the source change, otherwise the published site drifts from the source).

Pages serves this repo with its **legacy Jekyll builder**, so `docs/.nojekyll` is load-bearing: without it Jekyll walks all ~80 rendered posts on every deploy, and any `{{` or `{%` that lands in a code block becomes a build failure rather than a page. It is listed under `project: resources:` in `_quarto.yml` rather than merely committed, because a project render deletes and rebuilds `docs/` — an unlisted `docs/.nojekyll` gets wiped on the next full render and the protection disappears silently. If a Pages build ever hangs or errors, check that `docs/.nojekyll` still exists before debugging content.

## Commands

- Render one post: `quarto render posts/<slug>/index.qmd` (or `index.ipynb`). Narrow blast radius — it leaves every other built page untouched — but note it **always executes that post's code**: `freeze` is only honoured on a *project* render, never on a single document. So this needs the post's real venv, and it is the slow path for a code-heavy post.
- Render the whole site: `QUARTO_PYTHON="$(pwd)/.venv/bin/python" quarto render .` (`make quatro` runs the bare form). This deletes and rebuilds `docs/`, but it *respects* `freeze: auto` (`posts/_metadata.yml`) — it re-executes only posts whose **source md5 has changed** since their `_freeze/` record was written, which today is none. It is the right tool for anything site-wide: nav, theme, `_quarto.yml`, or a stale `search.json`. Run `make check-posts` first; that is exactly what it checks.
  - `QUARTO_PYTHON` is not optional here. A bare `quarto render .` resolves a Python that cannot see `--user`-registered kernelspecs, dies on the first post pinning a named kernel — and by then it has already deleted `docs/`. Recover with `git checkout -- docs`.
  - Kernelspecs are resolved while Quarto *indexes* the project, before it ever consults `_freeze/`. A missing kernel therefore fails the whole render, frozen output or not. On a fresh clone run `make kernels-stub` first.
- `.ipynb` posts are never executed by Quarto by default; their stored cell outputs are used as-is. That is why none of the 8 have a `_freeze/` record.
- Preview: `quarto preview` (or `make preview`). Previewing the whole project indexes every post the first time, so prefer `quarto preview posts/<slug>/index.qmd` or serve the already-built `docs/` folder statically (e.g. `python -m http.server` from `docs/`) when you just need to eyeball one post.
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

Posts that only *display* code (all cells `#| eval: false` — e.g. `posts/langgraph-vs-llamaindex`) need no dedicated venv; they pin `jupyter: blog-base`, a shared kernel over the base `.venv` (`make install && make kernel`). Quarto still needs *some* working kernel to structurally process `{python}` cells even when nothing runs, so register it once on a fresh clone. (`mermaid` and other diagram blocks are handled by Quarto's own filters and need no kernel.)

**The Hugging Face posts share two venvs, not thirteen.** Thirteen posts (the HuggingFace-era ones: `datasets`, `hub`, `tokenizers`, `peft`, `transformers-library`, `hugging-face-evaluate-library`, `optimum`, `setfit`, `diffusers`, `classifying_text_chunks`, `langchain`, `evaluation-metrics`, `soft-vs-hard-prompts`) share ~90% of their dependency closure, most of it `torch`. Each venv measures ~1.2 GB and `uv` copies rather than hardlinks into them here, so thirteen `.venv-<slug>` directories would be ~15 GB of near-identical wheels. They pin one of two shared kernels instead:

- **`huggingface-blog`** over `.venv-huggingface` — `transformers` 5.x. Eleven posts.
- **`huggingface-t4-blog`** over `.venv-huggingface-t4` — `transformers` 4.57.x. Two posts, `optimum` and `setfit`, and only because those two packages cap it. `optimum-onnx` declares `transformers<4.58.0`; `setfit` declares no upper bound at all but imports `transformers.training_args.default_logdir`, which 5.x removed — so it *resolves* against 5.x and then fails at import. A resolver-only check will not catch that; import the packages before trusting a resolve.

Build them with the § Architecture recipe above and these package sets — `posts/<slug>/requirements.txt` pins the exact resolved versions, but that is a freeze, not a spec, so the intent is recorded here:

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

Note that `setfit` is deliberately absent from the first set: installing it there resolves fine and then breaks the kernel at import, for the reason above.

The trade this makes: `requirements.txt` is per-post as usual, but for these thirteen it is a copy of a shared freeze, so bumping a dependency for one post changes the versions recorded for all the posts on that kernel. `_freeze/` still hashes source only, so nothing silently re-executes — but the committed versions and the versions that produced a given post's output can drift apart. Re-render every post on a kernel when you bump that kernel's venv.

**Two tiers of code post, and why `_freeze/` is committed.** Eleven posts are reproducible from source: dedicated `.venv-<slug>`, named kernel, committed `requirements.txt`. Eighteen older ones are not — no pinned kernel, no pinned versions, and dependencies (`diffusers`, `setfit`, `peft`, `transformers`, an `openai` client) pinned to nothing. Their committed `_freeze/` record is the only reproducible copy of what they compute, which is why `_freeze/` is tracked rather than ignored.

Thirteen of those eighteen are being migrated out of that tier by the Hugging Face repair work: the two shared venvs above are their replacement environment. A post leaves `LEGACY_NO_ENV` only once it pins a kernel, carries a `requirements.txt` and has been re-rendered, so the count in that set is the live measure of how far along this is.

That has one consequence worth internalising: **editing a legacy post breaks it.** Quarto keys frozen output on an md5 of the source file, so *any* source edit — even a one-word prose fix — invalidates the record and makes the next project render try to execute a post that cannot execute. If you touch one, build it a venv + kernel + `requirements.txt` first, per the recipe above, and remove its entry from `LEGACY_NO_ENV` in `scripts/check_posts.py`. `make check-posts` enforces both halves of this; run it before any full render. This is not hypothetical — the cover-image commits edited 11 sources without re-rendering and left the freeze cache silently stale for months.

**One post calls a live API.** `llm-agents-from-first-principles` executes an LLM agent against Groq's free tier, so re-executing it needs `GROQ_API_KEY` in the environment — its `.venv-llm-agents` holds only Quarto's execution stack, because `agent.py` is stdlib-only. A *project* render never re-executes it (`freeze: auto`, and its `_freeze/` record is committed), so a keyless clone renders the site fine; only `quarto render posts/llm-agents-from-first-principles/index.qmd` needs the key, since single-document renders always execute. Its captured transcripts are one sample from a stochastic policy, not a reproducible fixed point: re-rendering legitimately changes the published output even at `temperature=0`, so don't re-render it to "refresh" anything, and never edit the transcripts by hand.

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
