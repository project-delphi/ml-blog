# Synthetic Musings

[![site](https://img.shields.io/badge/site-live-4A3AA7)](https://project-delphi.github.io/ml-blog/)
[![built with Quarto](https://img.shields.io/badge/built%20with-Quarto-75AADB)](https://quarto.org/)
[![license](https://img.shields.io/badge/license-CC%20BY%204.0%20%2F%20MIT-blue)](LICENSE)

Essays on machine learning, statistics, and the tooling around them — by Ravi Kalia.

**Read them at https://project-delphi.github.io/ml-blog/**

## What's here

105 posts written since February 2024 (97 Quarto `.qmd`, 8 Jupyter `.ipynb`), each one
self-contained under `posts/<slug>/`. The recurring threads are machine learning and
statistics, NLP and LLMs, mathematics — linear algebra, shape analysis, topology —
data engineering, and developer tooling.

40 of them execute their own code at render time and carry a pinned environment, so
for those the numbers and figures on the site are the ones the committed source
produces. The rest display code without running it, ship stored notebook outputs, or
predate the convention.

## A few to start with

- [Why the Bayesian Bootstrap?](https://project-delphi.github.io/ml-blog/posts/bayesian-bootstrap/)
  — the Dirichlet reweighting behind it, with an interactive widget.
- [Topological Data Analysis: Finding Clusters, Loops, and Voids Across Scales](https://project-delphi.github.io/ml-blog/posts/topological-data-analysis-clustering/)
  — what persistent homology sees that k-means can't.
- [LLM Agents from First Principles](https://project-delphi.github.io/ml-blog/posts/llm-agents-from-first-principles/)
  — a stdlib-only agent loop, run against a live API, transcripts and all.
- [From Formants to Foundation Models](https://project-delphi.github.io/ml-blog/posts/open-source-tts-history/)
  — fifty years of open-source speech synthesis, with audio from each era.
- [Training vs. Calibrating Epidemiological Models](https://project-delphi.github.io/ml-blog/posts/sir-training-vs-calibration/)
  — why fitting an SIR model isn't machine learning.
- [The Cheapest Ladder Is Also the Shortest](https://project-delphi.github.io/ml-blog/posts/cheapest-ladder-is-shortest/)
  — the most recent essay.

## Repo layout

```
posts/<slug>/     one post: index.qmd (or .ipynb), cover.png, sometimes
                  requirements.txt and a src/ of scripts run ahead of the render
docs/             the rendered site — committed, and served by GitHub Pages from main
_freeze/          Quarto's cached execution output — committed on purpose
scripts/          check_posts.py, the repo's only check
.claude/hooks/    block-main-commit.sh — refuses any commit while HEAD is on main
_quarto.yml       site config; index.qmd is the post listing
Makefile          install / kernel / kernels-stub / check-posts / quatro (render)
AGENTS.md         the working rules for coding agents; CLAUDE.md points at it
```

## Working on the blog

### Quickstart

```bash
make install       # base dev/lint toolchain into .venv, from uv.lock
make kernels-stub  # register every kernel name the posts pin
QUARTO_PYTHON="$(pwd)/.venv/bin/python" quarto render .   # rebuild docs/ from _freeze/
python -m http.server 8000 --directory docs               # read it at localhost:8000
```

That builds the whole site from the committed frozen output with no ML dependencies
installed. Previewing a *single* post is a different thing: `quarto preview
posts/<slug>/index.qmd` always executes that post's code, so it needs the post's own
`.venv-<slug>` built first (see [Rendering](#rendering)). Out of the box it works only
for the four `blog-base` posts, which run nothing.

`make kernels-stub` is not optional on a fresh clone. Quarto resolves kernelspecs
while it indexes the project — *before* it consults `_freeze/` — so one missing kernel
name fails the entire site build, frozen output or not. The stubs point at `.venv` and
carry no ML dependencies; they exist to satisfy the lookup.

There is no CI. `docs/` is rendered locally and committed.

### Rendering

**One post** — `quarto render posts/<slug>/index.qmd`. Narrow blast radius, but it
*always executes* that post's code, so it needs the post's real venv.

**The whole site** — `QUARTO_PYTHON="$(pwd)/.venv/bin/python" quarto render .`. This
respects `freeze: auto` and re-executes only posts whose source md5 changed. It's the
right tool for anything site-wide: nav, theme, `_quarto.yml`, a stale `search.json`.

`QUARTO_PYTHON` is not optional there. A bare `quarto render .` resolves a Python that
can't see `--user`-registered kernelspecs and dies on the first post pinning a named
kernel — after it has already deleted `docs/`. Recover with `git checkout -- docs`.

### Adding a post

1. Create `posts/<slug>/index.qmd` with the standard frontmatter — title, author,
   date, categories, `image: "./cover.png"`, tags, and `jupyter:` if it has code.
2. Add a `cover.png`; the body opens with `![Title](./cover.png)`.
3. Read [`STYLE.md`](STYLE.md) before writing the prose. It owns the spine, the
   section seams, and the rule that any post touching a dataset must situate it —
   provenance, who collected it and why, what a wrong answer would cost.
4. Pick a tier (below), render, then commit source and `docs/` together.

### Three tiers of post

Dependencies are isolated per post; `pyproject.toml` carries only dev/lint tooling.

- **Executes at render** — its own gitignored `.venv-<slug>`, a named kernel, and a
  committed `posts/<slug>/requirements.txt`. Thirteen Hugging Face posts share two
  kernels (`huggingface-blog`, `huggingface-t4-blog`) instead of thirteen ~1.2 GB
  venvs.
- **Displays code only** — every cell `#| eval: false`, pinning the shared
  `blog-base` kernel. Quarto still needs *some* working kernel to process `{python}`
  cells even when nothing runs.
- **Assets built ahead of the render** — a `src/` of scripts you run yourself, with
  their output committed. Quarto never executes these and `_freeze/` doesn't cover
  them.

[`ENVIRONMENTS.md`](ENVIRONMENTS.md) has the full recipe for building a post venv, and
the reasons each step is there.

### Two rules that bite

1. **Re-render and commit `docs/` in the same PR as the source change.** The site is
   served from `docs/` on `main`; a source-only commit silently drifts the published
   site from the repo.
2. **Never commit to `main`.** Branch → commit → push → PR → review → merge. A
   `PreToolUse` hook (`.claude/hooks/block-main-commit.sh`) denies any commit while
   HEAD is on `main` or `master`.

### Checks

- `make check-posts` — run it before any full render. It verifies that code posts pin
  a kernel and a `requirements.txt`, that every pinned kernel appears in
  `make kernels-stub`, and that no post's frozen output has drifted from its source.
  Stdlib-only, so it runs on any interpreter.
- `.venv/bin/pre-commit run --files <your paths>` for lint (black, ruff, mypy,
  pyupgrade). It is not on `PATH`, and the hooks are not installed into `.git/hooks/`,
  so nothing runs automatically on commit. Scope it to what you changed —
  `--all-files` rewrites a few hundred files of pre-existing lint debt. For spelling,
  `uvx codespell <file>`: it is configured in `pyproject.toml` but no hook runs it.

There is no test suite.

## Further reading

- [`AGENTS.md`](AGENTS.md) — the working rules: branch policy, rendering, why
  `_freeze/` is committed, and the traps `.gitignore` sets. `CLAUDE.md` is a pointer
  to it, so coding agents pick it up automatically.
- [`ENVIRONMENTS.md`](ENVIRONMENTS.md) — the venv tiers, the build recipes, and the
  failure each step prevents.
- [`STYLE.md`](STYLE.md) — the prose rules for posts.

## License

Prose, figures, and audio are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
code is MIT. See [`LICENSE`](LICENSE). Datasets referenced by posts keep their own
licenses.
