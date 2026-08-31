# AGENTS.md

Guidance for any coding agent working in this repository. This is the canonical file —
**make changes here, not in `CLAUDE.md`**, which only points at this one.

## What this is

A personal ML/data blog ("Synthetic Musings") built with [Quarto](https://quarto.org/)
and published to GitHub Pages at https://project-delphi.github.io/ml-blog/. Each post is
a self-contained `.qmd` or `.ipynb` under `posts/<slug>/`; the rendered site lives in
`docs/` and is served from `main`.

The product here is prose. The tooling exists to get prose onto the web with its
numbers intact; it is not the point of the repo.

Three files carry the detail:

- **`STYLE.md`** — the prose rules. Read it in full before writing or editing a post.
- **`ENVIRONMENTS.md`** — how to build a post's venv and kernel, and why each step.
- **`README.md`** — the reader-facing tour.

## Rules that break the site

- **Never commit to `main`.** Every change — new post, edit, one-line typo — goes:
  feature branch → commit → push → PR with a real description → review → merge.
- **Never run `python`, `python3`, or `pip` bare.** That resolves to Homebrew's
  interpreter, not this repo's. Use `.venv/bin/python`, `.venv-<slug>/bin/python`, or
  `uv run` — including for throwaway one-liners and `-m http.server`.
- **Commit the re-rendered `docs/` with the source that changed it.** There is no CI;
  the site is served from `docs/` on `main`, so a source-only commit silently drifts the
  published site from the repo.
- **Never edit a freeze-backed post without re-rendering it.** Quarto keys frozen
  output on an md5 of `index.qmd`, so a one-word prose fix invalidates the record and
  the next project render tries to execute the post.
- **Never `pre-commit run --all-files`.** The repo carries years of lint debt, so that
  rewrites hundreds of unrelated files into your diff.

The first two are enforced by `PreToolUse` hooks on `Bash`
(`.claude/hooks/block-main-commit.sh`, `block-bare-python.sh`). Both match on raw
command text, so a heredoc that merely writes those words into a file is denied too —
use the Write tool for that. The sibling `test-block-*.sh` assert each hook's
allow/block matrix; run the matching one after touching a hook.

## Editing an existing post

1. **Find out what tier it is.** Read its `jupyter:` field and check whether
   `_freeze/posts/<slug>/` exists. Freeze-backed means any edit forces a re-execution,
   so you need that post's real venv before you touch it — `ENVIRONMENTS.md` has the
   recipe, and the kernel name is usually not the slug.
2. **Match the register it is already in.** `STYLE.md` owns this; do not convert a post
   between registers unless asked.
3. **Edit prose only.** Leave executable cell bodies, widgets, and freeze figures alone
   unless the change is about them.
4. **If you touched `widgets.js` or `widget-data/*.json`, force a re-render.** See
   [Widget sidecars are outside the freeze hash](#widget-sidecars-are-outside-the-freeze-hash).
5. **Re-render with the project render.** Editing the source changes its md5, so
   `freeze: auto` re-executes that one post and nothing else, and the same pass
   refreshes `search.json`, which any prose edit leaves stale. That re-execution needs
   the post's **real** `.venv-<slug>` — on a clone where `make kernels-stub` registered
   a dependency-free stub under that kernel name, the render fails with
   `ModuleNotFoundError` after deleting `docs/` (`ENVIRONMENTS.md`). Use the
   single-document render only while iterating. Then `make check-posts`, then read
   [Before you ship a post](#before-you-ship-a-post).
6. **Ship** on a branch with `docs/` in the same commit. The `ship-pr` skill automates
   branch → commit → push → PR → review, **and merges to `main` itself** when the
   review reports nothing Critical — so use it when the change is meant to land, and do
   the steps by hand when the user asked only for a PR to look at.

The five legacy posts (`LEGACY_NO_ENV` in `scripts/check_posts.py`) are the exception:
they cannot be re-rendered at all. See [Why `_freeze/` is committed](#why-_freeze-is-committed).

## Adding a new post

1. `posts/<slug>/index.qmd` with the frontmatter below; the body opens with
   `![Title](./cover.png)`.
2. Write it to `STYLE.md` — Register B for a new technical post.
3. Add `cover.png` (see [Covers](#covers)).
4. If it runs code, build its venv and kernel, commit `requirements.txt`, and add the
   kernel name to `kernels-stub` in the `Makefile` (`ENVIRONMENTS.md`).
5. If it reads a data file at render time, un-ignore that directory in `.gitignore`
   **and** add the slug to three excludes in `.pre-commit-config.yaml`:
   `check-added-large-files` (a 500 kB cap that would reject the file),
   `trailing-whitespace` and `end-of-file-fixer` (which would rewrite cached upstream
   bytes so the committed copy no longer matches a fresh fetch). Copy the shape of the
   three existing carve-outs.
6. **Finish with a project render.** A single-document render writes only
   `docs/posts/<slug>/`, so the post goes live at its own URL while staying invisible on
   the home page and in search. `make check-posts` fails when a post is missing from
   `docs/listings.json`.

## Rendering

**One post** — `quarto render posts/<slug>/index.qmd`. Narrow blast radius, but it
**always executes that post's code** (`freeze` is honoured only on a project render), so
it needs the post's real venv.

**The whole site** — `QUARTO_PYTHON="$(pwd)/.venv/bin/python" quarto render .`. Deletes
and rebuilds `docs/`, but *respects* `freeze: auto`: it re-executes only posts whose
source md5 changed.

- `QUARTO_PYTHON` is not optional. A bare `quarto render .` resolves a Python that
  cannot see `--user`-registered kernelspecs and dies on the first post pinning a named
  kernel — *after* deleting `docs/`. Recover with `git checkout -- docs`.
- **Both Makefile shortcuts are traps.** `make quatro` is the bare `quarto render .`
  above, without `QUARTO_PYTHON`; `make preview` previews the whole project, which
  indexes every post. Type the full command for the one post instead.
- Kernelspecs resolve while Quarto *indexes* the project, before it consults `_freeze/`.
  A missing kernel fails the whole render, frozen output or not. On a fresh clone run
  `make kernels-stub` first.
- **Check `quarto --version` first.** Nothing pins it; this machine has 1.6.40. A newer
  Quarto rewrites the shared `docs/site_libs/` runtime, which broke older posts'
  JavaScript once already (`452f1fe` restored it by hand). Churn under `site_libs/` in
  your diff is a stop sign, not noise.
- **After any full render**, run `git status --short -- docs | grep '^ D'` and restore
  what it lists — see the third `.gitignore` trap below.

**Preview** — `quarto preview posts/<slug>/index.qmd`, or serve the built output with
`.venv/bin/python -m http.server 8000 --directory docs`. Hand back the preview URL as a
clickable link at the very end of the reply, on its own line, after the summary and
caveats. One link per reply.

**Once merged, kill every preview server**, confirm nothing survives, and say so:

```bash
pkill -f "quarto.*preview"
# Match static servers on cwd, not argv: `cd docs && python -m http.server` has
# no "docs" in its command line, and cwd-matching also spares other projects.
pgrep -f "http\.server" | while read -r pid; do
  cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')
  case "$cwd" in "$PWD"|"$PWD"/*) kill "$pid";; esac
done
```

That final reply ends with `https://project-delphi.github.io/ml-blog/posts/<slug>/`
instead, noting it 404s for a minute or two while Pages rebuilds.

## Checks and lint

There is no test suite. `make check-posts` runs `scripts/check_posts.py`, which verifies
that a code post pins a dedicated kernel (not the shared `python3`) and a
`requirements.txt`, that every pinned kernel appears in `make kernels-stub`, that no
post's frozen output has drifted from its source, and that every post appears in
`docs/listings.json`. Run it before any full render, and run it **through `make`** —
the recipe's bare `python3` is the only one in the repo, kept deliberately because the
checker is stdlib-only and must work on a clone with no `.venv`. Typing
`python3 scripts/check_posts.py` yourself is denied by the hook, which never sees the
interpreter inside a `make` recipe.

Lint is manual — the pre-commit hooks are **not** installed into `.git/hooks/`, so a
plain commit is never pre-vetted:

```bash
.venv/bin/pre-commit run --files <paths>   # not on PATH; scope to files you touched
uvx codespell <file>                       # spelling: NOT a pre-commit hook
.venv/bin/ruff check <file>                # read-only baseline; pre-commit fixes in place
```

`codespell` looks wired up — there is a `[tool.codespell]` section in `pyproject.toml` —
but no hook invokes it, so `pre-commit` never spell-checks. And `no-commit-to-branch` is
a real hook, so any `pre-commit run` while HEAD is on `main` reports a failure that has
nothing to do with your files.

## Before you ship a post

Three defects live *between* artifacts, so every per-artifact check passes and only a
deliberate read catches them: a widget or figure that teaches the opposite of the
sentence pointing at it, a worked example that contradicts the conclusion, and a heading
that contradicts its own section. `STYLE.md` has the procedure for each.

## Architecture

Posts are dependency-isolated: `pyproject.toml` carries only dev/lint tooling, and each
executing post gets its own `.venv-<slug>` plus a named kernel. See `ENVIRONMENTS.md`.

### Why `_freeze/` is committed

`_freeze/` is **tracked**. Five posts predate the venv-per-post convention
(`LEGACY_NO_ENV` in `scripts/check_posts.py`): no pinned kernel, no pinned versions,
dependencies pinned to nothing. Their frozen record is the only reproducible copy of
what they compute, and it is what would let the site render in CI with no ML
dependencies installed.

**Editing a legacy post breaks it** — the md5 changes and the next project render tries
to execute a post that cannot execute. To touch one, build it a venv + kernel +
`requirements.txt` first and delete its `LEGACY_NO_ENV` entry. Shrink that set, never
grow it. `STYLE.md` therefore does not license a style rewrite of those five.

A second exemption, `STALE_FREEZE_OK`, covers posts whose frozen output is knowingly
stale but inert because they have no code cells; the checker cancels the exemption
automatically if code cells appear.

### Widget sidecars are outside the freeze hash

Six posts render an interactive widget by reading a sibling `widgets.js` (and usually
`widget-data/*.json`) and printing it into an inline `<script>` block:
`bayesian-bootstrap`, `statistical-jackknife`, `svd-rotate-stretch-rotate`,
`tensor-inverses-in-practice`, `uses-of-tensor-factorizations`, `volcano-plots`.

Quarto hashes `index.qmd` **alone**. Editing a sidecar therefore leaves `_freeze/`
valid, and a project render keeps serving the old bundle with no warning. After changing
either file, **re-render that post explicitly** before committing. Do not reach for
deleting `_freeze/posts/<slug>/` instead: `check_freeze` returns clean when a
non-legacy record is simply absent, so `make check-posts` stays green while `docs/`
still serves the old bundle, and the next project render has to execute the post for
real — media pipeline and all.

A different mechanism handles browser-run Python exercises: the vendored
`_extensions/r-wasm/live/`, used only by `numpy-to-jax`, via `engine: jupyter`,
`format: live-html`, and a `pyodide:` package list. Don't reinvent it as a hand-rolled
widget, and don't edit the vendored extension.

### Three things `.gitignore` swallows

- **`data/` is ignored everywhere.** A post reading a data file at render time needs
  that path un-ignored, or a clone cannot rebuild it. Three carve-outs exist —
  `volcano-plots`, `dataset-to-biological-signature`, `tensor-inverses-in-practice` —
  each un-ignoring the directory *and* `/**`, since git will not descend into an
  excluded directory to find a negated file inside it.
- **Training artefacts are ignored on purpose**: `checkpoints/`, `posts/**/results/`,
  and Quarto's `posts/**/index_files/` and `index_cache/`. The published copies live in
  `docs/` and `_freeze/`.
- **A render-time asset ignored at the source but tracked under `docs/` disappears on
  the next full render.** `posts/uses-of-tensor-factorizations/media/clip-cp.wav` and
  `clip-hosvd.mp4` are the live case: the post's code writes them, so the source copies
  are ignored, but the rendered copies are tracked. A project render deletes `docs/`,
  and because the post is freeze-backed its code does not re-execute — so the rebuilt
  `docs/` silently drops them and the published page gets a dead `<video>`. Nothing
  warns you, which is why the render step ends with a check for deletions.

## Frontmatter and covers

A skeleton, not a literal template — `toc` and `code-fold` vary, and Register B needs
`number-sections: true`:

```yaml
title: "..."
subtitle: "..."           # optional
description: "..."        # optional; the listing-card text
author: "Ravi Kalia"
date: "YYYY-MM-DD"
categories: [Some, Categories]
image: "./cover.png"
tags: [some, tags]
jupyter: <kernel-name>    # only posts with {python} cells
format:
  html:
    toc: true
    code-fold: true
    number-sections: true # Register B
```

`.ipynb` posts embed the same YAML in a raw cell at the top. `posts/_metadata.yml`
applies `freeze: auto` and `title-block-banner: true` to every post; `_quarto.yml` sets
`output-dir: docs` and excludes `notes/`. The home-page listing is configured in the
root `index.qmd`, not `_quarto.yml` — including `categories: false`, which is why post
categories are not browsable on the site.

### Covers

Every post gets a `./cover.png`. Choose its source in this order:

1. **If** the post writes `cover.png` during its own render — `poor-persons-bayesian`,
   `tensor-factorizations`, `uses-of-tensor-factorizations`, all marked `skip: true` in
   `scripts/cover_sources.yml` — leave that file alone.
2. **Else if** the post has an in-post raster — a committed plot or photo, or a freeze
   figure under `_freeze/posts/<slug>/index/figure-html/` — copy the figure carrying the
   post's **main claim**, not simply the first image in the folder.
3. **Else** find a [Wikimedia Commons](https://commons.wikimedia.org/) image whose
   `LicenseShortName` is public domain, CC0, CC BY, CC BY-SA, Apache, MIT, or BSD —
   never NC or ND, and never a hotlinked Google result. Record the Commons title, page
   URL, and licence in `./cover-source.txt`.
4. **Else** stop. Do **not** draw a purple title card, category pill, or logomark.

Fit it with the shared generator; do not paste a title overlay on top:

```bash
uv run --with pillow --with pyyaml python scripts/make_cover.py \
  posts/<slug> --source <repo-relative-path>       # or --commons "File:Some_image.jpg"
uv run --with pillow --with pyyaml python scripts/make_cover.py --all
```

The default fills 1200×630 and centre-crops, which suits a wide figure and decapitates a
portrait — pass `--fit contain` for those, and look at the result rather than trusting
one run. Record the choice in `scripts/cover_sources.yml` (`source:` / `commons:` /
`skip: true`, plus `fit:`) so `--all` reproduces it and a later single-post run does not
silently re-crop. Do not add a per-post `src/make_cover.py`, and do not edit a
freeze-backed `index.qmd` merely to repoint `image:`.
