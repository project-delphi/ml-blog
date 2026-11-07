# Nothing pins the system Quarto, and a newer one rewrites docs/site_libs/ (see
# AGENTS.md). So the Makefile runs its own copy when it has one: `make quarto`
# installs $(QUARTO_VERSION) under ~/.local/share/quarto/, beside whatever the
# system has, and every target below then uses it. Without that copy they use
# `quarto` from PATH, and the render targets refuse any version but this one.
QUARTO_VERSION := 1.6.40
QUARTO_HOME := $(HOME)/.local/share/quarto/$(QUARTO_VERSION)
QUARTO ?= $(if $(wildcard $(QUARTO_HOME)/bin/quarto),$(QUARTO_HOME)/bin/quarto,quarto)

# setup blog virtual environment (--allow-existing keeps this idempotent;
# without it uv exits non-zero when .venv already exists)
venv:
	uv venv --allow-existing .venv
# install the base toolchain from uv.lock (not a fresh resolve), so every clone
# gets byte-identical versions; uv sync also prunes anything not in the lock.
# No `venv` prerequisite: uv sync creates and manages .venv itself, and running
# `uv venv` first would only risk building it against the wrong interpreter.
install:
	uv sync
# regenerate uv.lock from pyproject.toml without touching .venv -- useful for
# reviewing a dependency change before installing it. `make install` keeps the
# lock current on its own; per-post ML deps live in posts/<slug>/requirements.txt
lock:
	uv lock
# register the shared blog-base Jupyter kernel over .venv
kernel:
	.venv/bin/python -m ipykernel install --user --name blog-base
# Register every kernel name the posts pin, all pointing at .venv. Quarto
# resolves kernelspecs while indexing the project -- before it consults
# _freeze/ -- so a clone missing one of these cannot render *any* page, frozen
# or not. These are stubs: they satisfy the lookup so frozen output can be
# reused, and carry none of the ML dependencies. Build the real .venv-<slug>
# from posts/<slug>/requirements.txt only when you need to *execute* a post.
# Existing kernels are left alone: `ipykernel install --user` overwrites by
# name, so on a machine that already has the real .venv-<slug> kernels this
# would silently repoint them at the dependency-free .venv and break every
# targeted render. Only missing names are registered.
kernels-stub: install
	@for k in bayesian-bootstrap-blog causality-blog explainability-blog ipm-blog \
	          sir-blog skills-vs-commands svm-margin-blog tda-blog tda-filtered-blog \
	          tda-svm-blog tribes-blog huggingface-blog huggingface-t4-blog \
	          llm-agents llm-agent-memory recursive-inversion matrix-blog eigen-blog \
	          authorship-blog kendall-blog kite-blog jackknife-blog svd-blog \
	          volcano-blog signature-blog tensor-blog mfc-blog clustering-blog \
	          neutrophil-axis-blog gapdh-net-blog pca-blog ppca-blog \
	          efferocytosis-blog efferocytosis-guide-blog cp-tucker-blog calibration-blog \
	          collinearity-blog \
	          blog-base; do \
	  if .venv/bin/python -c "import sys;from jupyter_client.kernelspec import KernelSpecManager as K;sys.exit(0 if '$$k' in K().find_kernel_specs() else 1)"; then \
	    echo "kept     $$k (already registered)"; \
	  else \
	    .venv/bin/python -m ipykernel install --user --name $$k >/dev/null 2>&1 \
	      && echo "stubbed  $$k"; \
	  fi; \
	done

# ---- checks -----------------------------------------------------------------
# The two scripts under scripts/ that these recipes run bare are stdlib-only on
# purpose: they must work on a clone with no .venv. That bare `python3` is the
# only one in the repo; everywhere else name the interpreter.

# fail if a post executes code without a pinned kernel + requirements.txt, or
# if its frozen output has drifted from its source. See scripts/check_posts.py.
check-posts:                   # ARGS=--no-categories while adding a category
	python3 scripts/check_posts.py $(ARGS)
# Ask a bounded question about rendered output without reading any of it:
#   make docs-query ARGS="post train-dev-test-splits"
#   make docs-query ARGS="widget volcano-plots"
# Bare python3 for the same reason as above: stdlib only, so it works on a
# clone with no .venv, which is exactly where docs-inspect runs.
docs-query:
	@python3 scripts/docs_query.py $(ARGS)
# After a prose-only edit to a freeze-backed post: rewrite the frozen record so
# the next project render reuses the stored cell outputs instead of
# re-executing the post. Refuses if any code cell changed. See
# scripts/freeze_realign.py.
freeze-realign:
	@test -n "$(SLUG)" || { echo "usage: make freeze-realign SLUG=<slug> [ARGS='--check' | ARGS='--legacy']"; exit 2; }
	python3 scripts/freeze_realign.py $(SLUG) $(ARGS)
# ruff: lint + import order + formatting, read-only. `make fmt` applies fixes.
lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
fmt:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .
# prose spelling for .qmd / .md; config in pyproject.toml, allow-list in .codespell-ignore
spell:
	.venv/bin/codespell
# unit tests for the scripts under scripts/
test:
	.venv/bin/python -m pytest
# everything that does not need Quarto
check: lint spell test check-posts

# ---- rendering --------------------------------------------------------------
# Install the pinned Quarto from its release tarball into $(QUARTO_HOME): no
# admin rights, and the system copy is left alone. It unpacks into a scratch
# directory first, so an interrupted download never looks like an install. CI
# installs its own copy through quarto-actions.
quarto:
	@if test -x "$(QUARTO_HOME)/bin/quarto"; then echo "quarto $(QUARTO_VERSION) is already in $(QUARTO_HOME)"; exit 0; fi; \
	  test "$$(uname -s)" = Darwin || { echo "make quarto knows only the macOS tarball; install $(QUARTO_VERSION) from https://github.com/quarto-dev/quarto-cli/releases/tag/v$(QUARTO_VERSION)"; exit 1; }; \
	  mkdir -p "$(dir $(QUARTO_HOME))" && tmp=$$(mktemp -d "$(dir $(QUARTO_HOME)).partial.XXXXXX") && \
	  curl -fL --retry 3 -o "$$tmp/quarto.tar.gz" \
	    https://github.com/quarto-dev/quarto-cli/releases/download/v$(QUARTO_VERSION)/quarto-$(QUARTO_VERSION)-macos.tar.gz && \
	  mkdir "$$tmp/q" && tar -xzf "$$tmp/quarto.tar.gz" -C "$$tmp/q" && \
	  test "$$("$$tmp/q/bin/quarto" --version)" = "$(QUARTO_VERSION)" && \
	  mv "$$tmp/q" "$(QUARTO_HOME)" && rm -rf "$$tmp" && \
	  echo "installed quarto $(QUARTO_VERSION) in $(QUARTO_HOME)" \
	  || { rm -rf "$$tmp"; echo "make quarto failed; nothing was installed"; exit 1; }
check-quarto:
	@v=$$("$(QUARTO)" --version 2>/dev/null); test "$$v" = "$(QUARTO_VERSION)" \
	  || { echo "$(QUARTO) is $${v:-missing}; $(QUARTO_VERSION) is required (a newer one rewrites docs/site_libs/). Run make quarto to install it beside the system copy."; exit 1; }
# Print the Quarto these targets use, for a command that has to call it directly:
#   "$(make -s which-quarto)" render posts/<slug>/slides.qmd
which-quarto:
	@echo "$(QUARTO)"
# Render-time assets that are ignored at the source but tracked under docs/
# (the third .gitignore trap in AGENTS.md). A full render rebuilds docs/ from
# source and so drops them unless the post re-executed; `render` puts back
# only the ones the render deleted, before the deletion check.
RENDER_KEEP := docs/posts/uses-of-tensor-factorizations/media

# The whole site, respecting freeze. Output goes to render.log because a
# 120-post render emits thousands of lines; the tail and any error lines are
# echoed. QUARTO_PYTHON is not optional: a bare `quarto render .` resolves a
# Python that cannot see --user kernelspecs and dies after deleting docs/.
render: check-quarto
	python3 scripts/check_posts.py --no-listing
	@QUARTO_PYTHON="$(CURDIR)/.venv/bin/python" "$(QUARTO)" render . > render.log 2>&1; \
	  status=$$?; tail -40 render.log; \
	  grep -inE 'error|not found|traceback' render.log || true; \
	  test $$status -eq 0 || { echo "render failed (see render.log); restore with: git checkout -- docs"; exit $$status; }
	@git status --short -- $(RENDER_KEEP) | sed -n 's/^ D //p' | while read -r f; do git checkout -- "$$f"; done
	@$(MAKE) --no-print-directory docs-deleted
	@$(MAKE) --no-print-directory check-posts
# One post, always executed (freeze is honoured only on a project render), so
# it needs that post's real kernel registered. Iterate with this; ship with
# `make render`, which is what refreshes search.json and listings.json.
render-post: check-quarto
	@test -n "$(SLUG)" || { echo "usage: make render-post SLUG=<slug>"; exit 2; }
	QUARTO_PYTHON="$(CURDIR)/.venv/bin/python" "$(QUARTO)" render posts/$(SLUG)/index.qmd
# Live preview of one post, through the same pinned Quarto as the renders.
preview: check-quarto
	@test -n "$(SLUG)" || { echo "usage: make preview SLUG=<slug>"; exit 2; }
	QUARTO_PYTHON="$(CURDIR)/.venv/bin/python" "$(QUARTO)" preview posts/$(SLUG)/index.qmd
# A project render deletes docs/ and rebuilds it from source, so an asset that
# is ignored at the source but tracked under docs/ (see .gitignore) vanishes.
# Lists such deletions and fails; restore them with `git checkout -- <path>`.
docs-deleted:
	@if git status --short -- docs | grep '^ D'; then \
	  echo "docs-deleted: the render dropped tracked files; restore them with git checkout -- <path>"; exit 1; \
	else echo "docs-deleted: nothing deleted"; fi
# preview the built site without executing anything
serve:
	.venv/bin/python -m http.server 8000 --directory docs

help:
	@echo "venv - setup blog virtual environment"
	@echo "install - sync .venv from uv.lock (creates .venv; prunes extras)"
	@echo "lock - regenerate uv.lock from pyproject.toml, leaving .venv alone"
	@echo "kernel - register the blog-base Jupyter kernel"
	@echo "kernels-stub - register every kernel the posts pin (no ML deps; lets a fresh clone render from _freeze/)"
	@echo "check-posts - verify posts pin a kernel + requirements.txt and their frozen output is current"
	@echo "docs-query ARGS=\"post <slug>\" - a bounded question about rendered output, without reading it"
	@echo "freeze-realign SLUG=<slug> [ARGS=--check|--legacy] - accept a prose-only edit to a freeze-backed post without re-executing it"
	@echo "lint / fmt - ruff check + format (read-only / apply)"
	@echo "spell - codespell over prose"
	@echo "test - pytest over scripts/"
	@echo "check - lint + spell + test + check-posts"
	@echo "quarto - install the pinned Quarto ($(QUARTO_VERSION)) beside the system copy"
	@echo "which-quarto - print the Quarto the render targets use"
	@echo "render - full project render to render.log, then the docs/ deletion check"
	@echo "render-post SLUG=<slug> - execute and render one post"
	@echo "preview SLUG=<slug> - live preview of one post with the pinned Quarto"
	@echo "docs-deleted - list tracked docs/ files a render removed"
	@echo "serve - static server for docs/ on :8000"

.PHONY: venv install lock kernel kernels-stub check-posts docs-query freeze-realign lint fmt spell test check quarto check-quarto which-quarto render render-post preview docs-deleted serve help
