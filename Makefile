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
	@for k in bayesian-bootstrap-blog explainability-blog ipm-blog sir-blog \
	          skills-vs-commands svm-margin-blog tda-blog tda-filtered-blog \
	          tda-svm-blog tribes-blog huggingface-blog huggingface-t4-blog \
	          llm-agents blog-base; do \
	  if .venv/bin/python -c "import sys;from jupyter_client.kernelspec import KernelSpecManager as K;sys.exit(0 if '$$k' in K().find_kernel_specs() else 1)"; then \
	    echo "kept     $$k (already registered)"; \
	  else \
	    .venv/bin/python -m ipykernel install --user --name $$k >/dev/null 2>&1 \
	      && echo "stubbed  $$k"; \
	  fi; \
	done
# fail if a post executes code without a pinned kernel + requirements.txt, or
# if its frozen output has drifted from its source. See scripts/check_posts.py.
check-posts:
	python3 scripts/check_posts.py
# quarto preview
preview:
	quarto preview .
# quarto render
quatro:
	quarto render .
# help
help:
	@echo "venv - setup blog virtual environment"
	@echo "install - sync .venv from uv.lock (creates .venv; prunes extras)"
	@echo "lock - regenerate uv.lock from pyproject.toml, leaving .venv alone"
	@echo "kernel - register the blog-base Jupyter kernel"
	@echo "kernels-stub - register every kernel the posts pin (no ML deps; lets a fresh clone render from _freeze/)"
	@echo "check-posts - verify posts pin a kernel + requirements.txt and their frozen output is current"
	@echo "preview - quarto preview"
	@echo "quatro - quarto render"
	@echo "help - help"
