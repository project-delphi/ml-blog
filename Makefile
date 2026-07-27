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
	@echo "preview - quarto preview"
	@echo "quatro - quarto render"
	@echo "help - help"
