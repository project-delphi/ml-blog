# Build for "The Matrix That Rotates, Stretches, and Rotates Again"
#
# Everything runs through the post's dedicated virtualenv at the repo root, per
# the convention in ../../AGENTS.md. Never the system Python.

REPO    := $(shell cd ../.. && pwd)
VENV    := $(REPO)/.venv-svd
PY      := $(VENV)/bin/python
KERNEL  := svd-blog

.PHONY: all venv data widgets cover render clean distclean help

all: data widgets cover render

## venv: create the virtualenv and register the Jupyter kernel
venv:
	uv venv $(VENV) --python 3.12
	uv pip install --python $(PY) -r requirements.txt
	$(PY) -m ipykernel install --user --name $(KERNEL) \
		--display-name "Python (svd)"

## data: ensure assets/Filip.dat and the MovieLens archive are present
data:
	cd $(CURDIR) && $(PY) src/data.py

## widgets: rewrite the committed widget-data JSON (fails over a 1 MB budget)
widgets:
	cd $(CURDIR) && $(PY) src/export_widget_data.py

## cover: rewrite the social card from the in-post SVD figure
cover:
	cd $(REPO) && uv run --with pillow --with pyyaml python scripts/make_cover.py posts/svd-rotate-stretch-rotate

## render: build the post
render:
	cd $(REPO) && QUARTO_PYTHON="$(PY)" quarto render posts/svd-rotate-stretch-rotate/index.qmd

## clean: drop the MovieLens download, keeping everything committed
clean:
	rm -rf data/raw

## distclean: also drop the frozen output, forcing a full re-execution
distclean: clean
	rm -rf $(REPO)/_freeze/posts/svd-rotate-stretch-rotate

help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
