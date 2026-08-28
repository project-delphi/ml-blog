# Build for "Explainability Is a Localization Problem"
#
# Everything runs through the post's dedicated virtualenv at the repo root, per
# the convention in ../../CLAUDE.md. Never the system Python.

REPO    := $(shell cd ../.. && pwd)
VENV    := $(REPO)/.venv-explainability
PY      := $(VENV)/bin/python
KERNEL  := explainability-blog

.PHONY: all venv model figures cover render clean distclean help

all: figures render

## venv: create the virtualenv and register the Jupyter kernel
venv:
	uv venv $(VENV)
	uv pip install --python $(PY) -r requirements.txt
	$(PY) -m ipykernel install --user --name $(KERNEL) \
		--display-name "Python (explainability)"

## model: download MNIST if needed, train the CNN once, cache the checkpoint
model:
	$(PY) src/mnist_model.py

## figures: rebuild every PNG in figures/ and run every build-time gate
figures:
	$(PY) src/make_figures.py

## cover: rewrite cover.png from the Grad-CAM figure
cover:
	cd $(REPO) && uv run --with pillow --with pyyaml python scripts/make_cover.py posts/explainability-localization

## render: build the post (all gates run inside and fail the render on mismatch)
render:
	cd $(REPO) && QUARTO_PYTHON="$(PY)" quarto render posts/explainability-localization/index.qmd

## clean: drop the model checkpoint, keep the MNIST download
clean:
	rm -rf data/cache

## distclean: drop the MNIST download too
distclean: clean
	rm -rf data/raw

help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
