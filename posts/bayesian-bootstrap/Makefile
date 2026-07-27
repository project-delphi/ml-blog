# Build for "Why the Bayesian Bootstrap?"
#
# Everything runs through the post's dedicated virtualenv at the repo root, per
# the convention in ../../CLAUDE.md. Never the system Python.

REPO    := $(shell cd ../.. && pwd)
VENV    := $(REPO)/.venv-bayesian-bootstrap
PY      := $(VENV)/bin/python
KERNEL  := bayesian-bootstrap-blog

.PHONY: all venv data mnist widgets render clean distclean help

all: data mnist widgets render

## venv: create the virtualenv and register the Jupyter kernel
venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m ipykernel install --user --name $(KERNEL) \
		--display-name "Python (bayesian-bootstrap)"

## data: download and cache the S&P 500 and IBM return series
data:
	$(PY) -c "import sys; sys.path.insert(0,'src'); import data; \
	r = data.load_returns(); \
	print(f'sp500 {r.sp500.size} months, ibm {r.ibm.size} months, retrieved {r.retrieved}')"

## mnist: train once, cache per-example results, run the weighted likelihood bootstrap
mnist:
	$(PY) src/mnist_experiment.py

## widgets: write the committed widget-data JSON files
widgets:
	$(PY) src/export_widget_data.py

## render: build the post (figure layout QA runs inside and fails the build on any overlap)
render:
	cd $(REPO) && QUARTO_PYTHON="$(PY)" quarto render posts/bayesian-bootstrap/index.qmd

## clean: drop the caches but keep the downloads
clean:
	rm -rf data/cache

## distclean: drop downloads and caches too
distclean: clean
	rm -rf data/raw

help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
