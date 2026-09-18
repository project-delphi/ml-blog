# The Anatomy of a Volcano Plot

Source for the blog post *"The Anatomy of a Volcano Plot."*

The post argues that neither effect size nor statistical significance can rank a
high-dimensional screen on its own, builds the two axes that make the pair
readable, and then runs the resulting filter over a real RNA-seq experiment. One
widget lets the reader move the thresholds themselves.

## Contents

```
index.qmd                    the post (Quarto, jupyter engine, Python)
widgets.js                   the screening widget, dependency-free
post.css                   post-scoped stylesheet, compiled from the former theme.scss, included via `css:`
cover.png                    social card, the GSE52778 volcano
requirements.txt             pinned dependencies
data/airway_de_results.csv   DESeq2 results for GSE52778, committed (2.4 MB)
```

## Rendering

The post needs its own virtualenv and a registered Jupyter kernel, per the
convention in `../../AGENTS.md`. Never the system Python.

```sh
uv venv ../../.venv-volcano --python 3.12
uv pip install --python ../../.venv-volcano/bin/python -r requirements.txt
../../.venv-volcano/bin/python -m ipykernel install --user --name volcano-blog
```

Then, from the repository root:

```sh
QUARTO_PYTHON="$(pwd)/.venv-volcano/bin/python" \
  quarto render posts/volcano-plots/index.qmd
```

`QUARTO_PYTHON` is required from a non-interactive shell: Quarto otherwise looks
for the kernel through whatever Python it finds first and fails with
`Jupyter kernel 'volcano-blog' not found`.

### The bundle is a published resource

`widgets.js` is declared under `resources:` and loaded with a `<script src>`,
so a project render republishes it and a widget change needs no re-execution
and no kernel. Edit the file, run `make render`, commit `docs/` with it.

This post used to print the bundle into the page from a Python cell, which put
it outside Quarto's freeze hash: editing the widget left
`../../_freeze/posts/volcano-plots/` valid and a render kept serving the old
copy with no warning. That trap is gone here. `make docs-query ARGS="widget
volcano-plots"` checks the published bundle against the source if you want to
be sure.

## The widget

One widget, in `widgets.js`, **dependency-free**: no CDN, no module loader, no
framework. Observable JS does not work with the Quarto installed here (1.6.40) —
established while building [the Bayesian bootstrap
post](../bayesian-bootstrap/README.md) — so this follows the same hand-rolled
house pattern as [the SVD post](../svd-rotate-stretch-rotate/README.md).

It ships no data. It simulates its own screen of 3,000 hypothesis tests across
four populations (unchanged background, low-count noise, real-but-negligible
shifts, genuine responders), runs a real two-sample $t$-test per feature and a
Benjamini–Hochberg adjustment, and recomputes all of it as the reader moves a
control. The scatter is a `<canvas>`; the controls are plain DOM.

`draw` fixes the populations; `test` draws the replicates. That split matters:
the replicate count has to buy a genuinely tighter estimate, so raising it lifts
the micro-shift band without moving it sideways and pulls the low-count wings
back in. Each feature draws from its own stream in order, so going from 4 to 6
replicates keeps the first four measurements and adds two — the same experiment
continued, not a fresh one.

The simulation core (`VolcanoSim`) touches no DOM and exports itself under
CommonJS, so `node` can drive it directly:

```sh
node -e '
  const S = require("./widgets.js");
  const sample = S.draw(42, 0.3);
  const tested = S.test(sample, 4);
  console.log(S.score(sample, tested, 1.0, 2.0, 0));
'
```

That is how its statistical pieces were checked: the two-sided Student-$t$ tail
agrees with `scipy.stats.t.sf` to 1.5e-12 relative over a grid of $df$ and $t$;
the BH adjustment matches `statsmodels`' `multipletests(method="fdr_bh")` to
2e-16 over 3,000 tests; and the null features come out calibrated — 1.1% of the
truly-unchanged background clears $p \le 0.01$ at four replicates.

The static three-panel figure above the widget makes the same point from the
same four populations, so a reader with scripts disabled loses the interaction
but not the argument. It is drawn in Python with its own seed, so its numbers
and the widget's are independent draws of the same design and will not match
digit for digit.

## Data provenance

The airway RNA-seq results are derived from Himes et al. (2014), *PLoS ONE*
9(6): e99625, GEO accession GSE52778 — primary human airway smooth muscle cells
from four donors, treated with dexamethasone against untreated controls. The
committed CSV is the `DESeq2` differential expression table for 18,028
transcripts. The post's *airway* section carries the full statement.
