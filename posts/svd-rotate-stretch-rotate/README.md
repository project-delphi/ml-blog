# The Matrix That Rotates, Stretches, and Rotates Again

Source for the blog post *"The Matrix That Rotates, Stretches, and Rotates
Again: an intuitive tour of the SVD."*

The post starts from what a matrix does to the unit circle, derives
$A = U\Sigma V^\mathsf{T}$ from that picture, and then spends the singular
values three ways: as a compression dial on a real photograph, as a
conditioning diagnostic on a NIST certification dataset, and as a capacity knob
on MovieLens ratings. It then argues that the rest of the SVD's applications --
PCA, latent semantic analysis, pseudoinverses, model reduction -- are those same
three readings of the spectrum aimed at other matrices, and demonstrates the
sharpest case on a voice signal, where a trajectory matrix has an exactly known
rank and truncating to it lifts a synthetic vowel out of equal-parts noise. Two
widgets let the reader drive the maths directly.

## Contents

```
index.qmd              the post (Quarto, jupyter engine, Python)
widgets.js             both widgets, dependency-free
theme.scss             post-scoped theme layered over the site's `zephyr`
cover.png              social card, the rotate-stretch-rotate figure
requirements.txt       pinned dependencies
Makefile               data | widgets | cover | render | all
assets/Filip.dat       the NIST StRD Filip file, committed verbatim (4 KB)
widget-data/           committed JSON, 452 KB -- the widgets' data contract
src/
  data.py              fetch + cache Filip and MovieLens
  imagery.py           the photograph, its SVD, PSNR and storage accounting
  movielens.py         ratings matrix, held-out split, RMSE against rank
  voice.py             synthetic vowel, trajectory matrix, subspace denoising
  export_widget_data.py  writes the committed widget-data/*.json
data/raw/              MovieLens download, gitignored
```

## Rendering

The post needs its own virtualenv and a registered Jupyter kernel, per the
convention in `../../AGENTS.md`. Never the system Python.

```sh
make venv     # creates ../../.venv-svd, installs, registers the kernel
make all      # data -> widgets -> cover -> render
```

or, from the repository root:

```sh
QUARTO_PYTHON="$(pwd)/.venv-svd/bin/python" \
  quarto render posts/svd-rotate-stretch-rotate/index.qmd
```

`QUARTO_PYTHON` is required from a non-interactive shell: Quarto otherwise looks
for the kernel through whatever Python it finds first and fails with
`Jupyter kernel 'svd-blog' not found`.

## Regenerating the cached data

Everything committed under `assets/` and `widget-data/` is reproducible:

```sh
make data       # ensures assets/Filip.dat and data/raw/ml-100k.zip exist
make widgets    # rewrites widget-data/*.json from those two sources
make cover      # rewrites cover.png
```

`src/voice.py` is the exception to the pattern above: it needs no cached data
and no `make` target, because the signal is generated inside the render from a
seeded RNG. Changing `F0`, `N_HARM` or `SEED` changes the published numbers, and
the post quotes them inline, so re-render after touching it.

`make widgets` re-runs the MovieLens experiment (about 40 s, dominated by one
943 × 1682 dense SVD) and re-derives the photograph's leading 100 singular
triplets. It fails if the payload exceeds its 1 MB budget.

The render itself never touches the network. `assets/Filip.dat` is committed and
the photograph ships inside scikit-image, so only `make data` needs
connectivity, and only for MovieLens.

### The freeze trap

Quarto keys frozen output on an md5 of `index.qmd` **alone**. Editing
`widgets.js` or `widget-data/*.json` does not invalidate
`../../_freeze/posts/svd-rotate-stretch-rotate/`, so a project render will keep
serving the old bundle. After changing either, re-render this post explicitly
before committing.

## Widgets

Two, both in `widgets.js`, both **dependency-free**: no CDN, no module loader,
no framework.

- **W1 — your matrix, your ellipse.** Inline SVG. Computes a real 2×2 SVD in
  the browser, in closed form, by diagonalising $A^\mathsf{T}A$ — the same
  argument the post makes in prose. Ships no data at all.
- **W2 — the rank dial.** A `<canvas>` fed by `ImageData`. The page ships the
  photograph's leading 100 singular triplets as quantised int16 (450 KB) and
  rebuilds any rank in the browser. Moving the slider one step is a single
  rank-one update of an accumulator, so a dragged slider keeps up.

Quantising the singular vectors to int16 is invisible: dequantised
reconstructions reproduce the shipped PSNR figures to three decimal places.

Data reaches them inlined in the page as `<script id="svd-data">`, written at
render time from the committed JSON, so the widgets need no `fetch` and work
from a plain `file://` page as well as over HTTP.

### Why not Observable JS

The brief asked for OJS. It does not work with the Quarto version installed here
(1.6.40) — the Observable scheduler wedges on load with the entire stdlib queued
and never computed, so every cell renders empty. This was established
independently while building [the Bayesian bootstrap
post](../bayesian-bootstrap/README.md), whose widgets are hand-rolled for the
same reason. Both widgets here follow that house pattern.

## Accessibility and appearance

The post ships **light-only**, deliberately: the figures encode meaning in hue
and the photograph is a grayscale bitmap. `:root { color-scheme: light }` opts
out of Chrome's "Auto Dark Mode", which would otherwise invert the page
background while leaving the figures and the widget SVG light.

Both widgets are preceded by a static figure making the same point, so a reader
with scripts disabled loses the interaction but not the argument.

## Data provenance

See the post's *Data and attribution* section for the full statement. In short:

| source | licence | committed? |
|---|---|---|
| `skimage.data.astronaut()` — NASA, Eileen Collins | public domain | ships inside scikit-image |
| NIST StRD `Filip.dat` | public domain (US federal government) | yes, `assets/Filip.dat` |
| MovieLens 100k, GroupLens / U. Minnesota | redistribution **not** permitted | no — downloaded on demand |
| synthetic vowel, `src/voice.py` | n/a — generated at render, seeded | no data to commit |

## Related

[Matrix Factorizations as Optimization Problems](../matrix-factorizations/)
covers the same decomposition from the optimisation side, including the
Eckart–Young result this post cites. [The Directions a Matrix Refuses to
Turn](../eigendecomposition/) is the eigendecomposition companion whose spectral
theorem section 3 leans on.
