# Why the Bayesian Bootstrap?

Source for the blog post *"Why the Bayesian Bootstrap? Same simplex, different
weights — and where each one breaks."*

The post derives the Bayesian bootstrap from first principles, sets it against
Efron's nonparametric bootstrap, and demonstrates both on S&P 500 / IBM monthly
returns and on MNIST. Every closed form stated in the prose is also computed and
`assert`ed against simulation **at build time**, so the page does not render if
the algebra and the arithmetic disagree.

## Contents

```
index.qmd              the post (Quarto, jupyter engine, Python)
theme.scss             post-scoped theme layered over the site's `zephyr`
widgets.js             the seven interactive widgets, dependency-free
cover.png              social card, the Dirichlet simplex figure
requirements.txt       pinned dependencies
Makefile               data | mnist | widgets | render | all
src/
  bootstrap.py         SEED, palette, weight generators, weighted functionals
  data.py              download + cache the two return series
  mnist_experiment.py  train once + cache; weighted likelihood bootstrap
  export_widget_data.py  writes the committed widget-data/*.json
  layout_qa.py         figure text-collision checker (build gate)
widget-data/           committed JSON, 21.6 KB total — the widgets' data contract
data/raw/              downloads, gitignored
data/cache/            parquet/npz caches, gitignored
```

## Rendering

The post needs its own virtualenv and a registered Jupyter kernel, per the
convention in `../../AGENTS.md`. Never the system Python.

```sh
make venv     # creates ../../.venv-bayesian-bootstrap, installs, registers the kernel
make all      # data -> mnist -> widgets -> render
```

or, from the repository root:

```sh
QUARTO_PYTHON="$(pwd)/.venv-bayesian-bootstrap/bin/python" \
  quarto render posts/bayesian-bootstrap/index.qmd
```

`QUARTO_PYTHON` is required from a non-interactive shell: Quarto otherwise looks
for the kernel through whatever Python it finds first and fails with
`Jupyter kernel 'bayesian-bootstrap-blog' not found`.

## Cost, measured on the build machine

Apple silicon, torch 2.13.0 on the MPS backend.

| step | wall clock | cached? |
|---|---|---|
| download S&P 500 (Shiller CSV, ~90 KB) | ~1 s | `data/raw/` |
| download IBM (`all_stocks_5yr.csv`, 29.6 MB) | ~20 s | `data/raw/` |
| clean + cache both return series | < 1 s | `data/cache/returns.parquet` |
| MNIST download | ~30 s (first run only) | `data/raw/mnist/` |
| MNIST train once + cache per-example results | 4.2 s | `data/cache/mnist_eval.npz` |
| weighted likelihood bootstrap, B = 20 refits | 41 s (2.1 s/fit) | `data/cache/mnist_wlb.json` |
| export widget data | < 1 s | `widget-data/` (committed) |
| **render the post** | **~110 s** | see below |

The render is dominated by the bootstrap simulations: headline univariate results
use `B = 200_000` replicates, figures use `B = 20_000`, and every closed form is
checked against simulation as the page builds.

### A note on `freeze` and the 110 s

`posts/_metadata.yml` sets `freeze: auto`, and this post does write a freeze cache
(`../../_freeze/posts/bayesian-bootstrap/`) with a stable hash. But **Quarto only
consults the freeze cache during a project render** — rendering an individual
document always re-executes it. Measured on a controlled two-document project
with a deliberate 6-second cell:

| | time |
|---|---|
| `quarto render posts/p1/index.qmd` (1st) | 10.5 s |
| `quarto render posts/p1/index.qmd` (2nd, unchanged) | 10.3 s — re-executed |
| `quarto render` (whole project, unchanged) | 1.8 s — freeze hit |

So the "warm cache render in under 60 s" target is met by the mechanism the site
actually publishes through (a project render reuses the frozen output), but not by
the single-file command, which by design always executes. At the time of writing a
whole-project render fails for an unrelated reason — `posts/skills-vs-commands`
pins a Jupyter kernel that is not registered locally — which is pre-existing and
independent of this post.

## Data provenance

Both series were retrieved on **2026-07-26**.

**S&P 500** — `datasets/s-and-p-500`, the Shiller monthly series.
Each monthly level is the *average of that month's daily closes*, not a month-end
close, so monthly volatility is slightly understated. As a download check the post
asserts that the annualised volatility from Feb 1990 onward is ≈ 12.3%; the value
obtained was **12.29%**.

**IBM** — `plotly/datasets/all_stocks_5yr.csv`, daily closes 2013-02-08 to
2018-02-07, filtered to `Name == 'IBM'`, resampled to month-end. These are
*unadjusted* closes and therefore **price** returns: dividends are excluded, which
understates total return by roughly 3–4%/yr over this window. The incomplete final
month (2018-02) is dropped.

If a download fails, `src/data.py` raises with the URL. There is deliberately **no
fallback to synthetic data**.

### One number that differs from the brief

The post was written against a specification whose long-window row read
`n = 438` for the S&P 500 from Feb 1990. The live Shiller series gives
**n = 437** (Feb 1990 – Jun 2026), which shifts that row's standard errors in the
fourth decimal (0.1693 / 0.1695 / 0.1697 rather than 0.1691 / 0.1693 / 0.1695).
The computed value wins: every figure in the post is interpolated from the code,
not typed in. Every other verified value in the brief reproduced exactly —
including the full IBM and S&P sample descriptions, the median sign-test interval,
the tail atom, and the two-sample comparison.

## Widgets

Seven widgets (W1–W7), all in `widgets.js`. They are **dependency-free**: no CDN,
no module loading, no framework — just the DOM and inline SVG, including
hand-rolled `lgamma`, regularised incomplete beta, Beta quantiles and a seeded
LCG. Their data is inlined into the page at render time from the committed
`widget-data/*.json` files, so the widgets need no `fetch` and work from a plain
`file://` page as well as over HTTP.

The widgets do their own resampling in the browser; no precomputed replicate
arrays are shipped. Total widget data: **21.6 KB**, against a 1 MB budget.

### Why not Observable JS

The brief specified Observable JS, and the post was first built that way. OJS
does not work with the Quarto version installed here (1.6.40, bundling an OJS
runtime dated January 2025): the Observable scheduler wedges on load with the
entire stdlib — `FileAttachment`, `Inputs`, `Plot`, 47 variables — queued and
never computed, so *every* cell renders empty. This reproduces on a document whose
only content is `x = 41 + 1`, so it is not a property of this post's code.

A second, independent Quarto bug was found along the way and is worth recording:
an `{ojs}` cell containing **more than one top-level statement** emits DOM ids
`ojs-cell-1-1`, `ojs-cell-1-2`, … while registering the cell as `ojs-cell-1`, so
the runtime's element lookup returns `null`, and because the failure happens
inside the error handler it aborts every later cell too. One statement per cell
avoids that one.

Rewriting the widgets without Observable removed both problems and, as a bonus,
made them work offline.

## Reproducibility

One seed, `SEED = 20260726` in `src/bootstrap.py`. Every experiment derives its
own independent generator from it *by name* (`rng_for("median")`), so adding an
experiment never shifts the stream of an existing one.

Build-time gates, all of which fail the render:

- every closed form in the post is `assert`ed against simulation, with tolerances
  set by the Monte-Carlo standard error;
- exact enumerations where they are cheap — all 126 Efron patterns at `n = 5`
  summing to 1, the conditioned-Poisson/multinomial identity at `n = 3, 4, 5` to
  machine precision, the 35 equiprobable Dirichlet-multinomial patterns at `n = 4`;
- `src/layout_qa.py` walks every text artist in every figure and fails on any pair
  overlapping by more than ~60 px². Current status: **10 figures, 0 collisions.**

## Accessibility and appearance

The post ships **light-only**, deliberately: the figures encode meaning in hue and
sit on the paper ground. `:root { color-scheme: light }` opts out of Chrome's
"Auto Dark Mode", which would otherwise invert the page background while leaving
the figures and widget SVGs light. Readers who have Chrome's *force*-dark flag
enabled will still see a darkened page — that flag overrides `color-scheme` and
affects every light site equally.

Contrast against the paper ground `#F1F2EE` (WCAG AA needs 4.5:1 for body text):

| colour | role | ratio |
|---|---|---|
| `#171C1B` ink | body text | 15.33 |
| `#7A3B6B` plum | observed data | 7.01 |
| `#1D5C6E` petrol | Efron | 6.65 |
| `#8A5E06` dark ochre | Bayes, in running text | 5.07 |
| `#67706E` muted | captions | 4.53 |

`#C98A12` ochre is used for Bayes **in figures only** and never for body text; it
does not meet a text contrast threshold, which is why the darker `#8A5E06` exists.
The specified muted grey `#6E7876` gave 4.05:1 — fine for large text, short of AA
at the 0.84 rem caption size actually used — so it was darkened by 3% lightness
with the hue preserved.

At a 375 px viewport the twin comparison blocks stack, and wide tables and long
display equations scroll inside their own boxes rather than widening the page.

## Related

[The Poor Person's Bayesian](../poor-persons-bayesian/) is the survey-level
companion: what the bootstrap-as-cheap-Bayes analogy buys you, and the point at
which it stops being true.
