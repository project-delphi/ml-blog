# Explainability Is a Localization Problem

Source for the blog post *"Explainability Is a Localization Problem: saliency maps
on MNIST, confidence intervals on Iris."*

The post frames explainability as attributing an outcome to the parts responsible
for it, and separates eight methods along three axes: what counts as a part, what
counts as responsible, and whether the answer comes with error bars. Part I
implements saliency, occlusion, Grad-CAM and integrated gradients against a small
CNN on MNIST; Part II implements logistic coefficients with hand-derived standard
errors, one-way ANOVA, PCA and permutation importance on Fisher's irises.

Every code block in the post is extracted from these modules with
`inspect.getsource` at render time. There is no second, prettified copy of any
method, so the page cannot drift from the code that ran.

## Contents

```
index.qmd              the post (Quarto, jupyter engine, Python)
cover.png              social card, the Grad-CAM figure
requirements.txt       pinned direct dependencies
Makefile               venv | model | figures | cover | render | all
figures/               committed PNGs, one per figure in the post
src/
  house.py             SEED, palette, plot helpers, the inspect.getsource echo
  mnist_model.py       the CNN, trained once and cached; running-example choice
  attribution.py       saliency, occlusion, Grad-CAM, CAM, integrated gradients
  iris_stats.py        logistic fit + Wald inference, ANOVA, PCA, permutation
  figures.py           every figure, as a function returning a Figure
  make_figures.py      writes figures/*.png and runs every build gate
data/raw/              MNIST download, gitignored
data/cache/            model checkpoint, gitignored
```

## Re-running it

The post needs its own virtualenv and a registered Jupyter kernel, per the
convention in `../../AGENTS.md`. Never the system Python.

```sh
make venv       # creates ../../.venv-explainability, installs, registers the kernel
make figures    # trains the CNN on first run, then writes every PNG and runs every gate
make render     # builds the post
```

or, from the repository root:

```sh
QUARTO_PYTHON="$(pwd)/.venv-explainability/bin/python" \
  quarto render posts/explainability-localization/index.qmd
```

`QUARTO_PYTHON` is required from a non-interactive shell: Quarto otherwise looks
for the kernel through whatever Python it finds first and fails with
`Jupyter kernel 'explainability-blog' not found`.

## Cost, measured on the build machine

Apple silicon, torch 2.13.0, training on CPU.

| step | wall clock | cached? |
|---|---|---|
| MNIST download | ~30 s (first run only) | `data/raw/mnist/` |
| train the CNN, 4 epochs | 247 s (first run only) | `data/cache/cnn.pt` |
| all four attribution methods, both examples | < 1 s | no |
| every figure in the post | ~10 s | `figures/` |
| **render the post** | **~35 s warm** | see `_freeze/` |

Training is on CPU rather than MPS deliberately. The accelerator is roughly four
times faster here, but its kernels are not bit-reproducible across releases, and
the post names a specific misclassified test digit in its prose. A running
example that silently changes identity between renders is worse than a slower
build.

## Data provenance

**MNIST** — `torchvision.datasets.MNIST`, which fetches the LeCun/Cortes/Burges
redistribution of NIST Special Databases 1 and 3. SD-3 was written by Census
Bureau employees, SD-1 by American high-school students; the MNIST split mixes
both into train and test, unlike the original NIST split. Glyphs are
size-normalised into a 20×20 box and centred by mass in a 28×28 field, which is
why the pixels are anti-aliased greys rather than binary — a fact the post's
treatment of baselines depends on.

**Iris** — `sklearn.datasets.load_iris`, which ships Anderson's 1935 measurements
(50 flowers each of *Iris setosa*, *versicolor* and *virginica*, four
measurements in centimetres) as used in Fisher's 1936 paper introducing linear
discriminant analysis. The post states, rather than eliding, that the paper
appeared in the *Annals of Eugenics* and that Fisher was a eugenicist.

Which copy matters: two rows of the [UCI version](https://archive.ics.uci.edu/dataset/53/iris)
disagree with Fisher's published table. `sklearn` ships the corrected copy — its
own description says so, and it matches R's `datasets::iris`. Nothing in the post
turns on those two flowers, but the two versions are in circulation and a
reproduction against the UCI file will differ in the fourth decimal.

`sklearn`'s description also states outright that one class is linearly separable
from the other two and that those two are not separable from each other. The post
does not take its word for it; `iris_stats.is_separable` solves the feasibility
program and confirms both halves at build time.

## Reproducibility

One seed, `SEED = 20260731` in `src/house.py`. Experiments derive independent
generators from it *by name* (`rng_for("permutation")`), so adding an experiment
never shifts the stream of an existing one. Model training seeds `torch` directly
and runs on CPU.

Ten distinct gates run while the page builds — 14 assertions in total, because the
four Part I checks run separately for each of the two running examples. Each of
them fails the render:

**Part I** (`make_figures.mnist_gates`, run for both examples)

- integrated gradients satisfy completeness — the attributions sum to
  `f(x) − f(baseline)` — to better than `1e-3` relative;
- Grad-CAM computed through a forward hook is *bit-identical* to Grad-CAM
  computed through the model's `features`/`head` split;
- on this GAP-headed network, Grad-CAM equals Zhou et al.'s CAM divided by the 49
  cells of the 7×7 grid, to better than `1e-5` absolute;
- integrated gradients assign exactly zero to every pixel equal to the baseline;
- the largest saliency value in the map lands on a baseline-valued pixel. Unlike
  the others this is a fact about the checkpoint rather than a theorem, and it is
  gated because the post asserts it in prose: a re-render that moves the maximum
  onto the stroke should fail the build rather than silently falsify the text.

**Part II** (`iris_stats.self_check`)

- hand-derived logistic coefficients and standard errors match `statsmodels` to
  ~`1e-15`;
- hand-computed one-way ANOVA F statistics match `scipy.stats.f_oneway`;
- hand-computed PCA explained-variance ratios match `scikit-learn`;
- setosa is confirmed linearly separable, and versicolor/virginica confirmed not,
  by solving the feasibility program rather than inferring it from a diverging
  fit;
- hand-rolled permutation importances agree with `sklearn.inspection`'s within
  Monte-Carlo error.

The build-gate output is reproduced in a collapsible callout at the top of the
post, generated from the same call.

## Related

[Why the Bayesian Bootstrap?](../bayesian-bootstrap/) is the companion on the
uncertainty side: what it takes to put an honest interval around a quantity
computed from data, which is the column this post argues is still empty for most
attribution methods.
