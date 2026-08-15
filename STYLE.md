# Blog Writing Style

## Where these rules apply

These rules govern new posts, and edits to posts that already carry their own
venv, pinned kernel, and `requirements.txt`.

They do **not** license a rewrite of the five legacy posts — `data-types`,
`features-importance-after-clustering`, `poor-persons-bayesian`,
`post-with-code`, `working-with-quarto` (the `LEGACY_NO_ENV` set in
`scripts/check_posts.py`). Quarto keys frozen output on an md5 of the source,
so even a one-word prose fix there invalidates the `_freeze/` record and makes
the next project render try to execute a post that has no environment to
execute in. To bring one of those up to this style, build it a venv + kernel +
`requirements.txt` and drop its `LEGACY_NO_ENV` entry first, as CLAUDE.md
describes.

Nothing here is a mandate to sweep the existing corpus. Apply it to what you
are already writing.

## Structure
- The first time a section reaches for a new idea, mechanism, or dataset,
  write 2–4 sentences of plain-language warm-up before the code block,
  diagram, or example: what problem this solves, why it matters, and what
  the reader should hold in mind before seeing the mechanics.
- After that, one sentence before and one after carries it — a follow-up
  cell, a variation on the last one, a plot of a result you already
  motivated. If a reader can predict what the block does from the prose
  above it, one sentence is the right amount.
- Never open a section with a code block, bullet list, or table. Give
  context first, in prose.
- Introduce a concept from first principles before naming its technical
  term — explain the idea, then attach the label to it, not the other
  way around.

## Sentences and words
- Prefer short, direct sentences. Split the ones that stack clauses until
  the reader loses the subject before reaching the verb — two or more
  nested subordinate clauses is the usual tell.
- This is not a ban on the long sentence, and it is not a licence to
  flatten the house voice. Clauses joined by em-dashes to gloss a term or
  hang a concrete case off an abstract claim read as one clear beat; both
  calibration examples below do it. Keep those. The target is the sentence
  you have to read twice, not the sentence that is merely long.
- Editing an existing post, match its register. These rules tighten prose;
  they do not convert a post to a different voice.
- Use everyday words over formal/Latinate ones where a simpler word
  exists: "use" not "utilize," "help" not "facilitate," "start" not
  "commence," "about" not "approximately."
- Default to active voice ("the function returns X") over passive
  ("X is returned by the function").
- One idea per paragraph. If a paragraph covers two ideas, split it.

## Tone
- Write like you're explaining this to someone smart but unfamiliar
  with the topic — not like documentation for someone who already
  knows the shape of the solution.
- Avoid jargon without a plain-language restatement nearby the first
  time it's used.

## Calibration examples

Match this tone and pacing — motivate the problem, define the term
plainly, then move to mechanics:

**Example: cross-validation**
A model can fit the data it was trained on almost perfectly and still
fail on new data — this is the gap between memorizing and generalizing.
Cross-validation is a way to estimate how a model will perform on unseen
data using only the data you already have. The idea: split your dataset
into k equal parts ("folds"). Train the model on k-1 of them, and test
it on the one you held back. Repeat this k times, rotating which fold
is held out, then average the results.

**Example: the bootstrap**
When you compute a statistic from a sample — a mean, a regression
coefficient — you get one number, but you rarely know how much that
number would vary if you'd drawn a different sample. The bootstrap
estimates that variability without new data: it repeatedly draws new
samples from your original sample, with replacement, and recomputes
the statistic each time. The spread of those recomputed values
approximates the sampling distribution you'd otherwise need new data
to observe.

Avoid: opening with opinion, meandering into the topic before defining
it, assuming the reader already has context, or diving into code/examples
before explaining the problem in plain language.
