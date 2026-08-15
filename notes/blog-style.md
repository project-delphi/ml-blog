# Blog Writing Style

## Structure
- Before any code block, diagram, or example, write 2–4 sentences of
  plain-language warm-up: what problem this solves, why it matters, and
  what the reader should understand before seeing the mechanics.
- Never open a section with a code block, bullet list, or table. Give
  context first, in prose.
- Introduce a concept from first principles before naming its technical
  term — explain the idea, then attach the label to it, not the other
  way around.

## Sentences and words
- Prefer short, direct sentences. If a sentence has more than one
  subordinate clause, split it.
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
