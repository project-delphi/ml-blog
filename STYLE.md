# Blog Writing Style

## Which register

Pick one before writing or editing a post.

- **Register A — personal / old.** Claim headings, narrative spine, plain-English warm-up. Use it for personal essays, memoirs, and any post dated before 2025-08-28.
- **Register B — recent technical.** Dry, numbered, list-first documentation. Use it for technical posts dated 2025-08-28 or later, and for new technical posts.

Personal posts currently in Register A regardless of date: `cheapest-ladder-is-shortest`, `brian-ripley-rousseeuw-prize`, `nvidia-buys-the-pyg-team`, `first-industry-job`, `consciousness-recursive-prediction`. Do not convert those to Register B.

These rules do **not** license a rewrite of the five legacy posts — `data-types`, `features-importance-after-clustering`, `poor-persons-bayesian`, `post-with-code`, `working-with-quarto` (the `LEGACY_NO_ENV` set in `scripts/check_posts.py`). Quarto keys frozen output on an md5 of the source, so even a one-word prose fix there invalidates the `_freeze/` record and makes the next project render try to execute a post that has no environment to execute in. To bring one of those up to either register, build it a venv + kernel + `requirements.txt` and drop its `LEGACY_NO_ENV` entry first, as CLAUDE.md describes.

The purpose sentence and one-word closer below have been applied to the 50 posts with no `_freeze/` record (42 `.qmd`, 8 `.ipynb`) — the ones a project render rebuilds without executing anything. The other 51 posts, the freeze-backed ones, have **not** been swept: editing one invalidates its frozen output and forces a re-execution, so that pass has to be done kernel by kernel with the venvs to hand. `tensor-factorizations` and `uses-of-tensor-factorizations` are the exceptions — they already carry both, being where the convention started. Count the two sets before sizing that sweep (`ls -d posts/*/`, `ls -d _freeze/posts/*/`) rather than trusting these numbers, which go stale with every new post. Do not assume the corpus is uniform yet.

## Shared rules (both registers)

- **Purpose sentence first.** The first prose sentence of the body says what the post is for, in plain English — what question it answers, or what the reader will be able to do afterwards. It sits immediately after the `![Title](./cover.png)` line (and after any "Made with…" credit block), before the first heading. This is not a licence for the `In this post we will…` / `This article explores…` formula: say the thing, don't announce that you are about to. Where a post already opens on a sentence that does this, leave it.
- **One-word closer last.** Every post ends on a single line of one-word sentences that summarises it and lands its consequence: eight to twelve tokens, each a single capitalised word ending in a period, reading as two to four telegraphic clauses. It is a *new* final line — the existing last paragraph stays. It goes before `## References` where one exists.

  ```
  Tensors. Outrun. Matrices. Factorizations. Compress. Inverses. Require. Products. Choose. First.
  Don't. Flatten. Modes. Carry. Meaning. Ranks. Trade. Memory. Factorizations. Almost. Always.
  ```

  It must be true to *this* post's argument. A closer that merely restates the title is worse than none.
- **Never invent first-person detail.** Personal essays land through specifics the author actually has. Write a scene in the second person or as a hypothetical someone; do not fabricate memories or citations.
- **Always situate the data.** Any post that uses, plots, or mentions a dataset must say where it came from before doing anything with it: provenance, collector and motive, what this post asks of it, what a wrong answer would cost, and why this method fits this data. Synthetic data is not exempt: say it is synthetic, give the generating process, and say what real situation it stands in for.
- **Do not change executable cell code** when restyling prose. Widgets, freeze figures, and `{python}` / `{pyodide}` bodies stay as they are.
- **References last.** If the post cites papers, books, docs, datasets, or other posts as sources, end with a `## References` section: a short bulleted list of those sources (author/year or title + link). Collect only sources the post already names or links — do not invent citations. Skip the section when there are none. This heading is an allowed topic label in Register A. Inline links in the body may stay; the list at the end is the bibliography.

---

## Register A — personal / old

Apply this register to what you are already writing in that voice. It is not a mandate to sweep the older corpus.

### Spine

**Give every post a spine.** A post is one argument, not a pile of sections. Name the core idea in a sentence before writing; every section advances it.

- **Open on the core idea and the stakes** — the question, why it's non-obvious, what changes once the reader knows. The purpose sentence (see Shared rules) carries this and comes first; it is still not a "in this post we will…" preamble, and in an essay it stays in the essay's voice rather than sliding into guide voice.
- **Headings state claims, not topics** — `## Efron's bootstrap is a weighted bootstrap in disguise`, not `## Background`, so the ToC reconstructs the argument.
- **Each section earns the next.** If two could swap without damage, merge or cut one. `###` is for steps within one idea, not new ideas.
- **Stitch every seam.** A section's first sentence links back to the previous result or the core idea; its last names the unresolved thing the next section answers — the gap, not the mechanics ("next, some code"). Same for code blocks and figures: a sentence before saying what it will show, one after saying what actually happened, quoting the numbers it produced. If an opening sentence reads identically with the previous section deleted, the seam isn't stitched.
- **Close by returning to the core idea** — restate the opening claim now that it's earned, what it buys, where it stops holding. Not a summary of sections. The one-word closer (see Shared rules) is the line after that close; if the post has sources, `## References` follows both.
- **Caveats inline**, as a short section where the objection occurs (`## Caveat: the uniform is the posterior, not the prior`) — not a "Limitations" bin at the end.

### Structure

- The first time a section reaches for a new idea, mechanism, or dataset, write 2–4 sentences of plain-language warm-up before the code block, diagram, or example: what problem this solves, why it matters, and what the reader should hold in mind before seeing the mechanics.
- After that, one sentence before and one after carries it — a follow-up cell, a variation on the last one, a plot of a result you already motivated. If a reader can predict what the block does from the prose above it, one sentence is the right amount.
- Never open a section with a code block, bullet list, or table. Give context first, in prose.
- Introduce a concept from first principles before naming its technical term — explain the idea, then attach the label to it, not the other way around.

### Sentences and words

- Prefer short, direct sentences. Split the ones that stack clauses until the reader loses the subject before reaching the verb — two or more nested subordinate clauses is the usual tell.
- This is not a ban on the long sentence, and it is not a licence to flatten the house voice. Clauses joined by em-dashes to gloss a term or hang a concrete case off an abstract claim read as one clear beat; both calibration examples below do it. Keep those. The target is the sentence you have to read twice, not the sentence that is merely long.
- Editing an existing Register A post, match its register. These rules tighten prose; they do not convert a post to a different voice. Personal essays especially must not drift into guide voice — claims turning into requirements lists, headings into section labels.
- Use everyday words over formal/Latinate ones where a simpler word exists: "use" not "utilize," "help" not "facilitate," "start" not "commence," "about" not "approximately."
- Default to active voice ("the function returns X") over passive ("X is returned by the function").
- One idea per paragraph. If a paragraph covers two ideas, split it.

### Tone

- Write like you're explaining this to someone smart but unfamiliar with the topic — not like documentation for someone who already knows the shape of the solution.
- Avoid jargon without a plain-language restatement nearby the first time it's used.

### Calibration examples

Match this tone and pacing — motivate the problem, define the term plainly, then move to mechanics:

**Example: cross-validation**
A model can fit the data it was trained on almost perfectly and still fail on new data — this is the gap between memorizing and generalizing. Cross-validation is a way to estimate how a model will perform on unseen data using only the data you already have. The idea: split your dataset into k equal parts ("folds"). Train the model on k-1 of them, and test it on the one you held back. Repeat this k times, rotating which fold is held out, then average the results.

**Example: the bootstrap**
When you compute a statistic from a sample — a mean, a regression coefficient — you get one number, but you rarely know how much that number would vary if you'd drawn a different sample. The bootstrap estimates that variability without new data: it repeatedly draws new samples from your original sample, with replacement, and recomputes the statistic each time. The spread of those recomputed values approximates the sampling distribution you'd otherwise need new data to observe.

Avoid: opening with opinion, meandering into the topic before defining it, assuming the reader already has context, or diving into code/examples before explaining the problem in plain language.

---

## Register B — recent technical

Optimize for rapid transfer of technical information. Do not try to make the text engaging.

### Tone

- Strictly utilitarian, objective, and dry.
- No narrative flair, storytelling, poetic language, or conversational transitions.
- No "in this post we will…" preamble. Start with the definition or procedure.

### Headings

- Use **topic labels**, not claims: `## Eigendecomposition`, not `## $A = PDP^{-1}$ is that idea written as a formula`.
- Enable Quarto numbering in frontmatter. Do **not** put `1.` / `1.1` in the heading text (that double-numbers):

```yaml
format:
  html:
    toc: true
    number-sections: true
```

- `##` is a top-level topic. `###` is a subtopic of the current `##`.

### Structure

- Default to bulleted or numbered lists whenever presenting multiple concepts, pros/cons, or sequential steps.
- Remaining paragraphs: one or two sentences.
- Isolate code, commands, and display math in fenced blocks.
- End when the information is delivered. No *paragraph* closer that restates an opening claim — the one-word closer (see Shared rules) is the single exception, and it is required. The last section is `## References` when the post has sources, with the closer immediately above that heading.
- Caveats go in a labelled section (`## Constraints`, `## Guarantees`) where they belong, not as a narrative return.

### Calibration fragment

```markdown
## Eigenvectors and eigenvalues

Multiplying by a matrix $A$ usually rotates and scales a vector.

- **Eigenvector** $v$: a direction that $A$ only scales: $Av = \lambda v$.
- **Eigenvalue** $\lambda$: the scale factor along $v$.

## Example matrix

This post uses the synthetic matrix

$$
A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}.
$$

It is a two-class linear map chosen so the eigenvalues are exactly $5$ and $2$.
```
