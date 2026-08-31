# Post environments

How to build, rebuild, and reason about the per-post Python environments. `AGENTS.md`
has the short version; this file has the recipes and the reason each step is there.

Never install with a bare `pip` — that hits the system Python or fails outright on an
externally-managed one. Name the interpreter every time.

## Three tiers of post

`pyproject.toml` carries only dev/lint tooling. No numpy, sklearn, or torch.

1. **Executes at render** — its own gitignored `.venv-<slug>`, a named Jupyter kernel,
   `jupyter: <kernel-name>` in the frontmatter, and a committed
   `posts/<slug>/requirements.txt`. Count them with `ls posts/*/requirements.txt`.
   Venv and kernel names are historical abbreviations and often do **not** match the
   slug (`convex-optimization-interior-point-methods` → `.venv-ipm` / `ipm-blog`;
   `sir-training-vs-calibration` → `.venv-sir` / `sir-blog`). Read the post's
   `jupyter:` field rather than guessing.
2. **Displays code only** — every cell `#| eval: false`, pinning `blog-base`, a shared
   kernel over the base `.venv` (`make install && make kernel`). The Claude-API posts
   are these. Quarto still needs *some* working kernel to structurally process
   `{python}` cells even when nothing runs. Mermaid and other diagram blocks go
   through Quarto's own filters and need no kernel.
3. **Assets built ahead of the render** — a `posts/<slug>/src/` of scripts you run
   yourself, output committed. See the last section.

`.ipynb` posts are never executed by Quarto; their stored cell outputs are used as-is,
which is why none of them have a `_freeze/` record.

## Building a post venv

```bash
uv venv .venv-<slug>
uv pip install --python .venv-<slug>/bin/python <post deps> ipykernel jupyter nbclient nbformat pyyaml
.venv-<slug>/bin/python -m ipykernel install --user --name <kernel-name>
QUARTO_PYTHON="$(pwd)/.venv-<slug>/bin/python" quarto render posts/<slug>/index.qmd
uv pip freeze --python .venv-<slug>/bin/python > posts/<slug>/requirements.txt
```

Every line fixes a real failure:

- `ipykernel` alone is not enough. Without Quarto's execution stack (`jupyter
  nbclient nbformat pyyaml`) a render dies with `ModuleNotFoundError: No module named
  'yaml'` from Quarto's `jupyter.py` shim.
- Without `QUARTO_PYTHON`, `quarto render` resolves a Python that cannot see a
  `--user`-registered kernel and fails with `Jupyter kernel '<name>' not found. Known
  kernels: python3`.
- The freeze is not optional. `requirements.txt` is what lets a re-render be checked
  against the versions that produced the published output.

**Also add the new kernel name to `kernels-stub` in the `Makefile`.** A name missing
there fails a project render on any fresh clone, since kernelspecs resolve before
`_freeze/` is consulted. `make check-posts` enforces this.

If a render picks up the wrong environment, `jupyter kernelspec list` and the kernel's
`argv[0]` say which interpreter it actually resolved to.

Venvs predating uv still work, driven by `.venv-<slug>/bin/python -m pip` (a `uv venv`
has no `pip`). Recreate one in place from its lockfile to migrate — kernel specs store
an absolute path, so reusing the directory name keeps the registration valid.

## The Hugging Face posts share two venvs, not thirteen

Thirteen posts (`datasets`, `hub`, `tokenizers`, `peft`, `transformers-library`,
`hugging-face-evaluate-library`, `optimum`, `setfit`, `diffusers`,
`classifying_text_chunks`, `langchain`, `evaluation-metrics`, `soft-vs-hard-prompts`)
share ~90% of their dependency closure, most of it `torch`. Each venv is ~1.2 GB and
`uv` copies rather than hardlinks here, so thirteen would be ~15 GB of near-identical
wheels. They pin one of two shared kernels:

- **`huggingface-blog`** over `.venv-huggingface` — `transformers` 5.x. Eleven posts.
- **`huggingface-t4-blog`** over `.venv-huggingface-t4` — `transformers` 4.57.x. Two
  posts, `optimum` and `setfit`, only because those packages cap it. `optimum-onnx`
  declares `transformers<4.58.0`; `setfit` declares no upper bound but imports
  `transformers.training_args.default_logdir`, which 5.x removed — so it *resolves*
  against 5.x and then fails at import. A resolver-only check misses that: import the
  packages before trusting a resolve.

`requirements.txt` records a freeze, not intent, so the package sets live here:

```bash
uv venv .venv-huggingface --python 3.12
uv pip install --python .venv-huggingface/bin/python \
  torch transformers datasets tokenizers huggingface_hub evaluate peft \
  sentence-transformers diffusers accelerate safetensors \
  rouge_score sacrebleu nltk absl-py bert_score scikit-learn numpy pandas matplotlib Pillow \
  torchvision opencv-python-headless \
  pypdf langchain langchain-core langchain-text-splitters langchain-huggingface \
  ipykernel jupyter nbclient nbformat pyyaml
.venv-huggingface/bin/python -m ipykernel install --user --name huggingface-blog

uv venv .venv-huggingface-t4 --python 3.12
uv pip install --python .venv-huggingface-t4/bin/python \
  "optimum-onnx[onnxruntime]" setfit transformers torch datasets evaluate \
  sentence-transformers scikit-learn sentencepiece \
  ipykernel jupyter nbclient nbformat pyyaml
.venv-huggingface-t4/bin/python -m ipykernel install --user --name huggingface-t4-blog
```

`setfit` is deliberately absent from the first set: it resolves fine there and then
breaks the kernel at import.

The trade: for these thirteen, `requirements.txt` is a copy of a shared freeze, so
bumping a dependency for one post changes the versions recorded for every post on that
kernel. `_freeze/` hashes source only, so nothing silently re-executes — but committed
versions and the versions that produced a post's output can drift apart.
**Re-render every post on a kernel when you bump that kernel's venv.**

## One post calls a live API

`llm-agents-from-first-principles` runs an LLM agent against Groq's free tier, so
re-executing it needs `GROQ_API_KEY`. Its `.venv-llm-agents` holds only Quarto's
execution stack, since `agent.py` is stdlib-only. A *project* render never re-executes
it, so a keyless clone renders the site fine; only a single-document render needs the
key.

Its captured transcripts are one sample from a stochastic policy, not a reproducible
fixed point: re-rendering legitimately changes published output even at
`temperature=0`. Don't re-render it to "refresh" anything, and never hand-edit the
transcripts.

## `posts/<slug>/src/` — assets built outside the render

Some posts carry a `src/` of scripts that run *ahead of* the render, with their output
committed: figure generators (`metagenomics`, `voice-ai-architectures-2026`), audio
generators (`open-source-tts-history`, `llm-agent-memory`, which write the committed
`audio/*.mp3`), photo preparation (`kite-shape-analysis`), full module sets
(`bayesian-bootstrap`, `svd-rotate-stretch-rotate`, `explainability-localization`), and
`export_widget_data.py` scripts that write the JSON behind a post's interactive widget.
List them with `ls -d posts/*/src/`.

These are **not** executed by Quarto and are not covered by `_freeze/` or
`make check-posts`. Regenerating an asset means running the script yourself and
committing the result. They may carry their own dependency pin
(`open-source-tts-history/src/requirements-audio.txt`) and their own venv with no
kernel and no post pinning it — `.venv-kokoro` exists solely to run `make_audio.py`,
which is why it looks orphaned next to the kernel-backed venvs.

Covers are not drawn here: `scripts/make_cover.py` writes `cover.png` from the map in
`scripts/cover_sources.yml`.
