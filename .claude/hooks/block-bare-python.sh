#!/usr/bin/env bash
# Blocks a bare `python` / `python3` / `pip` in command position.
# Wired up as a PreToolUse hook on Bash in .claude/settings.json.
# Policy lives in AGENTS.md: name the interpreter, never inherit one.
#
# A bare `python3` on this machine is Homebrew's, not this repo's. It is wrong
# even for a throwaway one-liner or a stdlib-only script, because "it does not
# import anything" is exactly the reasoning that ends with `pip install` into an
# externally-managed environment, or an `http.server` running for hours on the
# wrong interpreter.
set -uo pipefail

# Fail open rather than breaking every Bash call, but say so — a policy hook
# that silently stops enforcing is worse than one that is obviously off.
if ! command -v jq >/dev/null 2>&1; then
  echo "block-bare-python: jq not found; interpreter policy NOT enforced" >&2
  exit 0
fi

cmd=$(jq -r '.tool_input.command // ""')

# Drop whole-line shell comments before matching. Without this the hook denies
# AGENTS.md's own documented preview-server cleanup, whose comment line quotes
# `cd docs && python -m http.server` as the thing it is working around.
cmd=$(printf '%s\n' "$cmd" | grep -v '^[[:space:]]*#' || true)

# Only intervene when the interpreter is in *command position*: at the start of
# the command, or right after a shell operator (`;`, `&&`, `||`, `|`, `(`, `{`, a
# backtick), or behind `xargs` / `find -exec`,
# optionally behind a wrapper like `env` / `nohup` / `time` / `exec` and any
# leading VAR=value assignments.
#
# Anchoring on the operator rather than on whitespace is what keeps the hook
# quiet for the many commands that merely *name* an interpreter:
#   which -a python3        the name is an argument, nothing runs
#   uv run python -c ...    uv resolves the project interpreter, which is the
#                           documented way to run one
#   .venv/bin/python -m x   an explicit path — preceded by `/`, not an operator
#   QUARTO_PYTHON="$(pwd)/.venv/bin/python" quarto render .
#
# Like block-main-commit.sh, this matches the raw command text, so a heredoc
# whose body has a line starting with `python3` is denied too. Write such files
# with the Write tool.
interp='(python|python3|python3\.[0-9]+|pip|pip3)'
# Wrappers may carry their own flags: `xargs -n1 python3`, `sudo -H python3`.
# `command` takes no flags here on purpose: `command -v python3` is a lookup,
# not a run, and must stay allowed.
wrapper='((((env|nohup|time|exec|sudo|xargs)([[:space:]]+-[^[:space:]]+)*)|command)[[:space:]]+)*'
bare_re='(^|[;&|(){}`]|&&|\|\|)[[:space:]]*'"$wrapper"'([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+)*'"$interp"'([[:space:]]|$)'

# `find ... -exec python3 {} \;` has no operator in front of the interpreter, so
# it needs its own anchor rather than a wrapper entry in the pattern above.
exec_re='-exec[[:space:]]+'"$interp"'([[:space:]]|$)'

# `bash -c "python3 …"` hides the interpreter inside a quoted argument. Anchor
# on a shell name so this does not swallow `grep -c python3 file`.
shell_re='(bash|sh|zsh|dash|ksh)[[:space:]]+([^[:space:]]+[[:space:]]+)*-c[[:space:]]+.?'"$interp"'([[:space:]]|$)'

matched=0
for re in "$bare_re" "$exec_re" "$shell_re"; do
  if printf '%s' "$cmd" | grep -Eq "$re"; then
    matched=1
    break
  fi
done
if [ "$matched" -eq 0 ]; then
  exit 0
fi

jq -n '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: "Refusing to run a bare `python`/`python3`/`pip` — that is Homebrew'"'"'s interpreter, not this repo'"'"'s (see AGENTS.md). Name one explicitly: `.venv/bin/python`, `.venv-<slug>/bin/python`, or `uv run python`. Install with `uv pip install --python .venv-<slug>/bin/python`, never bare `pip`. This holds for one-liners, stdlib-only scripts like scripts/check_posts.py, and `-m http.server` too."
  }
}'
