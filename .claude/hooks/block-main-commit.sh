#!/usr/bin/env bash
# Blocks `git commit` while HEAD is on main/master.
# Wired up as a PreToolUse hook on Bash in .claude/settings.json.
# Policy lives in AGENTS.md: feature branch -> PR -> review -> merge.
set -uo pipefail

# Fail open rather than breaking every Bash call, but say so — a policy hook
# that silently stops enforcing is worse than one that is obviously off.
if ! command -v jq >/dev/null 2>&1; then
  echo "block-main-commit: jq not found; branch policy NOT enforced" >&2
  exit 0
fi

cmd=$(jq -r '.tool_input.command // ""')

# Only intervene when `commit` is the actual git subcommand — i.e. the first
# token after `git` and any global options (`-C <path>`, `-c <cfg>`, `--no-pager`).
# Anchoring this way still catches compound forms (`git add -A && git commit …`)
# without snagging commands that merely mention the word (`git help commit`).
commit_re='(^|[^[:alnum:]_./-])git([[:space:]]+(-[cC][[:space:]]+[^;&|[:space:]]+|--[^;&|[:space:]]+|-[^-;&|[:space:]]+))*[[:space:]]+commit([^[:alnum:]_-]|$)'
if ! printf '%s' "$cmd" | grep -Eq "$commit_re"; then
  exit 0
fi

# A command that *creates* a branch before committing is exactly the right thing
# to do, so let it through (e.g. `git switch -c rk/foo && git commit -m "..."`).
# Deliberately narrow: a bare `git switch main && git commit` must still be
# blocked, so plain switches/checkouts do not count as an escape hatch.
create_re='git[[:space:]]+(switch[[:space:]]+-[cC]|checkout[[:space:]]+-[bB]|worktree[[:space:]]+add)([^[:alnum:]_-]|$)'
if printf '%s' "${cmd%%commit*}" | grep -Eq "$create_re"; then
  exit 0
fi

# Judge the branch where the command will actually run, not where the hook does.
# Every worktree has its own HEAD, and this repo keeps several
# (`../ml-blog-<topic>`), so `cd ../ml-blog-foo && git commit` on a feature
# branch would otherwise be denied for a `main` that belongs to a different
# checkout. Only a leading `cd` and `git -C <path>` redirect the target; a path
# that does not resolve to a work tree falls back to the hook's own cwd, so a
# bogus or unreadable directory is still judged against main rather than waved
# through.
#
# Both patterns read only the command's first line: a multi-line command (a
# commit message spanning lines, say) would otherwise glue every later line
# onto the path and resolve nothing.
target=""
if printf '%s' "$cmd" | head -n 1 | grep -Eq '^[[:space:]]*cd[[:space:]]'; then
  target=$(printf '%s' "$cmd" | head -n 1 |
    sed -E 's/^[[:space:]]*cd[[:space:]]+//; s/[[:space:]]*(&&|\|\||;).*$//' |
    tr -d '\042\047')
elif printf '%s' "${cmd%%commit*}" | head -n 1 | grep -Eq 'git[[:space:]]+-C[[:space:]]'; then
  target=$(printf '%s' "${cmd%%commit*}" | head -n 1 |
    sed -E 's/.*git[[:space:]]+-C[[:space:]]+//; s/[[:space:]].*$//' |
    tr -d '\042\047')
fi

branch=""
if [ -n "$target" ] && [ -d "$target" ]; then
  branch=$(git -C "$target" rev-parse --abbrev-ref HEAD 2>/dev/null) || branch=""
fi
if [ -z "$branch" ]; then
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || exit 0
fi

case "$branch" in
  main | master) ;;
  *) exit 0 ;;
esac

jq -n --arg b "$branch" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: ("Refusing to commit directly to \($b) — this repo requires feature branch -> PR -> review -> merge (see AGENTS.md). Create a branch first (`git switch -c <slug>`), then commit there. The /ship-pr skill does the whole loop.")
  }
}'
