#!/usr/bin/env bash
# Blocks `git commit` while HEAD is on main/master.
# Wired up as a PreToolUse hook on Bash in .claude/settings.json.
# Policy lives in CLAUDE.md: feature branch -> PR -> review -> merge.
set -uo pipefail

cmd=$(jq -r '.tool_input.command // ""')

# Only intervene on commands that actually reach `git commit`, including
# compound forms like `git add -A && git commit -m "..."` and `git -C dir commit`.
if ! printf '%s' "$cmd" | grep -Eq '(^|[^[:alnum:]_./-])git([[:space:]]+[^;&|[:space:]]+)*[[:space:]]+commit([^[:alnum:]_-]|$)'; then
  exit 0
fi

# A command that switches branch before it commits is exactly the right thing to
# do, so let it through (e.g. `git switch -c rk/foo && git commit -m "..."`).
if printf '%s' "${cmd%%commit*}" | grep -Eq 'git[[:space:]]+(switch|checkout|worktree)\b'; then
  exit 0
fi

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || exit 0
case "$branch" in
  main | master) ;;
  *) exit 0 ;;
esac

jq -n --arg b "$branch" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: ("Refusing to commit directly to \($b) — this repo requires feature branch -> PR -> review -> merge (see CLAUDE.md). Create a branch first (`git switch -c <slug>`), then commit there. The /ship-pr skill does the whole loop.")
  }
}'
