#!/usr/bin/env bash
# Allow/block matrix for block-main-commit.sh. Run directly: .claude/hooks/test-block-main-commit.sh
# Builds a throwaway repo on `main` so the branch check actually engages, then
# again on a feature branch to confirm the hook stays out of the way there.
set -uo pipefail

hook="$(cd "$(dirname "$0")" && pwd)/block-main-commit.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git init -q -b main "$tmp"
git -C "$tmp" commit -q --allow-empty -m init

fails=0

# verdict <command> -> prints BLOCKED or allowed
verdict() {
  local out
  out=$(printf '{"tool_input":{"command":%s}}' "$(jq -Rn --arg c "$1" '$c')" | (cd "$tmp" && "$hook"))
  [ -n "$out" ] && echo BLOCKED || echo allowed
}

check() {
  local want=$1 cmd=$2 got
  got=$(verdict "$cmd")
  if [ "$got" = "$want" ]; then
    printf '  ok   %-52s %s\n' "$cmd" "$got"
  else
    printf '  FAIL %-52s want %s, got %s\n' "$cmd" "$want" "$got"
    fails=$((fails + 1))
  fi
}

echo "on main — commits must be blocked:"
check BLOCKED 'git commit -m x'
check BLOCKED 'git commit --amend --no-edit'
check BLOCKED 'git add -A && git commit -m "x"'
check BLOCKED 'cd docs && git commit -m x'
check BLOCKED 'git -C /some/dir commit --amend'
check BLOCKED 'git --no-pager commit -m x'
# Switching *onto* main is not an escape hatch.
check BLOCKED 'git switch main && git commit -m x'
check BLOCKED 'git checkout main && git commit -m x'
check BLOCKED 'git worktree list && git commit -m x'

echo "on main — creating a branch first is the sanctioned path:"
check allowed 'git switch -c rk/foo && git commit -m x'
check allowed 'git checkout -b rk/bar; git commit -m x'
check allowed 'git worktree add ../wt -b rk/baz && git commit -m x'

echo "on main — non-commit commands must pass untouched:"
check allowed 'git status'
check allowed 'git help commit'
check allowed 'git config --global commit.gpgsign false'
check allowed 'git log --grep commit'
check allowed 'git log --oneline | grep commit'
check allowed 'gh pr merge 12'
check allowed 'quarto render posts/foo/index.qmd'

echo "on a feature branch — the hook must not fire at all:"
git -C "$tmp" checkout -q -b rk/feature
check allowed 'git commit -m x'
check allowed 'git add -A && git commit -m "x"'

if [ "$fails" -eq 0 ]; then
  echo "PASS"
else
  echo "FAIL: $fails case(s)"
  exit 1
fi
