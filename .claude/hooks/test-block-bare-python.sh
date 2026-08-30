#!/usr/bin/env bash
# Allow/block matrix for block-bare-python.sh. Run directly: .claude/hooks/test-block-bare-python.sh
# The interesting half is the allow list: a hook that also blocks `uv run python`
# or `.venv/bin/python` is one people turn off, which enforces nothing.
set -uo pipefail

hook="$(cd "$(dirname "$0")" && pwd)/block-bare-python.sh"
fails=0

# verdict <command> -> prints BLOCKED or allowed
verdict() {
  local out
  out=$(printf '{"tool_input":{"command":%s}}' "$(jq -Rn --arg c "$1" '$c')" | "$hook")
  [ -n "$out" ] && echo BLOCKED || echo allowed
}

check() {
  local want=$1 cmd=$2 got
  got=$(verdict "$cmd")
  if [ "$got" = "$want" ]; then
    printf '  ok   %-62s %s\n' "$cmd" "$got"
  else
    printf '  FAIL %-62s want %s, got %s\n' "$cmd" "$want" "$got"
    fails=$((fails + 1))
  fi
}

echo "bare interpreter in command position — blocked:"
check BLOCKED 'python3 scripts/check_posts.py'
check BLOCKED 'python scripts/check_posts.py'
check BLOCKED 'python3.12 -c "import sys"'
check BLOCKED 'pip install pillow'
check BLOCKED 'pip3 install --user pyyaml'
check BLOCKED 'python3 -m http.server 8000 --directory docs'
check BLOCKED 'nohup python3 -m http.server 8123 &'
check BLOCKED 'cd docs && python3 -m http.server'
check BLOCKED 'make check-posts; python3 -c "print(1)"'
check BLOCKED 'cat x.json | python3 -m json.tool'
check BLOCKED 'env PYTHONPATH=. python3 run.py'
check BLOCKED 'PYTHONPATH=. python3 run.py'
check BLOCKED 'ls *.py | xargs python3 lint.py'
check BLOCKED 'echo `python3 -c "print(1)"`'
check BLOCKED 'find . -name "*.py" -exec python3 {} \;'
check BLOCKED 'xargs -n1 python3 lint.py'
check BLOCKED 'sudo -H python3 setup.py install'
check BLOCKED 'command python3 x.py'
check BLOCKED 'bash -c "python3 -c 1"'
check BLOCKED "sh -c 'python3 foo.py'"

echo
echo "an interpreter that is actually named — allowed:"
check allowed '.venv/bin/python scripts/check_posts.py'
check allowed './.venv/bin/python -m http.server 8123 --directory docs'
check allowed '/Users/me/repo/.venv-slug/bin/python -m ipykernel install --user --name k'
check allowed 'uv run python scripts/make_cover.py --all'
check allowed 'uv run --with pillow --with pyyaml python scripts/make_cover.py posts/x'
check allowed 'uv pip install --python .venv-slug/bin/python ipykernel'
check allowed 'uv venv .venv-slug --python 3.12'
check allowed 'QUARTO_PYTHON="$(pwd)/.venv/bin/python" quarto render .'

echo
echo "merely naming an interpreter — allowed:"
check allowed 'which -a python3 python'
check allowed 'command -v python3'
check allowed 'grep -rn "python3" CLAUDE.md'
check allowed 'pgrep -fl "http.server"'
check allowed 'git commit -m "stop using python3 directly"'
check allowed 'echo "run python3 yourself"'
check allowed 'ls .venv/bin/python'
check allowed 'grep -c python3 CLAUDE.md'
check allowed '# cd docs && python -m http.server is what we avoid'

echo
if [ "$fails" -eq 0 ]; then
  echo "test-block-bare-python: OK"
else
  echo "test-block-bare-python: $fails failure(s)"
fi
exit $((fails > 0))
