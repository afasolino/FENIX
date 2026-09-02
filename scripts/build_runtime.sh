#!/usr/bin/env bash

main() {
  ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
  if [ -z "$ROOT" ]; then
    echo "Not inside Git." >&2
    return 2
  fi
  cd "$ROOT" || return 2

  PY="$ROOT/.venv/bin/python"
  if [ ! -x "$PY" ]; then
    echo "Expected FENIX interpreter is missing: $PY" >&2
    return 2
  fi

  "$PY" -m scripts.build_runtime "$@"
}

main "$@"
