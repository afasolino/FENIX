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

  "$PY" instrumentation/prepare_runtime.py     --runtime external/runtime/qwen38     --output .runtime/instrumented/qwen38
  PREPARE_RC="$?"
  if [ "$PREPARE_RC" -ne 0 ]; then
    echo "Transactional instrumentation preparation failed." >&2
    return "$PREPARE_RC"
  fi

  "$PY" -m scripts.fenix_podman build     -t fenix-qwen38:locked     -f .runtime/instrumented/qwen38/docker/Dockerfile     .runtime/instrumented/qwen38
}

main "$@"
