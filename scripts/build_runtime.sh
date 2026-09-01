#!/usr/bin/env bash
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$ROOT" ]; then echo "Not inside Git." >&2; return 2 2>/dev/null || true; fi
cd "$ROOT" || return 2 2>/dev/null || true

MANIFEST="external/runtime/qwen38/fenix-instrumentation-manifest.json"
if [ ! -f "$MANIFEST" ]; then
  python3 instrumentation/prepare_runtime.py --runtime external/runtime/qwen38
  if [ "$?" -ne 0 ]; then
    echo "Instrumentation preparation failed." >&2
    return 3 2>/dev/null || true
  fi
else
  echo "Using existing pinned FENIX instrumentation manifest: $MANIFEST"
fi

docker build -t fenix-qwen38:locked \
  -f external/runtime/qwen38/docker/Dockerfile \
  external/runtime/qwen38
