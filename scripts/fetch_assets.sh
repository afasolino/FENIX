#!/usr/bin/env bash
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$ROOT" ]; then echo "Not inside Git." >&2; return 2 2>/dev/null || true; fi
cd "$ROOT" || return 2 2>/dev/null || true
mkdir -p external/runtime external/models
RUNTIME_DIR="external/runtime/qwen38"
RUNTIME_REV="7b5f0465db90fc49d6324904f48ad995ebdcb62f"
if [ ! -d "$RUNTIME_DIR/.git" ]; then
  git clone https://github.com/DominikBucko/qwen38-flash-next-2x3090.git "$RUNTIME_DIR"
fi
git -C "$RUNTIME_DIR" fetch origin
git -C "$RUNTIME_DIR" checkout --detach "$RUNTIME_REV"
ACTUAL="$(git -C "$RUNTIME_DIR" rev-parse HEAD 2>/dev/null)"
if [ "$ACTUAL" != "$RUNTIME_REV" ]; then echo "Runtime revision mismatch: $ACTUAL" >&2; return 3 2>/dev/null || true; fi
MODEL_REPO="albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE"
MODEL_REV="ef554143369a706525336f6b42a09094835dc077"
MODEL_DIR="external/models/qwen38-flash-next-w4a16-fp8ple"
if command -v hf >/dev/null 2>&1; then
  mkdir -p "$MODEL_DIR"
  hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
else
  echo "Install huggingface_hub CLI ('hf') and rerun to download the pinned checkpoint."
fi
