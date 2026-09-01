#!/usr/bin/env bash
# Usage:
#   bash scripts/nsys_profile.sh profiles/run01 <command> [args...]
# The command is profiled once; do not use this repetition for headline throughput.
OUT="$1"
shift
if [ -z "$OUT" ] || [ "$#" -eq 0 ]; then
  echo "usage: $0 <output-prefix> <command> [args...]" >&2
  return 2 2>/dev/null || true
fi
mkdir -p "$(dirname "$OUT")"
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output="$OUT" \
  "$@"
