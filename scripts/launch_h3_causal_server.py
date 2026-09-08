#!/usr/bin/env python3
"""Launch the disposable H3 causal-prefetch trace server.

The model and pinned runtime checkout may live in the frozen FENIX checkout;
the instrumented code is baked into the disposable image.  The runtime mount is
retained read-only for the same ranking/config contract used by normal FENIX
launches. Prefix caching is disabled to preserve the H1/H2 trace semantics.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts import launch_vllm


DEFAULT_IMAGE = "fenix-qwen38:h3-causal-prefetch-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=launch_vllm.parse_gpu_ids, default=["0"])
    parser.add_argument("--port", type=int, default=8018)
    parser.add_argument("--cpu-offload-gb", type=float, default=40.0)
    parser.add_argument("--hot-experts", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    model = args.model_dir.resolve()
    runtime = args.runtime_dir.resolve()
    traces = args.trace_dir.resolve()
    if not (model / "model.safetensors.index.json").is_file():
        print(f"checkpoint index is missing: {model}")
        return 2
    if not (runtime / ".git").exists():
        print(f"pinned runtime checkout is missing: {runtime}")
        return 2
    traces.mkdir(parents=True, exist_ok=True)

    environment, command = launch_vllm.build_command(
        model_directory=model,
        runtime_directory=runtime,
        trace_directory=traces,
        gpu_ids=args.gpus,
        port=args.port,
        cpu_offload_gib=args.cpu_offload_gb,
        hot_experts=args.hot_experts,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        trace_enabled=True,
        runtime_image=args.image,
        expandable_segments=True,
        ple_storage_mode="resident",
        ple_bank_manifest=None,
    )
    environment["FENIX_PREFIX_CACHING"] = "0"
    image_index = command.index(args.image)
    command[image_index:image_index] = ["-e", "FENIX_PREFIX_CACHING=0"]
    serve_index = command.index("serve", image_index + 2)
    command.insert(serve_index + 2, "--no-enable-prefix-caching")

    launch_vllm.emit_launch_preamble(environment, command)
    if not args.execute:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
