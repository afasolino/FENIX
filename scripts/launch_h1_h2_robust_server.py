#!/usr/bin/env python3
"""Launch primary H1/H2 robustness traces with automatic prefix caching disabled."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts import launch_vllm

DEFAULT_POLICY_IMAGE = "fenix-qwen38:h1-h2-robust-v1"


def build_policy_command(
    *,
    model_directory: Path,
    runtime_directory: Path,
    trace_directory: Path,
    gpu_ids: list[str],
    port: int,
    cpu_offload_gib: float,
    hot_experts: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    kv_cache_memory_bytes: int,
    runtime_image: str,
    expandable_segments: bool,
) -> tuple[dict[str, str], list[str]]:
    environment, command = launch_vllm.build_command(
        model_directory=model_directory,
        runtime_directory=runtime_directory,
        trace_directory=trace_directory,
        gpu_ids=gpu_ids,
        port=port,
        cpu_offload_gib=cpu_offload_gib,
        hot_experts=hot_experts,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        trace_enabled=True,
        runtime_image=runtime_image,
        expandable_segments=expandable_segments,
        ple_storage_mode="resident",
        ple_bank_manifest=None,
    )
    environment["FENIX_PREFIX_CACHING"] = "0"
    try:
        image_index = command.index(runtime_image)
    except ValueError as exc:
        raise RuntimeError("runtime image token is missing from constructed command") from exc
    command[image_index:image_index] = ["-e", "FENIX_PREFIX_CACHING=0"]
    try:
        serve_index = command.index("serve", image_index + 2)
    except ValueError as exc:
        raise RuntimeError("vLLM serve token is missing from constructed command") from exc

    # vLLM requires the model to be the first positional argument after `serve`.
    if serve_index + 1 >= len(command) or command[serve_index + 1].startswith("--"):
        raise RuntimeError("vLLM model positional argument is missing after serve")
    command.insert(serve_index + 2, "--no-enable-prefix-caching")
    return environment, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=launch_vllm.parse_gpu_ids, default=["0"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cpu-offload-gb", type=float, default=40.0)
    parser.add_argument("--hot-experts", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--image", default=DEFAULT_POLICY_IMAGE)
    parser.add_argument("--disable-expandable-segments", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    model_directory = args.model_dir.resolve()
    runtime_directory = (root / "external/runtime/qwen38").resolve()
    trace_directory = (root / "traces/raw").resolve()

    if not (model_directory / "model.safetensors.index.json").is_file():
        print("checkpoint index is missing")
        return 2
    if not (runtime_directory / ".git").is_dir():
        print("pinned runtime checkout is missing")
        return 2

    environment, command = build_policy_command(
        model_directory=model_directory,
        runtime_directory=runtime_directory,
        trace_directory=trace_directory,
        gpu_ids=args.gpus,
        port=args.port,
        cpu_offload_gib=args.cpu_offload_gb,
        hot_experts=args.hot_experts,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        runtime_image=args.image,
        expandable_segments=not args.disable_expandable_segments,
    )
    launch_vllm.emit_launch_preamble(environment, command)
    if not args.execute:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
