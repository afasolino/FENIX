#!/usr/bin/env python3
"""Construct or execute the pinned FENIX vLLM container launch."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


IMAGE = "fenix-qwen38:locked"
SERVED_MODEL_NAME = "qwen3.8-flash-next"


def parse_gpu_ids(raw: str) -> list[str]:
    gpu_ids = [value.strip() for value in raw.split(",") if value.strip()]
    if not gpu_ids:
        raise argparse.ArgumentTypeError("at least one GPU ID is required")
    return gpu_ids


def build_environment(
    hot_experts: int,
    trace_enabled: bool,
    runtime_directory: Path,
    expandable_segments: bool = True,
) -> dict[str, str]:
    environment = {
        "VLLM_PLE_CPU_OFFLOAD": "1",
        "VLLM_WNA16_DYNAMIC_LRU": "1",
        "VLLM_WNA16_MIXED_VMM_HOT_CACHE": "1",
        "VLLM_WNA16_STATIC_HOT_CACHE_SIZE": str(hot_experts),
        "VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS": "16",
        "FENIX_TRACE": "1" if trace_enabled else "0",
        "FENIX_TRACE_DIR": "/fenix-traces",
        "PYTORCH_CUDA_ALLOC_CONF": (
            "expandable_segments:True"
            if expandable_segments
            else "expandable_segments:False"
        ),
        "VLLM_PLE_OFFLOAD_READY_TIMEOUT": "1200",
    }

    ranking = runtime_directory / "configs/static_hot_cache_rankings.json"
    if ranking.exists():
        environment["VLLM_WNA16_STATIC_HOT_CACHE_FILE"] = (
            "/runtime/configs/static_hot_cache_rankings.json"
        )

    return environment


def build_command(
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
    trace_enabled: bool,
    runtime_image: str = IMAGE,
    expandable_segments: bool = True,
) -> tuple[dict[str, str], list[str]]:
    tensor_parallel_size = len(gpu_ids)
    environment = build_environment(
        hot_experts,
        trace_enabled,
        runtime_directory,
        expandable_segments,
    )
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    command = [
        sys.executable,
        "-m",
        "scripts.fenix_podman",
        "run",
        "--rm",
        "--security-opt=label=disable",
        "--device",
        "nvidia.com/gpu=all",
        "--entrypoint",
        "vllm",
        "--ipc",
        "host",
        "--cap-add",
        "SYS_PTRACE",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "-p",
        f"127.0.0.1:{port}:{port}",
        "-v",
        f"{model_directory}:/model:ro",
        "-v",
        f"{runtime_directory}:/runtime:ro",
        "-v",
        f"{trace_directory}:/fenix-traces",
    ]

    for key, value in environment.items():
        command.extend(["-e", f"{key}={value}"])

    compilation_config = (
        '{"mode":0,"cudagraph_mode":"NONE"}'
        if trace_enabled
        else '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
    )

    serve = [
        "serve",
        "/model",
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--moe-backend",
        "humming",
        "--dtype",
        "bfloat16",
        "--language-model-only",
        "--load-format",
        "safetensors",
        "--safetensors-load-strategy",
        "lazy",
        "--offload-backend",
        "uva",
        "--cpu-offload-gb",
        str(cpu_offload_gib),
        "--cpu-offload-params",
        "experts",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--kv-cache-dtype",
        "auto",
        "--kv-cache-memory-bytes",
        str(kv_cache_memory_bytes),
        "--enable-chunked-prefill",
        "--mamba-cache-mode",
        "align",
        "--compilation-config",
        compilation_config,
        "--no-async-scheduling",
        "--disable-custom-all-reduce",
        "--generation-config",
        "vllm",
        "--reasoning-parser",
        "qwen3",
        "--trust-remote-code",
    ]

    if tensor_parallel_size == 1:
        # The pinned preview image predates the upstream uniproc PLE startup
        # fix. Force the path on which PLE worker spawn/readiness is wired.
        serve.extend(["--distributed-executor-backend", "mp"])
    else:
        serve.extend(
            [
                "--enable-expert-parallel",
                "--all2all-backend",
                "allgather_reducescatter",
            ]
        )

    command.extend([runtime_image, *serve])
    return environment, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gpus", type=parse_gpu_ids, default=["0"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cpu-offload-gb", type=float, required=True)
    parser.add_argument("--hot-experts", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument(
        "--disable-expandable-segments",
        action="store_true",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    model_directory = args.model_dir.resolve()
    runtime_directory = (root / "external/runtime/qwen38").resolve()
    trace_directory = (root / "traces/raw").resolve()

    if not (model_directory / "model.safetensors.index.json").exists():
        print("checkpoint index is missing", file=sys.stderr)
        return 2
    if not (runtime_directory / ".git").is_dir():
        print("pinned runtime checkout is missing", file=sys.stderr)
        return 2

    environment, command = build_command(
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
        trace_enabled=args.trace,
        runtime_image=args.image,
        expandable_segments=not args.disable_expandable_segments,
    )

    print(json.dumps(environment, indent=2))
    print(shlex.join(command))

    if not args.execute:
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
