#!/usr/bin/env python3
"""Capture a bounded causal MoE trace from an already-running H3 trace server."""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from analysis import process_moe_trace
from scripts import bench_openai, trace_capture


DEFAULT_PROMPT = (
    "Explain why exact routed-expert storage prefetch in a sparse MoE model is "
    "causally constrained by the router output, using a concrete systems example."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8018/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if args.requests < 1 or args.max_tokens < 2:
        raise SystemExit("--requests must be >=1 and --max-tokens must be >=2")

    trace_source = args.trace_dir.resolve() / "moe_runtime.jsonl"
    out = args.out_dir.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite causal trace directory: {out}")
    out.mkdir(parents=True)

    start = trace_capture.file_offset(trace_source)
    clients: list[dict] = []
    wall_start = time.perf_counter_ns()
    for ordinal in range(args.requests):
        request_id = f"fenix-h3-causal-{ordinal:04d}-{uuid.uuid4().hex[:8]}"
        result = bench_openai.run_one(
            args.url,
            args.model,
            args.prompt,
            args.max_tokens,
            args.temperature,
            request_id,
        )
        result.update(ordinal=ordinal, concurrency=1)
        clients.append(result)
    wall_end = time.perf_counter_ns()

    # Give trace writes already issued by the completed request a scheduling
    # boundary before taking the file size. This is not included in any causal
    # timing field; those timestamps were recorded before trace emission.
    time.sleep(0.05)
    end = trace_capture.file_offset(trace_source)

    client_path = out / "client.jsonl"
    trace_capture.atomic_write_text(
        client_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in clients),
    )
    events = trace_capture.capture_jsonl_window(
        trace_source,
        start,
        end,
        out / "moe_runtime.jsonl",
    )
    normalized, summary = process_moe_trace.normalize(
        out / "moe_runtime.jsonl",
        client_path,
    )
    trace_capture.atomic_write_text(
        out / "moe_normalized.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized),
    )

    result = {
        "schema_version": 1,
        "artifact_kind": "fenix_h3_causal_trace_capture",
        "requests": len(clients),
        "max_tokens": args.max_tokens,
        "concurrency": 1,
        "wall_s": (wall_end - wall_start) / 1e9,
        "source_trace": str(trace_source),
        "source_window": {"start": start, "end": end, "records": len(events)},
        "normalization": summary,
        "client_sha256": trace_capture.sha256_file(client_path),
        "moe_runtime_sha256": trace_capture.sha256_file(out / "moe_runtime.jsonl"),
        "moe_normalized_sha256": trace_capture.sha256_file(out / "moe_normalized.jsonl"),
    }
    trace_capture.write_json(out / "capture.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
