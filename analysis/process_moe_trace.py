#!/usr/bin/env python3
"""Normalize concurrency-one MoE runtime events into request-correlated records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analysis.process_ple_trace import correlate_request, load_jsonl


def classify_phase(timestamp_ns: int, client_record: dict[str, Any]) -> str:
    first_token_ns = client_record.get("first_token_ns")
    if first_token_ns is None:
        return "unknown"
    return "decode" if timestamp_ns >= int(first_token_ns) else "prefill"


def normalize(
    runtime_path: Path,
    client_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    runtime_events = load_jsonl(runtime_path)
    clients = [
        record for record in load_jsonl(client_path) if "error" not in record
    ]
    if not runtime_events:
        raise ValueError("runtime trace contains no MoE events")
    if not clients:
        raise ValueError("client trace contains no successful requests")
    if any(int(record.get("concurrency", 1)) != 1 for record in clients):
        raise ValueError("exact MoE correlation currently requires concurrency=1")

    output: list[dict[str, Any]] = []
    for event in runtime_events:
        if "timestamp_ns" not in event:
            raise ValueError("MoE runtime event is missing timestamp_ns")
        timestamp_ns = int(event["timestamp_ns"])
        request = correlate_request(timestamp_ns, clients)
        selected = [int(value) for value in event.get("selected_expert_ids", [])]
        if not selected:
            raise ValueError("MoE runtime event contains no selected experts")

        cache_hit = event.get("cache_hit")
        if cache_hit is not None and len(cache_hit) != len(selected):
            raise ValueError("cache_hit length does not match selected_expert_ids")

        item = dict(event)
        item.update(
            request_id=str(request["request_id"]),
            phase=classify_phase(timestamp_ns, request),
            concurrency=1,
            selected_expert_ids=selected,
        )
        output.append(item)

    return output, {
        "runtime_events": len(runtime_events),
        "client_requests": len(clients),
        "correlated_events": len(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records, summary = normalize(args.runtime, args.client)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
