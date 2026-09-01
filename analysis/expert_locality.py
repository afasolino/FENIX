#!/usr/bin/env python3
"""Analyze routed-expert locality and observed runtime residency behavior."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Hashable, Sequence

from analysis.ple_locality import exact_reuse_distances, logarithmic_histogram


def quantile(values: Sequence[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[max(index, 0)]


def analyze_trace(trace_path: Path) -> dict[str, object]:
    expert_sequence: list[tuple[int, int]] = []
    frequencies: collections.Counter[tuple[int, int]] = collections.Counter()

    runtime_cache_hits = 0
    runtime_cache_observations = 0
    transfer_events = 0
    transfer_bytes = 0
    resident_counts: list[int] = []

    for line_number, line in enumerate(trace_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue

        record = json.loads(line)
        if "layer" not in record:
            raise ValueError(f"{trace_path}:{line_number}: missing layer")

        layer = int(record["layer"])
        selected = [int(expert) for expert in record.get("selected_expert_ids", [])]
        keys = [(layer, expert) for expert in selected]

        expert_sequence.extend(keys)
        frequencies.update(keys)

        cache_hits = record.get("cache_hit")
        if cache_hits is not None:
            if len(cache_hits) != len(selected):
                raise ValueError(
                    f"{trace_path}:{line_number}: cache_hit length does not match "
                    "selected_expert_ids"
                )
            runtime_cache_hits += sum(bool(value) for value in cache_hits)
            runtime_cache_observations += len(cache_hits)

        transferred = record.get("transfer_expert_ids", [])
        transfer_events += len(transferred)
        transfer_bytes += int(record.get("transfer_bytes", 0))

        if "resident_expert_ids" in record:
            resident_counts.append(len(record["resident_expert_ids"]))

    if not expert_sequence:
        raise ValueError("expert trace contains no routed-expert selections")

    distances, cold_accesses = exact_reuse_distances(expert_sequence)

    return {
        "schema_version": 1,
        "evidence_kind": "local_measured_trace_analysis",
        "expert_selections": len(expert_sequence),
        "unique_layer_experts": len(frequencies),
        "reuse_distance": {
            "definition": "unique (layer, expert) keys selected since the previous selection of the same key",
            "reused_selections": len(distances),
            "cold_selections": cold_accesses,
            "p50_layer_experts": quantile(distances, 0.50),
            "p95_layer_experts": quantile(distances, 0.95),
            "p99_layer_experts": quantile(distances, 0.99),
            "max_layer_experts": max(distances) if distances else None,
            "histogram": logarithmic_histogram(distances),
        },
        "observed_runtime": {
            "cache_hit_rate": (
                runtime_cache_hits / runtime_cache_observations
                if runtime_cache_observations
                else None
            ),
            "cache_observations": runtime_cache_observations,
            "expert_transfer_events": transfer_events,
            "expert_transfer_bytes": transfer_bytes,
            "resident_experts_mean": (
                sum(resident_counts) / len(resident_counts)
                if resident_counts
                else None
            ),
        },
        "top_layer_experts": [
            {"layer": layer, "expert": expert, "selections": count}
            for (layer, expert), count in frequencies.most_common(200)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = analyze_trace(args.trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
