#!/usr/bin/env python3
"""Joint request-level characterization of exact C=1 PLE and MoE traces."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from analysis.expert_locality import parse_layer_id
from analysis.process_ple_trace import load_jsonl


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[max(index, 0)])


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "mean": sum(samples) / len(samples) if samples else None,
        "p50": _quantile(samples, 0.50),
        "p95": _quantile(samples, 0.95),
        "min": min(samples) if samples else None,
        "max": max(samples) if samples else None,
    }


def analyze(
    ple_path: Path,
    moe_path: Path,
    *,
    expected_request_ids: set[str] | None = None,
) -> dict[str, Any]:
    ple_records = load_jsonl(ple_path)
    moe_records = load_jsonl(moe_path)
    if not ple_records:
        raise ValueError("normalized PLE trace is empty")
    if not moe_records:
        raise ValueError("normalized MoE trace is empty")

    ple_by_request: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    moe_by_request: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in ple_records:
        ple_by_request[str(record["request_id"])].append(record)
    for record in moe_records:
        moe_by_request[str(record["request_id"])].append(record)

    ple_ids = set(ple_by_request)
    moe_ids = set(moe_by_request)
    if ple_ids != moe_ids:
        raise ValueError(
            "PLE/MoE request sets differ: "
            f"ple_only={sorted(ple_ids - moe_ids)}, "
            f"moe_only={sorted(moe_ids - ple_ids)}"
        )
    if expected_request_ids is not None and ple_ids != expected_request_ids:
        raise ValueError(
            "normalized trace request set differs from successful clients: "
            f"missing={sorted(expected_request_ids - ple_ids)}, "
            f"unexpected={sorted(ple_ids - expected_request_ids)}"
        )

    row_bytes_values = {
        int(record["bytes"])
        for record in ple_records
        if record.get("bytes") is not None
    }
    if len(row_bytes_values) != 1:
        raise ValueError(
            "normalized PLE trace must expose one uniform row byte width"
        )
    row_bytes = next(iter(row_bytes_values))

    requests: list[dict[str, Any]] = []
    for request_id in sorted(ple_ids):
        ple = ple_by_request[request_id]
        moe = moe_by_request[request_id]
        rows = [int(record["physical_row_id"]) for record in ple]

        expert_keys: list[tuple[int, int]] = []
        transfer_events = 0
        transfer_bytes = 0
        cache_hits = cache_observations = 0
        for record in moe:
            layer = parse_layer_id(record["layer"])
            selected = [int(value) for value in record.get("selected_expert_ids", [])]
            expert_keys.extend((layer, expert) for expert in selected)
            transfer_events += len(record.get("transfer_expert_ids", []))
            transfer_bytes += int(record.get("transfer_bytes", 0))
            hit_values = record.get("cache_hit")
            if hit_values is not None:
                cache_hits += sum(bool(value) for value in hit_values)
                cache_observations += len(hit_values)

        phase = {}
        for phase_name in ("prefill", "decode", "unknown"):
            phase_ple = [record for record in ple if record.get("phase") == phase_name]
            phase_moe = [record for record in moe if record.get("phase") == phase_name]
            if not phase_ple and not phase_moe:
                continue
            phase[phase_name] = {
                "ple_row_accesses": len(phase_ple),
                "moe_events": len(phase_moe),
                "expert_selections": sum(
                    len(record.get("selected_expert_ids", []))
                    for record in phase_moe
                ),
                "expert_transfer_bytes": sum(
                    int(record.get("transfer_bytes", 0)) for record in phase_moe
                ),
            }

        requests.append(
            {
                "request_id": request_id,
                "ple_row_accesses": len(rows),
                "ple_unique_rows": len(set(rows)),
                "ple_unique_working_set_bytes": len(set(rows)) * row_bytes,
                "expert_selections": len(expert_keys),
                "unique_layer_experts": len(set(expert_keys)),
                "expert_transfer_events": transfer_events,
                "expert_transfer_bytes": transfer_bytes,
                "observed_expert_cache_hit_rate": (
                    cache_hits / cache_observations
                    if cache_observations
                    else None
                ),
                "phase": phase,
            }
        )

    return {
        "schema_version": 1,
        "evidence_kind": "local_measured_trace_analysis",
        "performance_eligible": False,
        "can_establish_motivation": False,
        "correlation_mode": "exact_request_correlation",
        "request_count": len(requests),
        "row_bytes": row_bytes,
        "request_metrics": requests,
        "aggregate": {
            "ple_row_accesses": _summary(
                item["ple_row_accesses"] for item in requests
            ),
            "ple_unique_rows": _summary(
                item["ple_unique_rows"] for item in requests
            ),
            "ple_unique_working_set_bytes": _summary(
                item["ple_unique_working_set_bytes"] for item in requests
            ),
            "expert_selections": _summary(
                item["expert_selections"] for item in requests
            ),
            "unique_layer_experts": _summary(
                item["unique_layer_experts"] for item in requests
            ),
            "expert_transfer_bytes": _summary(
                item["expert_transfer_bytes"] for item in requests
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ple", type=Path, required=True)
    parser.add_argument("--moe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.ple, args.moe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
