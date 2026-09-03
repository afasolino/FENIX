#!/usr/bin/env python3
"""Derive deterministic per-layer host-cache expert rankings from MoE traces."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from analysis.capacity_tradeoff import load_expert_geometry
from analysis.expert_locality import parse_layer_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_counts(
    trace_paths: list[Path],
    *,
    num_hidden_layers: int,
    num_experts: int,
) -> tuple[list[collections.Counter[int]], int]:
    counts = [collections.Counter() for _ in range(num_hidden_layers)]
    selections = 0

    for trace_path in trace_paths:
        for line_number, line in enumerate(trace_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{trace_path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{trace_path}:{line_number}: expected object")
            if "layer" not in record:
                raise ValueError(f"{trace_path}:{line_number}: missing layer")
            selected = record.get("selected_expert_ids", [])
            if not isinstance(selected, list):
                raise ValueError(
                    f"{trace_path}:{line_number}: selected_expert_ids must be a list"
                )
            layer = parse_layer_id(record["layer"])
            if not 0 <= layer < num_hidden_layers:
                raise ValueError(
                    f"{trace_path}:{line_number}: layer {layer} outside model geometry"
                )
            for raw_expert in selected:
                expert = int(raw_expert)
                if not 0 <= expert < num_experts:
                    raise ValueError(
                        f"{trace_path}:{line_number}: expert {expert} outside model geometry"
                    )
                counts[layer][expert] += 1
                selections += 1

    if selections == 0:
        raise ValueError("MoE traces contain no expert selections")
    return counts, selections


def rank_layer(counter: collections.Counter[int], num_experts: int) -> list[int]:
    """Rank observed experts by frequency, then append unseen IDs ascending."""

    observed = sorted(counter, key=lambda expert: (-counter[expert], expert))
    unseen = [expert for expert in range(num_experts) if expert not in counter]
    ranking = observed + unseen
    if len(ranking) != num_experts or len(set(ranking)) != num_experts:
        raise AssertionError("expert ranking is not a complete permutation")
    return ranking


def build_ranking_document(
    trace_paths: list[Path],
    config_path: Path,
) -> dict[str, Any]:
    if not trace_paths:
        raise ValueError("at least one MoE trace is required")
    missing = [str(path) for path in trace_paths if not path.is_file()]
    if missing:
        raise ValueError("MoE trace does not exist: " + ", ".join(missing))

    num_hidden_layers, num_experts = load_expert_geometry(config_path)
    counts, selections = load_counts(
        trace_paths,
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
    )

    layers = {
        str(layer): {
            "ranking": rank_layer(counts[layer], num_experts),
            "observed_unique_experts": len(counts[layer]),
            "selections": sum(counts[layer].values()),
        }
        for layer in range(num_hidden_layers)
    }
    return {
        "schema_version": 1,
        "artifact_kind": "trace_derived_expert_host_ranking",
        "can_establish_motivation": False,
        "ranking_policy": "descending_selection_count_then_expert_id",
        "unobserved_policy": "append_ascending_expert_id",
        "num_hidden_layers": num_hidden_layers,
        "num_experts": num_experts,
        "total_selections": selections,
        "source_traces": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in trace_paths
        ],
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moe-trace", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/campaign.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        document = build_ranking_document(args.moe_trace, args.config)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
