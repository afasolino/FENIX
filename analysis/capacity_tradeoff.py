#!/usr/bin/env python3
"""Screen host-memory budgets using a trace-driven expert-residency model.

This model is deliberately non-promotional: its output is tagged
``trace_projection`` and cannot establish the FENIX motivation claim.
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ExpertKey = tuple[int, int]


@dataclass(frozen=True)
class Projection:
    host_budget_gib: float
    placement: str
    expert_capacity: int
    expert_cache_hit_rate: float
    expert_misses_per_selection: float
    expert_storage_bytes_per_selection: float


def load_expert_sequence(trace_path: Path) -> list[ExpertKey]:
    sequence: list[ExpertKey] = []

    for line_number, line in enumerate(trace_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "layer" not in record:
            raise ValueError(f"{trace_path}:{line_number}: missing layer")

        layer = int(record["layer"])
        for expert in record.get("selected_expert_ids", []):
            sequence.append((layer, int(expert)))

    if not sequence:
        raise ValueError("expert trace contains no routed-expert selections")

    return sequence


def simulate_lru(sequence: Iterable[ExpertKey], capacity: int) -> tuple[int, int, int]:
    cache: collections.OrderedDict[ExpertKey, None] = collections.OrderedDict()
    selections = hits = misses = 0

    for key in sequence:
        selections += 1
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            continue

        misses += 1
        if capacity > 0:
            cache[key] = None
            if len(cache) > capacity:
                cache.popitem(last=False)

    return selections, hits, misses


def project_budget(
    sequence: list[ExpertKey],
    host_budget_gib: float,
    expert_bytes: int,
    ple_host_bytes: int,
    placement: str,
) -> Projection:
    host_budget_bytes = int(host_budget_gib * 1024**3)

    if placement == "ple_in_host_dram":
        bytes_available_for_experts = max(0, host_budget_bytes - ple_host_bytes)
    elif placement == "ple_externalized":
        bytes_available_for_experts = host_budget_bytes
    else:
        raise ValueError(f"unsupported placement: {placement}")

    expert_capacity = bytes_available_for_experts // expert_bytes
    selections, hits, misses = simulate_lru(sequence, expert_capacity)

    return Projection(
        host_budget_gib=host_budget_gib,
        placement=placement,
        expert_capacity=expert_capacity,
        expert_cache_hit_rate=hits / selections,
        expert_misses_per_selection=misses / selections,
        expert_storage_bytes_per_selection=(misses * expert_bytes) / selections,
    )


def load_budgets(config_path: Path) -> list[float]:
    config = json.loads(config_path.read_text())
    raw = config["experiments"]["capacity_tradeoff"]["host_memory_budgets_gib"]
    budgets = [float(value) for value in raw]
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("capacity-tradeoff budgets must be positive")
    return budgets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moe-trace", type=Path, required=True)
    parser.add_argument("--expert-bytes", type=int, required=True)
    parser.add_argument("--ple-host-bytes", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/campaign.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.expert_bytes <= 0:
        raise SystemExit("--expert-bytes must be positive")
    if args.ple_host_bytes < 0:
        raise SystemExit("--ple-host-bytes cannot be negative")

    sequence = load_expert_sequence(args.moe_trace)
    rows = []
    for budget in load_budgets(args.config):
        for placement in ("ple_in_host_dram", "ple_externalized"):
            rows.append(
                asdict(
                    project_budget(
                        sequence,
                        budget,
                        args.expert_bytes,
                        args.ple_host_bytes,
                        placement,
                    )
                )
            )

    result = {
        "schema_version": 1,
        "evidence_kind": "trace_projection",
        "can_establish_motivation": False,
        "expert_bytes": args.expert_bytes,
        "ple_host_bytes": args.ple_host_bytes,
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
