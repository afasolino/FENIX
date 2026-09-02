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
from typing import Any, Iterable

from analysis.expert_locality import parse_layer_id


ExpertKey = tuple[int, int]


@dataclass(frozen=True)
class Projection:
    host_budget_gib: float
    placement: str
    host_budget_bytes: int
    bytes_available_for_experts: int
    expert_capacity: int
    expert_cache_hit_rate: float
    expert_misses_per_selection: float
    expert_storage_bytes_per_selection: float


@dataclass(frozen=True)
class CapacityInputs:
    expert_bytes: int
    ple_host_bytes: int
    ple_row_bytes: int | None
    ple_addressable_rows: int | None
    expert_bytes_source: str
    ple_host_bytes_source: str


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def load_expert_sequence(trace_path: Path) -> list[ExpertKey]:
    sequence: list[ExpertKey] = []

    for line_number, record in enumerate(_load_jsonl(trace_path), start=1):
        if "layer" not in record:
            raise ValueError(f"{trace_path}:{line_number}: missing layer")

        layer = parse_layer_id(record["layer"])
        for expert in record.get("selected_expert_ids", []):
            sequence.append((layer, int(expert)))

    if not sequence:
        raise ValueError("expert trace contains no routed-expert selections")

    return sequence


def derive_expert_bytes(trace_path: Path) -> int:
    """Derive one expert cache-slot byte size from explicit transfer records."""

    observed: set[int] = set()
    for line_number, record in enumerate(_load_jsonl(trace_path), start=1):
        transferred = record.get("transfer_expert_ids", [])
        if not isinstance(transferred, list):
            raise ValueError(
                f"{trace_path}:{line_number}: transfer_expert_ids must be a list"
            )
        if not transferred:
            continue
        transfer_bytes = record.get("transfer_bytes")
        if not isinstance(transfer_bytes, int) or isinstance(transfer_bytes, bool):
            raise ValueError(
                f"{trace_path}:{line_number}: transfer_bytes must be an integer"
            )
        if transfer_bytes <= 0 or transfer_bytes % len(transferred):
            raise ValueError(
                f"{trace_path}:{line_number}: transfer_bytes is not divisible by "
                "the number of transferred experts"
            )
        observed.add(transfer_bytes // len(transferred))

    if len(observed) != 1:
        raise ValueError(
            "cannot derive one expert byte size from trace; observed="
            f"{sorted(observed)}"
        )
    return next(iter(observed))


def _model_geometry(config_path: Path) -> tuple[int, int, int]:
    payload = json.loads(config_path.read_text())
    try:
        model = payload["model"]
        ngram_size = int(model["ngram_size"])
        heads_per_ngram = int(model["heads_per_ngram"])
        vocab = int(model["ngram_vocab_size_base"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("campaign model PLE geometry is incomplete") from exc
    if ngram_size < 2 or heads_per_ngram < 1 or vocab < 1:
        raise ValueError("campaign model PLE geometry must be positive")
    return ngram_size, heads_per_ngram, vocab


def derive_ple_host_bytes(
    trace_path: Path,
    config_path: Path,
) -> tuple[int, int, int]:
    """Derive addressable PLE bytes from measured row width and model geometry."""

    observed_row_bytes: set[int] = set()
    for line_number, record in enumerate(_load_jsonl(trace_path), start=1):
        raw = record.get("bytes", record.get("row_bytes"))
        if raw is None:
            continue
        if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
            raise ValueError(
                f"{trace_path}:{line_number}: PLE row bytes must be positive"
            )
        observed_row_bytes.add(raw)
    if len(observed_row_bytes) != 1:
        raise ValueError(
            "cannot derive one PLE row byte width from trace; observed="
            f"{sorted(observed_row_bytes)}"
        )

    row_bytes = next(iter(observed_row_bytes))
    ngram_size, heads_per_ngram, vocab = _model_geometry(config_path)
    addressable_rows = (ngram_size - 1) * heads_per_ngram * vocab
    return addressable_rows * row_bytes, row_bytes, addressable_rows


def resolve_capacity_inputs(
    *,
    moe_trace: Path,
    ple_trace: Path | None,
    config_path: Path,
    expert_bytes_override: int | None,
    ple_host_bytes_override: int | None,
) -> CapacityInputs:
    if expert_bytes_override is not None:
        if expert_bytes_override <= 0:
            raise ValueError("expert_bytes override must be positive")
        expert_bytes = expert_bytes_override
        expert_source = "manual_override"
    else:
        expert_bytes = derive_expert_bytes(moe_trace)
        expert_source = "measured_moe_transfer_trace"

    row_bytes = addressable_rows = None
    if ple_host_bytes_override is not None:
        if ple_host_bytes_override < 0:
            raise ValueError("ple_host_bytes override cannot be negative")
        ple_host_bytes = ple_host_bytes_override
        ple_source = "manual_override"
    else:
        if ple_trace is None:
            raise ValueError(
                "--ple-trace is required unless --ple-host-bytes is provided"
            )
        ple_host_bytes, row_bytes, addressable_rows = derive_ple_host_bytes(
            ple_trace,
            config_path,
        )
        ple_source = "measured_row_width_plus_versioned_geometry"

    return CapacityInputs(
        expert_bytes=expert_bytes,
        ple_host_bytes=ple_host_bytes,
        ple_row_bytes=row_bytes,
        ple_addressable_rows=addressable_rows,
        expert_bytes_source=expert_source,
        ple_host_bytes_source=ple_source,
    )


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
        host_budget_bytes=host_budget_bytes,
        bytes_available_for_experts=bytes_available_for_experts,
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


def paired_deltas(rows: list[Projection]) -> list[dict[str, Any]]:
    by_budget: dict[float, dict[str, Projection]] = collections.defaultdict(dict)
    for row in rows:
        by_budget[row.host_budget_gib][row.placement] = row

    output = []
    for budget in sorted(by_budget):
        pair = by_budget[budget]
        baseline = pair.get("ple_in_host_dram")
        externalized = pair.get("ple_externalized")
        if baseline is None or externalized is None:
            raise ValueError(f"incomplete placement pair for budget {budget}")
        baseline_bytes = baseline.expert_storage_bytes_per_selection
        reduction = (
            (baseline_bytes - externalized.expert_storage_bytes_per_selection)
            / baseline_bytes
            if baseline_bytes > 0
            else 0.0
        )
        output.append(
            {
                "host_budget_gib": budget,
                "baseline_expert_capacity": baseline.expert_capacity,
                "externalized_expert_capacity": externalized.expert_capacity,
                "additional_expert_capacity": (
                    externalized.expert_capacity - baseline.expert_capacity
                ),
                "expert_storage_bytes_reduction_fraction": reduction,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moe-trace", type=Path, required=True)
    parser.add_argument("--ple-trace", type=Path)
    parser.add_argument("--expert-bytes", type=int)
    parser.add_argument("--ple-host-bytes", type=int)
    parser.add_argument("--config", type=Path, default=Path("configs/campaign.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        inputs = resolve_capacity_inputs(
            moe_trace=args.moe_trace,
            ple_trace=args.ple_trace,
            config_path=args.config,
            expert_bytes_override=args.expert_bytes,
            ple_host_bytes_override=args.ple_host_bytes,
        )
        sequence = load_expert_sequence(args.moe_trace)
        projections = [
            project_budget(
                sequence,
                budget,
                inputs.expert_bytes,
                inputs.ple_host_bytes,
                placement,
            )
            for budget in load_budgets(args.config)
            for placement in ("ple_in_host_dram", "ple_externalized")
        ]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    result = {
        "schema_version": 2,
        "evidence_kind": "trace_projection",
        "can_establish_motivation": False,
        "budget_scope": "ple_plus_expert_managed_capacity",
        "inputs": asdict(inputs),
        # Keep legacy top-level names for downstream compatibility.
        "expert_bytes": inputs.expert_bytes,
        "ple_host_bytes": inputs.ple_host_bytes,
        "rows": [asdict(row) for row in projections],
        "budget_deltas": paired_deltas(projections),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
