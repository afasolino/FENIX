#!/usr/bin/env python3
"""Analyze locality in an exact PLE row-access trace."""

from __future__ import annotations

import argparse
import collections
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable, Sequence


class FenwickTree:
    """Fenwick tree used to compute exact stack reuse distance in O(n log n)."""

    def __init__(self, size: int) -> None:
        self._tree = [0] * (size + 1)

    def add(self, index: int, delta: int) -> None:
        while index < len(self._tree):
            self._tree[index] += delta
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        total = 0
        while index > 0:
            total += self._tree[index]
            index -= index & -index
        return total


def exact_reuse_distances(sequence: Sequence[Hashable]) -> tuple[list[int], int]:
    """Return LRU stack distances and the number of first-touch accesses."""

    tree = FenwickTree(len(sequence))
    last_position: dict[Hashable, int] = {}
    distances: list[int] = []
    cold_accesses = 0

    for position, key in enumerate(sequence, start=1):
        previous = last_position.get(key)
        if previous is None:
            cold_accesses += 1
        else:
            distinct_after_previous = (
                tree.prefix_sum(position - 1) - tree.prefix_sum(previous)
            )
            distances.append(distinct_after_previous)
            tree.add(previous, -1)

        tree.add(position, 1)
        last_position[key] = position

    return distances, cold_accesses


def quantile(values: Sequence[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[max(index, 0)]


def lru_hit_count(sequence: Sequence[Hashable], capacity: int) -> int:
    if capacity <= 0:
        return 0

    cache: collections.OrderedDict[Hashable, None] = collections.OrderedDict()
    hits = 0

    for key in sequence:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            continue

        cache[key] = None
        if len(cache) > capacity:
            cache.popitem(last=False)

    return hits


def logarithmic_histogram(values: Iterable[int]) -> list[dict[str, int]]:
    buckets: collections.Counter[int] = collections.Counter()
    for value in values:
        bucket = 0 if value == 0 else int(math.floor(math.log2(value))) + 1
        buckets[bucket] += 1

    result = []
    for bucket, count in sorted(buckets.items()):
        if bucket == 0:
            lower = upper = 0
        else:
            lower = 1 << (bucket - 1)
            upper = (1 << bucket) - 1
        result.append({"min": lower, "max": upper, "count": count})
    return result


def analyze_trace(
    trace_path: Path,
    capacities_gib: Sequence[float],
    explicit_row_bytes: int | None = None,
) -> dict[str, object]:
    sequence: list[int] = []
    requests_by_row: dict[int, set[str | None]] = collections.defaultdict(set)
    observed_row_bytes: set[int] = set()

    for line_number, line in enumerate(trace_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            row = int(record["physical_row_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{trace_path}:{line_number}: invalid physical_row_id"
            ) from exc

        sequence.append(row)
        requests_by_row[row].add(record.get("request_id"))
        if record.get("bytes") is not None:
            observed_row_bytes.add(int(record["bytes"]))

    if not sequence:
        raise ValueError("PLE trace contains no row accesses")

    row_bytes = explicit_row_bytes
    if row_bytes is None:
        if len(observed_row_bytes) != 1:
            raise ValueError(
                "row byte size is unavailable or non-uniform; pass --row-bytes"
            )
        row_bytes = next(iter(observed_row_bytes))

    frequencies = collections.Counter(sequence)
    distances, cold_accesses = exact_reuse_distances(sequence)

    cache_curve = []
    for capacity_gib in capacities_gib:
        capacity_rows = int(capacity_gib * 1024**3 // row_bytes)
        hits = lru_hit_count(sequence, capacity_rows)
        cache_curve.append(
            {
                "capacity_gib": capacity_gib,
                "capacity_rows": capacity_rows,
                "hit_rate": hits / len(sequence),
            }
        )

    return {
        "schema_version": 1,
        "evidence_kind": "local_measured_trace_analysis",
        "accesses": len(sequence),
        "unique_rows": len(frequencies),
        "row_bytes": row_bytes,
        "working_set_bytes": len(frequencies) * row_bytes,
        "inter_request_reused_rows": sum(
            len(request_ids) > 1 for request_ids in requests_by_row.values()
        ),
        "reuse_distance": {
            "definition": "unique PLE rows accessed since the previous access to the same row",
            "reused_accesses": len(distances),
            "cold_accesses": cold_accesses,
            "p50_rows": quantile(distances, 0.50),
            "p95_rows": quantile(distances, 0.95),
            "p99_rows": quantile(distances, 0.99),
            "max_rows": max(distances) if distances else None,
            "histogram": logarithmic_histogram(distances),
        },
        "top_rows": [
            {"physical_row_id": row, "accesses": count}
            for row, count in frequencies.most_common(100)
        ],
        "cache_curve": cache_curve,
    }


def parse_capacities(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("capacities must be positive")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--row-bytes", type=int)
    parser.add_argument(
        "--capacities-gib",
        default="0.125,0.25,0.5,1,2,4,8,16,32,48",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = analyze_trace(
        args.trace,
        parse_capacities(args.capacities_gib),
        args.row_bytes,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
