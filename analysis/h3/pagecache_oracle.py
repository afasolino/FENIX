"""Rootless future-aware page-cache dominance oracle for H3.

The oracle is intentionally stronger than the Linux page-cache control that the
frozen v6 campaign attempted to measure.  It grants free PLE storage traffic,
free expert terminal-page traffic, all declared cache capacity to expert full
pages, exact future knowledge, and perfect hiding of storage latency for the
remaining misses.  Only unavoidable LPDDR fill/bus traffic remains.

Consequently the artifact is valid only for gap falsification.  It can never be
used to establish conventional-memory sufficiency.
"""
from __future__ import annotations

import heapq
from array import array
from pathlib import Path
from typing import Any, Iterator

from analysis.h3.common import GIB, H3Error, load_json, sha256_file, write_json

_U64_INF = (1 << 64) - 1


def load_analysis_amendment(path: Path, contract_path: Path) -> dict[str, Any]:
    amendment = load_json(path)
    if amendment.get("artifact_kind") != "fenix_h3_analysis_amendment_contract":
        raise H3Error("unexpected H3 analysis-amendment artifact")
    if amendment.get("base_contract_sha256") != sha256_file(contract_path):
        raise H3Error("analysis amendment is not bound to the supplied H3 contract")
    oracle = amendment.get("pagecache_oracle") or {}
    required_true = (
        "ple_storage_traffic_free",
        "expert_partial_tail_free",
        "all_capacity_dedicated_to_expert_pages",
        "storage_latency_hidden",
        "gap_falsification_allowed",
    )
    for key in required_true:
        if oracle.get(key) is not True:
            raise H3Error(f"analysis amendment lacks hostile-oracle guarantee {key}")
    if oracle.get("conventional_sufficiency_allowed") is not False:
        raise H3Error("oracle must be prohibited from conventional-sufficiency promotion")
    if oracle.get("algorithm") != "belady_min_equal_page_future_oracle":
        raise H3Error("analysis amendment declares an unsupported oracle algorithm")
    if len(str(amendment.get("frozen_measurement_head", ""))) != 40:
        raise H3Error("analysis amendment lacks a frozen measurement commit")
    return amendment


def _iter_iolog_reads(
    iolog: Path,
    *,
    ple_row_bytes: int,
    expert_useful_bytes: int,
    expert_stride_bytes: int,
) -> Iterator[tuple[str, int]]:
    with iolog.open("rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            parts = line.split()
            if not parts or parts[0] == "fio":
                continue
            if len(parts) >= 2 and parts[1] in {"add", "open", "close"}:
                continue
            if len(parts) != 4 or parts[1] != "read":
                raise H3Error(f"{iolog}:{line_number}: unexpected iolog record")
            try:
                offset = int(parts[2])
                length = int(parts[3])
            except ValueError as exc:
                raise H3Error(f"{iolog}:{line_number}: invalid read geometry") from exc
            if offset < 0 or length <= 0:
                raise H3Error(f"{iolog}:{line_number}: invalid read geometry")
            if length == ple_row_bytes:
                yield "ple", offset
            elif length == expert_useful_bytes:
                if offset % expert_stride_bytes:
                    raise H3Error(f"{iolog}:{line_number}: expert read is not stride aligned")
                yield "expert", offset // expert_stride_bytes
            else:
                raise H3Error(
                    f"{iolog}:{line_number}: unexpected read length {length}"
                )


def _next_use_table(
    iolog: Path,
    *,
    ple_row_bytes: int,
    expert_useful_bytes: int,
    expert_stride_bytes: int,
) -> tuple[array, int, int, int]:
    next_use = array("Q")
    last: dict[int, int] = {}
    ple_reads = 0
    expert_reads = 0
    for kind, key in _iter_iolog_reads(
        iolog,
        ple_row_bytes=ple_row_bytes,
        expert_useful_bytes=expert_useful_bytes,
        expert_stride_bytes=expert_stride_bytes,
    ):
        if kind == "ple":
            ple_reads += 1
            continue
        position = expert_reads
        expert_reads += 1
        next_use.append(_U64_INF)
        previous = last.get(key)
        if previous is not None:
            next_use[previous] = position
        last[key] = position
    if expert_reads == 0:
        raise H3Error("page-cache oracle source contains no expert reads")
    return next_use, ple_reads, expert_reads, len(last)


def _simulate_sequence(
    expert_keys: Iterator[int],
    next_use: array,
    *,
    pages_per_object: int,
    capacity_pages: int,
) -> dict[str, int]:
    """Exact MIN for equal-size/equal-cost pages grouped by expert object.

    Every full page of one expert has the same object-level reference sequence.
    A resident-page count per expert is therefore equivalent to expanding all
    pages individually.  Lazy heap entries are periodically rebuilt so memory is
    proportional to live cache state rather than trace length.
    """
    if pages_per_object <= 0 or capacity_pages <= 0:
        raise H3Error("oracle cache geometry must be positive")
    resident_count: dict[int, int] = {}
    resident_next: dict[int, int] = {}
    version: dict[int, int] = {}
    heap: list[tuple[int, int, int]] = []
    resident_pages = 0
    peak_pages = 0
    hit_pages = 0
    miss_pages = 0
    evicted_pages = 0
    accesses = 0

    def rebuild() -> None:
        heap.clear()
        heap.extend(
            (-resident_next[key], version[key], key) for key in resident_count
        )
        heapq.heapify(heap)

    for position, raw_key in enumerate(expert_keys):
        if position >= len(next_use):
            raise H3Error("oracle second pass has more expert reads than first pass")
        key = int(raw_key)
        nxt = int(next_use[position])
        present = int(resident_count.get(key, 0))
        if not 0 <= present <= pages_per_object:
            raise H3Error("oracle resident-page accounting is corrupt")
        hit_pages += present
        misses = pages_per_object - present
        miss_pages += misses
        resident_pages += misses
        resident_count[key] = pages_per_object
        resident_next[key] = nxt
        version[key] = version.get(key, 0) + 1
        heapq.heappush(heap, (-nxt, version[key], key))

        while resident_pages > capacity_pages:
            while heap:
                neg_next, candidate_version, victim = heapq.heappop(heap)
                if (
                    victim in resident_count
                    and version.get(victim) == candidate_version
                    and -neg_next == resident_next.get(victim)
                ):
                    break
            else:
                raise H3Error("oracle eviction heap became empty")
            excess = resident_pages - capacity_pages
            count = resident_count[victim]
            evict = min(count, excess)
            if evict <= 0:
                raise H3Error("oracle eviction made no progress")
            remaining = count - evict
            resident_pages -= evict
            evicted_pages += evict
            version[victim] = version.get(victim, 0) + 1
            if remaining:
                resident_count[victim] = remaining
                heapq.heappush(
                    heap,
                    (-resident_next[victim], version[victim], victim),
                )
            else:
                resident_count.pop(victim, None)
                resident_next.pop(victim, None)

        peak_pages = max(peak_pages, resident_pages)
        accesses += 1
        live = max(1, len(resident_count))
        if len(heap) > max(16_384, 4 * live):
            rebuild()

    if accesses != len(next_use):
        raise H3Error(
            f"oracle second-pass count {accesses} != first-pass count {len(next_use)}"
        )
    if resident_pages > capacity_pages:
        raise H3Error("oracle exceeded declared cache capacity")
    return {
        "expert_accesses": accesses,
        "expert_full_page_hits": hit_pages,
        "expert_full_page_misses": miss_pages,
        "expert_full_pages_evicted": evicted_pages,
        "peak_resident_pages": peak_pages,
        "resident_pages_final": resident_pages,
        "resident_expert_objects_final": len(resident_count),
    }


def simulate_grouped_belady(
    sequence: list[int], *, pages_per_object: int, capacity_pages: int
) -> dict[str, int]:
    """Small-sequence reference entry point used by qualification tests."""
    next_use = array("Q", [_U64_INF] * len(sequence))
    last: dict[int, int] = {}
    for position, raw_key in enumerate(sequence):
        key = int(raw_key)
        previous = last.get(key)
        if previous is not None:
            next_use[previous] = position
        last[key] = position
    return _simulate_sequence(
        iter(int(value) for value in sequence),
        next_use,
        pages_per_object=pages_per_object,
        capacity_pages=capacity_pages,
    )


def build_pagecache_oracle(
    window_path: Path,
    contract_path: Path,
    amendment_path: Path,
    capacity_gib: float,
    out_path: Path,
) -> dict[str, Any]:
    contract = load_json(contract_path)
    amendment = load_analysis_amendment(amendment_path, contract_path)
    window = load_json(window_path)
    if window.get("artifact_kind") != "fenix_h3_pagecache_window":
        raise H3Error("unexpected source artifact for page-cache oracle")
    if window.get("full_trace_replayed") is not True:
        raise H3Error("page-cache oracle requires a full-stratum window")
    if window.get("contract_sha256") != sha256_file(contract_path):
        raise H3Error("page-cache window uses another H3 contract")
    if float(window.get("capacity_gib", -1)) != float(capacity_gib):
        raise H3Error("oracle capacity differs from source page-cache window")

    oracle_cfg = amendment["pagecache_oracle"]
    stratum = str(window.get("stratum"))
    if stratum not in {str(value) for value in oracle_cfg["required_strata"]}:
        raise H3Error(f"oracle is not declared for stratum {stratum}")
    if float(oracle_cfg["capacity_gib"]) != float(capacity_gib):
        raise H3Error("analysis-amendment capacity differs from oracle request")

    source_identity = window.get("execution_repository")
    if not isinstance(source_identity, dict):
        raise H3Error("page-cache window lacks measurement execution identity")
    if source_identity.get("head") != amendment["frozen_measurement_head"]:
        raise H3Error("page-cache window is not from the frozen measurement commit")

    iolog = Path(str(window.get("iolog", "")))
    if not iolog.is_file() or sha256_file(iolog) != window.get("iolog_sha256"):
        raise H3Error("page-cache oracle source iolog is missing or changed")

    geometry = contract["geometry"]
    page_bytes = int(geometry["storage_page_bytes"])
    expert_useful = int(geometry["expert_useful_bytes"])
    expert_stride = int(geometry["expert_storage_stride_bytes"])
    ple_row = int(geometry["ple_row_bytes"])
    full_pages = expert_useful // page_bytes
    ignored_tail = expert_useful % page_bytes
    if full_pages <= 0 or ignored_tail <= 0:
        raise H3Error("oracle expects an expert transfer with a partial final page")
    if expert_stride % page_bytes:
        raise H3Error("expert storage stride is not page aligned")

    capacity_bytes = int(float(capacity_gib) * GIB)
    capacity_pages = capacity_bytes // page_bytes
    next_use, ple_reads, expert_reads, unique_experts = _next_use_table(
        iolog,
        ple_row_bytes=ple_row,
        expert_useful_bytes=expert_useful,
        expert_stride_bytes=expert_stride,
    )

    def expert_keys() -> Iterator[int]:
        for kind, key in _iter_iolog_reads(
            iolog,
            ple_row_bytes=ple_row,
            expert_useful_bytes=expert_useful,
            expert_stride_bytes=expert_stride,
        ):
            if kind == "expert":
                yield key

    simulated = _simulate_sequence(
        expert_keys(),
        next_use,
        pages_per_object=full_pages,
        capacity_pages=capacity_pages,
    )
    lower_tier_read_bytes = simulated["expert_full_page_misses"] * page_bytes
    total_expert_full_bytes = expert_reads * full_pages * page_bytes
    logical_bytes = int(window["logical_bytes"])
    result = {
        "schema_version": 1,
        "artifact_kind": "fenix_h3_pagecache_oracle",
        "oracle_role": "gap_falsification_only",
        "algorithm": "belady_min_equal_page_future_oracle",
        "stratum": stratum,
        "capacity_gib": float(capacity_gib),
        "capacity_bytes": capacity_bytes,
        "cache_page_bytes": page_bytes,
        "cache_capacity_pages": capacity_pages,
        "future_knowledge": True,
        "ple_storage_traffic_free": True,
        "ple_cache_capacity_bytes": 0,
        "expert_partial_tail_free": True,
        "expert_partial_tail_bytes_ignored_per_access": ignored_tail,
        "expert_full_pages_per_access": full_pages,
        "all_capacity_dedicated_to_expert_pages": True,
        "metadata_overhead_bytes": 0,
        "process_memory_overhead_bytes": 0,
        "storage_latency_hidden": True,
        "logical_read_bytes": logical_bytes,
        "ple_logical_reads": ple_reads,
        "expert_logical_reads": expert_reads,
        "unique_experts": unique_experts,
        **simulated,
        "expert_full_reference_bytes": total_expert_full_bytes,
        "expert_full_bytes_avoided": total_expert_full_bytes - lower_tier_read_bytes,
        "lower_tier_read_bytes": lower_tier_read_bytes,
        "lower_tier_read_fraction_of_logical_bytes": (
            lower_tier_read_bytes / logical_bytes if logical_bytes else None
        ),
        "gap_falsification_allowed": True,
        "conventional_sufficiency_allowed": False,
        "dominance_semantics": (
            "PLE storage and expert terminal-page traffic are free; all declared "
            "capacity is dedicated to expert full pages; replacement has exact "
            "future knowledge; remaining storage latency is perfectly hidden; "
            "only expert-capacity-miss LPDDR fill/bus traffic remains"
        ),
        "source_window": str(window_path.resolve()),
        "source_window_sha256": sha256_file(window_path),
        "source_iolog": str(iolog.resolve()),
        "source_iolog_sha256": sha256_file(iolog),
        "trace_manifest_sha256": window.get("manifest_sha256"),
        "contract_sha256": sha256_file(contract_path),
        "analysis_amendment_sha256": sha256_file(amendment_path),
        "source_measurement_execution_repository": source_identity,
        "frozen_measurement_head": amendment["frozen_measurement_head"],
    }
    write_json(out_path, result)
    return result
