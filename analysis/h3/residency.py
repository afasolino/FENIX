"""Byte-accurate finite-LPDDR residency replay for H3."""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Hashable

from analysis.h3.cache import CacheObject, cache_objects, make_cache, storage_record
from analysis.h3.common import GIB, H3Error, load_json, sha256_file, write_json
from analysis.h3.trace import Geometry, TraceEvent, iter_case_epochs


def _geometry(manifest: dict[str, Any]) -> Geometry:
    return Geometry(**{key: int(value) for key, value in manifest["geometry"].items()})


def _offline_static_frequency_profile(
    case_dir: Path,
    geometry: Geometry,
    phase_filter: str | None,
) -> dict[str, Any]:
    """Emit exact reuse-frequency evidence with separate residency/transfer sizes.

    Useful-object and VM-granularity placement are different optimization
    problems.  A PLE row can consume 160 B of managed resident capacity while a
    lower-tier miss transfers a 4-KiB page; expert useful bytes similarly differ
    from the aligned storage stride.  Keeping those dimensions separate prevents
    the offline oracle from accidentally weakening useful-object placement.
    """
    useful_ple: Counter[tuple[Hashable, ...]] = Counter()
    useful_expert: Counter[tuple[Hashable, ...]] = Counter()
    vm_ple: Counter[tuple[Hashable, ...]] = Counter()
    vm_expert: Counter[tuple[Hashable, ...]] = Counter()

    for epoch in iter_case_epochs(case_dir, geometry, phase_filter=phase_filter):
        epoch_useful_ple: set[tuple[Hashable, ...]] = set()
        epoch_useful_expert: set[tuple[Hashable, ...]] = set()
        epoch_vm_ple: set[tuple[Hashable, ...]] = set()
        epoch_vm_expert: set[tuple[Hashable, ...]] = set()
        for event in epoch.events:
            if event.kind == "ple":
                epoch_useful_ple.update(("ple_row", int(value)) for value in event.object_ids)
            elif event.kind == "expert":
                if event.layer is None:
                    raise H3Error("expert event lacks layer for offline profile")
                epoch_useful_expert.update(
                    ("expert", int(event.layer), int(value)) for value in event.object_ids
                )
            else:
                raise H3Error(f"unknown trace event kind {event.kind}")
            for obj in cache_objects(event, geometry, "vm_granularity_lru"):
                for storage_key in obj.storage_keys:
                    (epoch_vm_ple if obj.kind == "ple" else epoch_vm_expert).add(storage_key)
        useful_ple.update(epoch_useful_ple)
        useful_expert.update(epoch_useful_expert)
        vm_ple.update(epoch_vm_ple)
        vm_expert.update(epoch_vm_expert)

    def histogram(counter: Counter[tuple[Hashable, ...]]) -> dict[str, int]:
        out: Counter[int] = Counter(int(value) for value in counter.values())
        return {str(freq): int(count) for freq, count in sorted(out.items(), reverse=True)}

    def node(
        counter: Counter[tuple[Hashable, ...]], resident_bytes: int, transfer_bytes: int
    ) -> dict[str, Any]:
        return {
            "resident_object_bytes": int(resident_bytes),
            "transfer_object_bytes": int(transfer_bytes),
            # object_bytes retained only as a compatibility alias for readers;
            # capacity/service code in v5 uses the two explicit dimensions.
            "object_bytes": int(resident_bytes),
            "frequency_histogram": histogram(counter),
            "unique_objects": len(counter),
        }

    return {
        "role": "post_fio_service_time_optimized_static_frequency_profile",
        "first_touch": "compulsory_miss",
        "profiles": {
            "useful_object": {
                "granularity": "useful_resident_objects_with_storage_transfer_footprint",
                "ple": node(useful_ple, geometry.ple_row_bytes, geometry.storage_page_bytes),
                "expert": node(
                    useful_expert, geometry.expert_bytes, geometry.expert_stride_bytes
                ),
            },
            "vm_granularity": {
                "granularity": "vm_storage_objects",
                "ple": node(vm_ple, geometry.storage_page_bytes, geometry.storage_page_bytes),
                "expert": node(
                    vm_expert, geometry.expert_stride_bytes, geometry.expert_stride_bytes
                ),
            },
        },
        "policy_profile_mapping": {
            "useful_object_lru": "useful_object",
            "useful_object_lfu": "useful_object",
            "vm_granularity_lru": "vm_granularity",
            "vm_granularity_lfu": "vm_granularity",
        },
    }

def replay_case(
    manifest_path: Path,
    stratum: str,
    capacity_gib: float,
    policy: str,
    out_dir: Path,
    phase_filter: str | None = None,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    geometry = _geometry(manifest)
    cases = [case for case in manifest["cases"] if case["stratum"] == stratum]
    if len(cases) != 1:
        raise H3Error(f"manifest contains {len(cases)} cases for {stratum}")
    case = cases[0]
    case_dir = Path(case["case_dir"])
    for relative, expected in case["source_sha256"].items():
        path = case_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise H3Error(f"source trace drift: {path}")

    capacity_bytes = int(float(capacity_gib) * GIB)
    cache = make_cache(policy, capacity_bytes)
    counters: Counter[str] = Counter()
    phase_counters: dict[str, Counter[str]] = defaultdict(Counter)
    out_dir.mkdir(parents=True, exist_ok=True)
    miss_path = out_dir / "misses.jsonl.gz"
    logical_trace_path = out_dir / "logical_access_trace.jsonl.gz"
    sequence = 0
    logical_sequence = 0

    with gzip.open(miss_path, "wt", encoding="utf-8", compresslevel=6) as stream, gzip.open(
        logical_trace_path, "wt", encoding="utf-8", compresslevel=6
    ) as logical_stream:
        for epoch in iter_case_epochs(case_dir, geometry, phase_filter=phase_filter):
            event_objects: list[tuple[TraceEvent, list[CacheObject]]] = []
            all_objects: list[CacheObject] = []
            for event in epoch.events:
                objects = cache_objects(event, geometry, policy)
                event_objects.append((event, objects))
                all_objects.extend(objects)
            hits = cache.classify_epoch(all_objects)
            storage_misses: dict[tuple[Hashable, ...], tuple[TraceEvent, str]] = {}

            for event, objects in event_objects:
                logical_record = {
                    "sequence": logical_sequence,
                    "timestamp_ns_order_only": epoch.timestamp_ns,
                    "stratum": stratum,
                    "phase": event.phase,
                    "operation": "logical_read",
                    "kind": event.kind,
                    "layer": event.layer,
                    "object_ids": [int(value) for value in event.object_ids],
                }
                logical_stream.write(json.dumps(logical_record, separators=(",", ":")) + "\n")
                logical_sequence += 1
                phase = phase_counters[event.phase]
                if event.kind == "ple":
                    logical = len(event.object_ids)
                    useful = logical * geometry.ple_row_bytes
                    counters["ple_row_accesses"] += logical
                    counters["lpddr_read_useful_bytes"] += useful
                    phase["ple_row_accesses"] += logical
                    phase["lpddr_read_useful_bytes"] += useful
                else:
                    logical = len(event.object_ids)
                    useful = logical * geometry.expert_bytes
                    counters["expert_selections"] += logical
                    counters["lpddr_read_useful_bytes"] += useful
                    phase["expert_selections"] += logical
                    phase["lpddr_read_useful_bytes"] += useful

                for obj in objects:
                    if hits[obj.key]:
                        counters[f"{obj.kind}_cache_hits"] += 1
                        phase[f"{obj.kind}_cache_hits"] += 1
                    else:
                        counters[f"{obj.kind}_cache_misses"] += 1
                        phase[f"{obj.kind}_cache_misses"] += 1
                        for storage_key in obj.storage_keys:
                            # Coalesce identical lower-tier fetches within the same timestamp.
                            storage_misses.setdefault(storage_key, (event, obj.kind))

            fill_records: list[dict[str, Any]] = []
            for storage_key, (event, _) in sorted(storage_misses.items(), key=lambda item: repr(item[0])):
                kind, offset, length, useful = storage_record(storage_key, geometry)
                fill_records.append({
                    "storage_key": list(storage_key),
                    "kind": kind,
                    "offset_bytes": int(offset),
                    "length_bytes": int(length),
                    "phase": event.phase,
                })
                record = {
                    "sequence": sequence,
                    "timestamp_ns_order_only": epoch.timestamp_ns,
                    "stratum": stratum,
                    "request_id": event.request_id,
                    "phase": event.phase,
                    "kind": kind,
                    "offset_bytes": offset,
                    "length_bytes": length,
                    "useful_bytes": useful,
                    "storage_key": list(storage_key),
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                sequence += 1
                counters["storage_requests"] += 1
                counters["storage_bytes"] += length
                counters[f"{kind}_storage_requests"] += 1
                counters[f"{kind}_storage_bytes"] += length
                counters["lpddr_fill_write_bytes"] += length
                phase = phase_counters[event.phase]
                phase["storage_requests"] += 1
                phase["storage_bytes"] += length
                phase[f"{kind}_storage_bytes"] += length
                phase["lpddr_fill_write_bytes"] += length

            if fill_records:
                logical_stream.write(json.dumps({
                    "sequence": logical_sequence,
                    "timestamp_ns_order_only": epoch.timestamp_ns,
                    "stratum": stratum,
                    "phase": None,
                    "operation": "fill_write",
                    "fills": fill_records,
                }, separators=(",", ":")) + "\n")
                logical_sequence += 1

            cache.commit_epoch(all_objects)

    h3_contract_sha = (manifest.get("inputs") or {}).get("h3_contract_sha256")
    if not h3_contract_sha:
        raise H3Error("trace manifest lacks exact H3 contract provenance")
    offline_profile = _offline_static_frequency_profile(case_dir, geometry, phase_filter)
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_residency_replay",
        "evidence_kind": "trace_projection_exact_capacity",
        "stratum": stratum,
        "phase_filter": phase_filter,
        "policy": policy,
        "replacement_policy": "lfu" if policy.endswith("_lfu") else "lru",
        "capacity_gib": float(capacity_gib),
        "capacity_bytes": capacity_bytes,
        "cold_start": True,
        "same_timestamp_atomic": True,
        "same_timestamp_storage_coalescing": True,
        "final_resident_bytes": cache.resident_bytes,
        "counters": dict(counters),
        "phase_counters": {name: dict(values) for name, values in phase_counters.items()},
        "offline_static_frequency_profile": offline_profile,
        "deterministic_ple_prefetch_sensitivity": {
            "role": "optimistic_conventional_sensitivity_only",
            "assumption": "all PLE lower-tier reads are issued early enough to hide storage latency; bytes and LPDDR fills are not removed",
            "ple_prefetchable_storage_bytes": int(counters.get("ple_storage_bytes", 0)),
            "nonprefetchable_expert_storage_bytes": int(counters.get("expert_storage_bytes", 0)),
        },
        "ple_and_expert_perfect_prefetch_sensitivity": {
            "role": "hostile_conventional_upper_envelope_not_a_measured_predictor",
            "assumption": "all PLE and expert lower-tier latency is perfectly hidden while storage traffic and LPDDR fills remain physically charged",
            "prefetchable_storage_bytes": int(counters.get("storage_bytes", 0)),
        },
        "oracle": {
            "definition": "same logical conditional-state demand with all conditional state already resident in LPDDR",
            "lpddr_read_useful_bytes": counters["lpddr_read_useful_bytes"],
            "storage_bytes": 0,
            "lpddr_fill_write_bytes": 0,
        },
        "logical_access_trace": {
            "path": str(logical_trace_path.resolve()),
            "sha256": sha256_file(logical_trace_path),
            "records": int(logical_sequence),
            "semantics": "full logical conditional reads plus finite-hierarchy fill writes in source timestamp order",
        },
        "miss_stream": {
            "path": str(miss_path.resolve()),
            "sha256": sha256_file(miss_path),
            "records": counters["storage_requests"],
            "bytes": counters["storage_bytes"],
        },
        "contract_sha256": str(h3_contract_sha),
        "provenance": {
            "trace_manifest": str(manifest_path.resolve()),
            "trace_manifest_sha256": sha256_file(manifest_path),
            "contract_sha256": str(h3_contract_sha),
        },
    }
    write_json(out_dir / "summary.json", result)
    return result
