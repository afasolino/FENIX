"""Same-endpoint H3 memory-service bounds and campaign promotion gate."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from analysis.h3.common import (
    H3Error,
    load_json,
    require_contract_sha,
    require_same_execution_repository,
    sha256_file,
    write_json,
)
from analysis.h3.storage import project_storage_bytes_ns, project_storage_ns


def _service_ns(byte_count: float, bandwidth_gb_s: float) -> float:
    # Decimal GB/s and ns: 1 byte/ns == 1 GB/s.
    if bandwidth_gb_s <= 0:
        raise H3Error("bandwidth must be positive")
    return float(byte_count) / float(bandwidth_gb_s)


def _expected_lpddr_ids(contract: dict[str, Any]) -> set[str]:
    lpddr = contract["lpddr"]
    regimes = [str(value) for value in lpddr.get("calibration_regimes", ["streaming"])]
    ids = {
        f"ramulator_{regime}_nCS{int(ncs)}"
        for regime in regimes
        for ncs in lpddr["ncs_values"]
    }
    ids.add("platform_theoretical_peak")
    return ids


def _storage_coeff_envelope(fio: dict[str, Any], qd: int) -> dict[str, float]:
    ple = fio["class_coefficients"][f"ple:qd{int(qd)}"]
    expert = fio["class_coefficients"][f"expert:qd{int(qd)}"]
    return {
        "fastest_low_ns_per_byte": min(
            float(ple["ns_per_byte_ci95"][0]),
            float(expert["ns_per_byte_ci95"][0]),
        ),
        "slowest_high_ns_per_byte": max(
            float(ple["ns_per_byte_ci95"][1]),
            float(expert["ns_per_byte_ci95"][1]),
        ),
    }


def _candidate_bounds(
    *,
    read_bytes: float,
    fill_bytes: float,
    storage: dict[str, Any],
    read_bw: float,
    write_bw: float,
    mixed_bw: float,
) -> tuple[float, float, float, float, float]:
    oracle = _service_ns(read_bytes, read_bw)
    fill = _service_ns(fill_bytes, write_bw)
    # Reads and fills share the same LPDDR data interface. The measured mixed
    # bandwidth therefore supplies a stronger lower bound than treating the two
    # directions as perfectly independent.
    shared_bus = _service_ns(read_bytes + fill_bytes, mixed_bw)
    lower = max(oracle, shared_bus, float(storage["low_ns"]))
    upper = oracle + fill + float(storage["high_ns"])
    return oracle, fill, shared_bus, lower, upper



def _hist_top_k_savings(
    node: dict[str, Any], selected: int, ns_per_byte: float
) -> tuple[float, int]:
    """Return optimistic saved service and transfer bytes for top-frequency objects."""
    remaining = max(0, int(selected))
    obj_bytes = int(node["transfer_object_bytes"])
    saved_ns = 0.0
    saved_bytes = 0
    histogram = node.get("frequency_histogram") or {}
    for frequency_text, object_count in sorted(
        histogram.items(), key=lambda item: int(item[0]), reverse=True
    ):
        if remaining <= 0:
            break
        frequency = int(frequency_text)
        take = min(remaining, int(object_count))
        repeat_misses_saved = max(0, frequency - 1) * take
        bytes_saved = repeat_misses_saved * obj_bytes
        saved_bytes += bytes_saved
        saved_ns += bytes_saved * float(ns_per_byte)
        remaining -= take
    return saved_ns, saved_bytes


def _service_time_optimized_offline_bound(
    profile: dict[str, Any],
    fio: dict[str, Any],
    qd: int,
    capacity_bytes: int,
    lpddr_write_bandwidth_gb_s: float,
    policy: str,
    baseline_ple_storage_bytes: int,
    baseline_expert_storage_bytes: int,
) -> dict[str, Any]:
    """Solve the two-size static oracle with measured service-time objective.

    Resident footprint and lower-tier transfer footprint are intentionally
    distinct.  Savings are capped by the exact replay's observed per-class
    storage bytes, so the optimistic frequency bound cannot subtract more traffic
    than the finite hierarchy actually generated.
    """
    mapping = profile.get("policy_profile_mapping") or {}
    profile_name = mapping.get(str(policy))
    profiles = profile.get("profiles") or {}
    selected_profile = profiles.get(profile_name) if profile_name else None
    if not isinstance(selected_profile, dict):
        raise H3Error(f"offline frequency profile does not support policy {policy}")
    ple = selected_profile["ple"]
    expert = selected_profile["expert"]
    write_coeff = 1.0 / float(lpddr_write_bandwidth_gb_s)
    ple_coeff = float(fio["class_coefficients"][f"ple:qd{qd}"]["ns_per_byte_median"]) + write_coeff
    expert_coeff = float(fio["class_coefficients"][f"expert:qd{qd}"]["ns_per_byte_median"]) + write_coeff
    ple_resident = int(ple["resident_object_bytes"])
    expert_resident = int(expert["resident_object_bytes"])
    max_experts = min(int(expert["unique_objects"]), int(capacity_bytes) // expert_resident)
    best: tuple[float, int, int, int, int] | None = None
    for expert_count in range(max_experts + 1):
        remaining = int(capacity_bytes) - expert_count * expert_resident
        ple_count = min(int(ple["unique_objects"]), remaining // ple_resident)
        expert_saved_ns, expert_saved_bytes = _hist_top_k_savings(
            expert, expert_count, expert_coeff
        )
        ple_saved_ns, ple_saved_bytes = _hist_top_k_savings(ple, ple_count, ple_coeff)
        # Storage coalescing/shared pages can make independent-object savings
        # non-additive.  This is deliberately an optimistic gap-falsification
        # oracle, but it still cannot save more than measured baseline traffic.
        expert_saved_bytes = min(int(baseline_expert_storage_bytes), int(expert_saved_bytes))
        ple_saved_bytes = min(int(baseline_ple_storage_bytes), int(ple_saved_bytes))
        saved_ns = (
            expert_saved_bytes * expert_coeff + ple_saved_bytes * ple_coeff
        )
        candidate = (
            saved_ns,
            expert_count,
            ple_count,
            expert_saved_bytes,
            ple_saved_bytes,
        )
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    saved_ns, expert_count, ple_count, expert_saved_bytes, ple_saved_bytes = best
    ple_storage = max(0, int(baseline_ple_storage_bytes) - ple_saved_bytes)
    expert_storage = max(0, int(baseline_expert_storage_bytes) - expert_saved_bytes)
    return {
        "role": "optimistic_oracle_trained_static_bound",
        "objective": "measured_fio_storage_plus_lpddr_fill_service_ns_saved",
        "queue_depth": int(qd),
        "capacity_bytes": int(capacity_bytes),
        "policy": str(policy),
        "profile": str(profile_name),
        "selected_ple_objects": int(ple_count),
        "selected_expert_objects": int(expert_count),
        "ple_resident_object_bytes": ple_resident,
        "ple_transfer_object_bytes": int(ple["transfer_object_bytes"]),
        "expert_resident_object_bytes": expert_resident,
        "expert_transfer_object_bytes": int(expert["transfer_object_bytes"]),
        "resident_bytes": int(ple_count * ple_resident + expert_count * expert_resident),
        "saved_storage_service_ns_median_objective": float(saved_ns),
        "ple_storage_bytes": int(ple_storage),
        "expert_storage_bytes": int(expert_storage),
        "storage_bytes": int(ple_storage + expert_storage),
        "optimistic_shared_transfer_savings": True,
    }

def _candidate_interval_summary(
    candidates: list[dict[str, Any]], oracle: float
) -> dict[str, Any]:
    realizable = [row for row in candidates if row.get("promotion_role") == "realizable"]
    if not realizable:
        raise H3Error("H3 decision has no realizable conventional candidate")
    all_low = min(float(row["service_ns_bounds"][0]) for row in candidates)
    all_high = min(float(row["service_ns_bounds"][1]) for row in candidates)
    real_low = min(float(row["service_ns_bounds"][0]) for row in realizable)
    real_high = min(float(row["service_ns_bounds"][1]) for row in realizable)
    return {
        "gap_falsification_penalty_fraction_bounds": [
            max(0.0, all_low / oracle - 1.0),
            max(0.0, all_high / oracle - 1.0),
        ],
        "realizable_penalty_fraction_bounds": [
            max(0.0, real_low / oracle - 1.0),
            max(0.0, real_high / oracle - 1.0),
        ],
    }

def _evaluate_service_counters(
    counters: dict[str, Any],
    lpddr: dict[str, Any],
    fio: dict[str, Any],
    contract: dict[str, Any],
    queue_depth: int,
    *,
    offline_profile: dict[str, Any] | None = None,
    prefetch_sensitivity: dict[str, Any] | None = None,
    all_prefetch_sensitivity: dict[str, Any] | None = None,
    pagecache: list[tuple[Path, dict[str, Any]]] | None = None,
    additional_lpddr_points: list[dict[str, Any]] | None = None,
    capacity_bytes: int | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    read_bytes = float(counters.get("lpddr_read_useful_bytes", 0))
    fill_bytes = float(counters.get("lpddr_fill_write_bytes", 0))
    if read_bytes <= 0:
        raise H3Error("residency replay has no LPDDR conditional-state reads")
    synthetic_residency = {"counters": counters}
    storage = project_storage_ns(synthetic_residency, fio, int(queue_depth))
    thresholds = contract["decision_gate"]
    coeff_envelope = _storage_coeff_envelope(fio, int(queue_depth))

    points: list[dict[str, Any]] = []
    global_low = float("inf")
    global_high = 0.0
    calibrations = list(lpddr["sensitivity_points"]) + list(additional_lpddr_points or [])
    for calibration in calibrations:
        read_bw = float(calibration["aggregate_read_bandwidth_gb_s"])
        write_bw = float(calibration["aggregate_write_bandwidth_gb_s"])
        mixed_bw = float(
            calibration.get(
                "aggregate_mixed_total_bandwidth_gb_s",
                min(read_bw, write_bw),
            )
        )
        oracle, fill, shared_bus, explicit_low, explicit_high = _candidate_bounds(
            read_bytes=read_bytes,
            fill_bytes=fill_bytes,
            storage=storage,
            read_bw=read_bw,
            write_bw=write_bw,
            mixed_bw=mixed_bw,
        )
        candidates: list[dict[str, Any]] = [
            {
                "baseline": "explicit_trace_cache_plus_direct_storage",
                "promotion_role": "realizable",
                "service_ns_bounds": [explicit_low, explicit_high],
                "storage_read_bytes": float(counters.get("storage_bytes", 0)),
                "shared_lpddr_bus_lower_bound_ns": shared_bus,
            }
        ]

        if offline_profile is not None:
            if capacity_bytes is None:
                raise H3Error("offline service-time optimization requires capacity_bytes")
            if policy is None:
                raise H3Error("offline service-time optimization requires policy")
            offline_bound = _service_time_optimized_offline_bound(
                offline_profile,
                fio,
                int(queue_depth),
                int(capacity_bytes),
                write_bw,
                str(policy),
                int(counters.get("ple_storage_bytes", 0)),
                int(counters.get("expert_storage_bytes", 0)),
            )
            bound_storage = project_storage_bytes_ns(
                float(offline_bound.get("ple_storage_bytes", 0)),
                float(offline_bound.get("expert_storage_bytes", 0)),
                fio,
                int(queue_depth),
            )
            bound_fill_bytes = float(offline_bound.get("storage_bytes", 0))
            _, bound_fill, bound_bus, bound_low, bound_high = _candidate_bounds(
                read_bytes=read_bytes,
                fill_bytes=bound_fill_bytes,
                storage=bound_storage,
                read_bw=read_bw,
                write_bw=write_bw,
                mixed_bw=mixed_bw,
            )
            candidates.append(
                {
                    "baseline": "service_time_optimized_offline_frequency_bound",
                    "promotion_role": "gap_falsification_only",
                    "role": "optimistic_oracle_trained_conventional_upper_bound",
                    "storage_read_bytes": bound_fill_bytes,
                    "storage_service_ns": bound_storage,
                    "lpddr_fill_service_ns": bound_fill,
                    "shared_lpddr_bus_lower_bound_ns": bound_bus,
                    "service_ns_bounds": [bound_low, bound_high],
                }
            )

        if prefetch_sensitivity is not None:
            # PLE addresses are deterministic. This sensitivity gives the
            # conventional hierarchy perfect latency hiding for PLE storage
            # reads while retaining their storage traffic and LPDDR fill bytes.
            expert_storage_bytes = float(
                prefetch_sensitivity.get("nonprefetchable_expert_storage_bytes", 0)
            )
            expert_only = project_storage_bytes_ns(
                0.0, expert_storage_bytes, fio, int(queue_depth)
            )
            shared_prefetch_bus = _service_ns(read_bytes + fill_bytes, mixed_bw)
            prefetch_low = max(oracle, shared_prefetch_bus, float(expert_only["low_ns"]))
            prefetch_high = oracle + fill + float(expert_only["high_ns"])
            candidates.append(
                {
                    "baseline": "deterministic_PLE_perfect_prefetch_sensitivity",
                    "promotion_role": "gap_falsification_only",
                    "role": "optimistic_latency_hiding_sensitivity_not_measured_prefetcher",
                    "ple_storage_latency_hidden": True,
                    "ple_storage_bytes_retained": float(
                        prefetch_sensitivity.get("ple_prefetchable_storage_bytes", 0)
                    ),
                    "expert_storage_service_ns": expert_only,
                    "shared_lpddr_bus_lower_bound_ns": shared_prefetch_bus,
                    "service_ns_bounds": [prefetch_low, prefetch_high],
                }
            )

        if all_prefetch_sensitivity is not None:
            # Deliberately stronger than a measured predictor: all storage latency
            # is assumed hidden, but the finite hierarchy still pays every LPDDR
            # fill byte and therefore competes for the shared LPDDR interface.
            perfect_bus = _service_ns(read_bytes + fill_bytes, mixed_bw)
            candidates.append(
                {
                    "baseline": "PLE_and_expert_perfect_prefetch_sensitivity",
                    "promotion_role": "gap_falsification_only",
                    "role": "hostile_conventional_upper_envelope_not_a_measured_predictor",
                    "storage_latency_hidden": True,
                    "storage_bytes_retained": float(
                        all_prefetch_sensitivity.get("prefetchable_storage_bytes", 0)
                    ),
                    "shared_lpddr_bus_lower_bound_ns": perfect_bus,
                    "service_ns_bounds": [max(oracle, perfect_bus), oracle + fill],
                }
            )

        for pagecache_path, pagecache_payload in pagecache or []:
            pagecache_storage_bytes = float(
                pagecache_payload["storage_read_bytes_cgroup"]
            )
            # A conservative interval must not use the fastest class for both
            # ends. Lower is intentionally favorable; upper uses the slowest
            # measured class coefficient.
            pagecache_storage_low = (
                pagecache_storage_bytes
                * coeff_envelope["fastest_low_ns_per_byte"]
            )
            pagecache_storage_high = (
                pagecache_storage_bytes
                * coeff_envelope["slowest_high_ns_per_byte"]
            )
            pagecache_fill = _service_ns(pagecache_storage_bytes, write_bw)
            pagecache_bus = _service_ns(
                read_bytes + pagecache_storage_bytes, mixed_bw
            )
            candidates.append(
                {
                    "baseline": f"linux_pagecache_{pagecache_payload['mode']}",
                    "promotion_role": "realizable",
                    "artifact": str(pagecache_path),
                    "artifact_sha256": sha256_file(pagecache_path),
                    "storage_read_bytes": pagecache_storage_bytes,
                    "storage_service_ns_bounds": [
                        pagecache_storage_low,
                        pagecache_storage_high,
                    ],
                    "measured_fio_wall_time_ns": pagecache_payload.get(
                        "fio_runtime_ns"
                    ),
                    "lpddr_fill_service_ns": pagecache_fill,
                    "shared_lpddr_bus_lower_bound_ns": pagecache_bus,
                    "service_ns_bounds": [
                        max(oracle, pagecache_bus, pagecache_storage_low),
                        oracle + pagecache_fill + pagecache_storage_high,
                    ],
                    "normalization": (
                        "lower=observed backing-device bytes times fastest class low coefficient; "
                        "upper=observed bytes times slowest class high coefficient"
                    ),
                }
            )

        interval = _candidate_interval_summary(candidates, oracle)
        gap_low, gap_high = interval["gap_falsification_penalty_fraction_bounds"]
        real_low, real_high = interval["realizable_penalty_fraction_bounds"]
        # Global gap support must survive every LPDDR calibration and the most
        # optimistic conventional bound. Conventional sufficiency, conversely,
        # must be demonstrated by a realizable measured candidate.
        global_low = min(global_low, gap_low)
        global_high = max(global_high, real_high)
        points.append(
            {
                "sensitivity_id": calibration["sensitivity_id"],
                "regime": calibration.get("regime"),
                "aggregate_read_bandwidth_gb_s": read_bw,
                "aggregate_write_bandwidth_gb_s": write_bw,
                "aggregate_mixed_total_bandwidth_gb_s": mixed_bw,
                "all_lpddr_oracle_ns": oracle,
                "storage_service_ns": storage,
                "lpddr_fill_service_ns": fill,
                "shared_lpddr_bus_lower_bound_ns": shared_bus,
                "conventional_candidates": candidates,
                "gap_falsification_penalty_fraction_bounds": interval[
                    "gap_falsification_penalty_fraction_bounds"
                ],
                "realizable_penalty_fraction_bounds": interval[
                    "realizable_penalty_fraction_bounds"
                ],
                "penalty_fraction_bounds": [gap_low, real_high],
            }
        )

    sufficient = float(thresholds["conventional_sufficient_max_penalty_fraction"])
    gap = float(thresholds["memory_gap_min_penalty_fraction"])
    if global_high <= sufficient:
        verdict = "CONVENTIONAL_MEMORY_SUFFICIENT"
    elif global_low >= gap:
        verdict = "H3_MEMORY_GAP_SUPPORTED"
    else:
        verdict = "INCONCLUSIVE_ESCALATE_INTEGRATED_TIMING"
    return {
        "verdict": verdict,
        "global_penalty_fraction_bounds": [global_low, global_high],
        "sensitivity_points": points,
        "thresholds": thresholds,
    }


def decide_point(
    residency_path: Path,
    lpddr_path: Path,
    fio_path: Path,
    contract_path: Path,
    queue_depth: int,
    out_path: Path,
    pagecache_paths: Iterable[Path] | None = None,
    actual_lpddr_path: Path | None = None,
) -> dict[str, Any]:
    residency = load_json(residency_path)
    lpddr = load_json(lpddr_path)
    fio = load_json(fio_path)
    contract = load_json(contract_path)
    actual_lpddr = load_json(actual_lpddr_path) if actual_lpddr_path is not None else None
    if residency.get("artifact_kind") != "fenix_h3_residency_replay":
        raise H3Error("unexpected residency artifact")
    if lpddr.get("artifact_kind") != "fenix_h3_lpddr_calibration":
        raise H3Error("unexpected LPDDR calibration artifact")
    if fio.get("artifact_kind") != "fenix_h3_fio_calibration":
        raise H3Error("unexpected fio calibration artifact")
    contract_sha = sha256_file(contract_path)
    for label, payload in (
        ("residency", residency),
        ("LPDDR calibration", lpddr),
        ("fio calibration", fio),
    ):
        require_contract_sha(payload, contract_sha, label)
    expected_ids = _expected_lpddr_ids(contract)
    observed_ids = {
        str(row.get("sensitivity_id")) for row in lpddr.get("sensitivity_points", [])
    }
    if observed_ids != expected_ids:
        raise H3Error(
            f"LPDDR sensitivity set mismatch: {sorted(observed_ids)} != {sorted(expected_ids)}"
        )
    miss = residency.get("miss_stream") or {}
    fio_provenance = fio.get("provenance") or {}
    if fio_provenance.get("miss_stream_sha256") != miss.get("sha256"):
        raise H3Error(
            "fio calibration was not derived from this residency replay miss stream"
        )
    paired_design = fio.get("paired_window_design") or {}
    if paired_design.get("verified") is not True or int(paired_design.get("distinct_nonoverlapping_windows", 0)) < 3:
        raise H3Error("fio calibration lacks at least three verified paired non-overlapping source windows")

    actual_points: list[dict[str, Any]] = []
    if actual_lpddr is not None:
        if actual_lpddr.get("artifact_kind") != "fenix_h3_lpddr_actual_trace_calibration":
            raise H3Error("unexpected actual-H3 LPDDR calibration artifact")
        require_contract_sha(actual_lpddr, contract_sha, "actual-H3 LPDDR calibration")
        if (actual_lpddr.get("provenance") or {}).get("residency_sha256") != sha256_file(residency_path):
            raise H3Error("actual-H3 LPDDR calibration was not derived from this residency replay")
        if str(actual_lpddr.get("stratum")) != str(residency.get("stratum")):
            raise H3Error("actual-H3 LPDDR calibration stratum differs from residency replay")
        if str(actual_lpddr.get("policy")) != str(residency.get("policy")):
            raise H3Error("actual-H3 LPDDR calibration policy differs from residency replay")
        if float(actual_lpddr.get("capacity_gib")) != float(residency.get("capacity_gib")):
            raise H3Error("actual-H3 LPDDR calibration capacity differs from residency replay")
        if actual_lpddr.get("phase_filter") != residency.get("phase_filter"):
            raise H3Error("actual-H3 LPDDR calibration phase differs from residency replay")
        generic_ram = lpddr.get("ramulator_tool") or {}
        actual_ram = actual_lpddr.get("ramulator_tool") or {}
        if (generic_ram.get("head"), generic_ram.get("tree")) != (actual_ram.get("head"), actual_ram.get("tree")):
            raise H3Error("generic and actual-H3 LPDDR calibrations use different Ramulator builds")
        actual_points = list(actual_lpddr.get("sensitivity_points") or [])
        mappings = [str(value) for value in contract["lpddr"].get("actual_trace_address_mappings", [])]
        if not mappings:
            raise H3Error("contract lacks actual-H3 LPDDR address-layout sensitivities")
        expected_actual_ids = {
            f"actual_h3_policy_trace_{mapping}_nCS{int(ncs)}"
            for mapping in mappings
            for ncs in contract["lpddr"]["ncs_values"]
        }
        observed_actual_ids = {str(row.get("sensitivity_id")) for row in actual_points}
        if observed_actual_ids != expected_actual_ids:
            raise H3Error(
                "actual-H3 LPDDR sensitivity set mismatch: "
                f"{sorted(observed_actual_ids)} != {sorted(expected_actual_ids)}"
            )
        observed_mappings = {str(row.get("address_mapping_strategy")) for row in actual_points}
        if observed_mappings != set(mappings):
            raise H3Error(
                f"actual-H3 LPDDR mapping set mismatch: {sorted(observed_mappings)} != {sorted(mappings)}"
            )
        for row in actual_points:
            mapping_strategy = str(row.get("address_mapping_strategy"))
            if mapping_strategy not in mappings:
                raise H3Error(f"unexpected actual-H3 LPDDR mapping strategy {mapping_strategy}")
            for key in (
                "aggregate_read_bandwidth_gb_s",
                "aggregate_write_bandwidth_gb_s",
                "aggregate_mixed_total_bandwidth_gb_s",
            ):
                if float(row.get(key, 0)) <= 0:
                    raise H3Error(f"actual-H3 LPDDR calibration has invalid {key}")
            for calibration_key in ("read_calibration", "write_calibration", "mixed_calibration"):
                mapping = (row.get(calibration_key) or {}).get("address_mapping_validation") or {}
                if mapping.get("modulo_address_folding") is not False or int(mapping.get("cross_object_aliases", -1)) != 0:
                    raise H3Error("actual-H3 LPDDR calibration is not collision-free/no-alias")

    qualified_pagecache: list[tuple[Path, dict[str, Any]]] = []
    for path in pagecache_paths or []:
        payload = load_json(path)
        if payload.get("artifact_kind") != "fenix_h3_pagecache_measurement":
            raise H3Error(f"unexpected page-cache artifact: {path}")
        require_contract_sha(payload, contract_sha, f"page-cache {path}")
        if (
            payload.get("pressure_observed") is not True
            or payload.get("cold_cache_qualified") is not True
            or payload.get("no_oom") is not True
        ):
            raise H3Error(f"page-cache control is not qualified: {path}")
        if str(payload.get("stratum")) != str(residency["stratum"]):
            raise H3Error(f"page-cache stratum differs from residency replay: {path}")
        if float(payload.get("capacity_gib")) != float(residency["capacity_gib"]):
            raise H3Error(f"page-cache capacity differs from residency replay: {path}")
        if residency.get("phase_filter") is not None:
            raise H3Error("page-cache candidate is defined only for full-stratum decisions")
        if payload.get("full_trace_replayed") is not True:
            raise H3Error(f"page-cache control does not replay the full stratum: {path}")
        residency_manifest_sha = (residency.get("provenance") or {}).get(
            "trace_manifest_sha256"
        )
        if payload.get("trace_manifest_sha256") != residency_manifest_sha:
            raise H3Error(f"page-cache trace manifest differs from residency replay: {path}")
        if int(payload.get("logical_read_bytes", -1)) != int(
            residency["counters"].get("lpddr_read_useful_bytes", -2)
        ):
            raise H3Error(f"page-cache logical endpoint differs from residency replay: {path}")
        if payload.get("storage_binding_sha256") != fio_provenance.get(
            "storage_binding_sha256"
        ):
            raise H3Error(f"page-cache storage device differs from fio calibration: {path}")
        pagecache_tool = payload.get("fio_tool") or {}
        fio_tool = fio.get("fio_tool") or {}
        if (
            pagecache_tool.get("head"),
            pagecache_tool.get("binary_sha256"),
        ) != (fio_tool.get("head"), fio_tool.get("binary_sha256")):
            raise H3Error(f"page-cache fio binary differs from direct-storage calibration: {path}")
        qualified_pagecache.append((path, payload))

    modes = [str(payload.get("mode")) for _, payload in qualified_pagecache]
    if len(modes) != len(set(modes)):
        raise H3Error("duplicate page-cache mode supplied to one H3 decision")

    provenance_inputs = [
        ("residency", residency),
        ("LPDDR calibration", lpddr),
        ("fio calibration", fio),
        *([("actual-H3 LPDDR calibration", actual_lpddr)] if actual_lpddr is not None else []),
        *[(f"page-cache {path}", payload) for path, payload in qualified_pagecache],
    ]
    execution_identity = require_same_execution_repository(*provenance_inputs)

    overall = _evaluate_service_counters(
        dict(residency["counters"]),
        lpddr,
        fio,
        contract,
        int(queue_depth),
        offline_profile=residency.get("offline_static_frequency_profile"),
        prefetch_sensitivity=residency.get("deterministic_ple_prefetch_sensitivity"),
        all_prefetch_sensitivity=residency.get("ple_and_expert_perfect_prefetch_sensitivity"),
        pagecache=qualified_pagecache,
        additional_lpddr_points=actual_points,
        capacity_bytes=int(residency.get("capacity_bytes", 0)),
        policy=str(residency.get("policy")),
    )

    phase_summaries: dict[str, Any] = {}
    if residency.get("phase_filter") is None:
        required_phases = list(contract["workload_policy"].get("phase_reports", []))
        for phase in required_phases:
            phase_counters = dict((residency.get("phase_counters") or {}).get(phase) or {})
            if float(phase_counters.get("lpddr_read_useful_bytes", 0)) <= 0:
                raise H3Error(
                    f"full-stratum residency lacks required {phase} conditional-memory demand"
                )
            phase_summaries[phase] = _evaluate_service_counters(
                phase_counters,
                lpddr,
                fio,
                contract,
                int(queue_depth),
                # Offline bound and page-cache controls are full-trace constructs;
                # phase reports use the measured policy counters plus prefetch sensitivity.
                prefetch_sensitivity={
                    "ple_prefetchable_storage_bytes": int(
                        phase_counters.get("ple_storage_bytes", 0)
                    ),
                    "nonprefetchable_expert_storage_bytes": int(
                        phase_counters.get("expert_storage_bytes", 0)
                    ),
                },
                all_prefetch_sensitivity={
                    "prefetchable_storage_bytes": int(phase_counters.get("storage_bytes", 0))
                },
                additional_lpddr_points=actual_points,
            )

    fio_tool = fio.get("fio_tool") or {}
    ram_tool = lpddr.get("ramulator_tool") or {}
    campaign_fingerprint = {
        "contract_sha256": contract_sha,
        "trace_manifest_sha256": (residency.get("provenance") or {}).get("trace_manifest_sha256"),
        "storage_binding_sha256": fio_provenance.get("storage_binding_sha256"),
        "device_major_minor": fio_provenance.get("device_major_minor"),
        "fio_head": fio_tool.get("head"),
        "fio_tree": fio_tool.get("tree"),
        "fio_binary_sha256": fio_tool.get("binary_sha256"),
        "ramulator_head": ram_tool.get("head"),
        "ramulator_tree": ram_tool.get("tree"),
    }
    if any(not campaign_fingerprint.get(key) for key in campaign_fingerprint):
        raise H3Error(f"decision campaign fingerprint is incomplete: {campaign_fingerprint}")

    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_same_endpoint_decision",
        "scope": "conditional_memory_service_layer",
        "not_end_to_end_inference_latency": True,
        "stratum": residency["stratum"],
        "phase_filter": residency.get("phase_filter"),
        "policy": residency["policy"],
        "capacity_gib": residency["capacity_gib"],
        "queue_depth": int(queue_depth),
        "promotion_eligible_queue_depth": int(queue_depth) in set(
            int(value) for value in contract["storage"].get("promotion_queue_depths", [])
        ),
        "actual_lpddr_trace_calibration_used": actual_lpddr is not None,
        "actual_lpddr_address_mapping_strategies": (
            sorted({str(row.get("address_mapping_strategy")) for row in actual_points})
            if actual_points else []
        ),
        "pagecache_baseline_used": bool(qualified_pagecache),
        "pagecache_modes": sorted(
            str(payload["mode"]) for _, payload in qualified_pagecache
        ),
        "verdict": overall["verdict"],
        "global_penalty_fraction_bounds": overall["global_penalty_fraction_bounds"],
        "thresholds": overall["thresholds"],
        "sensitivity_points": overall["sensitivity_points"],
        "phase_summaries": phase_summaries,
        "phase_reporting_method": (
            "embedded phase summaries are projections only; paper promotion requires separate phase-filtered residency, fio, and actual-H3 Ramulator decision artifacts"
        ),
        "contract_sha256": contract_sha,
        "execution_repository_consistency_verified": True,
        "campaign_fingerprint": campaign_fingerprint,
        "claim_boundary": (
            "A supported H3 memory-service gap motivates integrated timing and a candidate intermediate tier; "
            "it is not a measured end-to-end speedup and does not establish FeRAM superiority."
        ),
        "provenance": {
            "residency_sha256": sha256_file(residency_path),
            "lpddr_sha256": sha256_file(lpddr_path),
            "fio_sha256": sha256_file(fio_path),
            "actual_lpddr_sha256": sha256_file(actual_lpddr_path) if actual_lpddr_path is not None else None,
            "contract_sha256": contract_sha,
            "execution_repository": execution_identity,
        },
    }
    write_json(out_path, result)
    return result


def _resolve_bound_artifact(
    manifest_path: Path,
    entry: dict[str, Any],
    expected_kind: str,
    contract_sha: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    raw = entry.get("artifact")
    expected_sha = entry.get("sha256")
    if not raw or not expected_sha:
        raise H3Error(f"prerequisite {label} is not bound to an artifact and SHA256")
    path = Path(str(raw))
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if not path.is_file():
        raise H3Error(f"prerequisite {label} artifact does not exist: {path}")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise H3Error(f"prerequisite {label} SHA mismatch")
    payload = load_json(path)
    if payload.get("artifact_kind") != expected_kind:
        raise H3Error(
            f"prerequisite {label} kind mismatch: {payload.get('artifact_kind')} != {expected_kind}"
        )
    require_contract_sha(payload, contract_sha, f"prerequisite {label}")
    return path, payload


def _validate_prerequisites(
    prerequisites_path: Path,
    contract: dict[str, Any],
    contract_sha: str,
    decision_identity: dict[str, Any],
    campaign_fingerprint: dict[str, Any],
    capacity_gib: float,
    pagecache_complete: bool,
    preliminary_verdict: str,
) -> dict[str, Any]:
    manifest = load_json(prerequisites_path)
    expected_manifest_kind = contract["prerequisites"]["manifest_artifact_kind"]
    if manifest.get("artifact_kind") != expected_manifest_kind:
        raise H3Error("unexpected H3 prerequisite manifest kind")
    evidence = manifest.get("evidence") or {}
    resolved: dict[str, Any] = {}
    labeled_payloads: list[tuple[str, dict[str, Any]]] = []
    for key, spec in contract["prerequisites"]["required_evidence"].items():
        entry = evidence.get(key)
        if not isinstance(entry, dict):
            raise H3Error(f"prerequisite manifest lacks evidence entry {key}")
        path, payload = _resolve_bound_artifact(
            prerequisites_path, entry, str(spec["artifact_kind"]), contract_sha, key
        )
        if key == "placement_invariance":
            if payload.get("derived_pass") is not True or payload.get("evaluator") != "ordered_routing_and_ple_semantic_identity_across_physical_placements":
                raise H3Error("placement-invariance prerequisite was not derived by the v6 ordered-semantic evaluator")
        elif key == "long_context_beyond_8k":
            if payload.get("derived_pass") is not True or payload.get("evaluator") != "trace_validated_multi_length_context_scaling_characterization":
                raise H3Error("long-context prerequisite was not derived by the v6 multi-length evaluator")
        elif key == "deployment_capacity_budget":
            if payload.get("derived") is not True or payload.get("platform_mode") != "primary":
                raise H3Error("deployment capacity budget is not a derived primary-platform artifact")
            if float(payload.get("physical_memory_gib", -1)) != float(contract["lpddr"]["platform_capacity_gib"]):
                raise H3Error("deployment capacity budget physical memory differs from H3 contract")
            candidates = {float(value) for value in payload.get("candidate_budgets_that_fit_gib", [])}
            if float(capacity_gib) not in candidates:
                raise H3Error(f"decision capacity {capacity_gib} GiB is not supported by deployment budget evidence")
        elif key == "tool_qualification":
            if payload.get("complete") is not True or payload.get("failures"):
                raise H3Error("tool qualification prerequisite is incomplete")
            observed = payload.get("observed") or {}
            env = observed.get("ramulator_python_environment") or {}
            relative_lock = env.get("resolved_lock_relative_path")
            lock_path = (
                (path.parent / str(relative_lock)).resolve()
                if relative_lock else Path(str(env.get("resolved_lock_path", "")))
            )
            if not lock_path.is_file() or sha256_file(lock_path) != env.get("resolved_lock_sha256"):
                raise H3Error("tool qualification resolved Ramulator environment lock is missing or changed")
            if not observed.get("ramulator_built_extensions") or not observed.get("build_toolchain"):
                raise H3Error("tool qualification lacks Ramulator build identity")
            fio_binary = observed.get("fio_binary") or {}
            ramulator = observed.get("ramulator2") or {}
            if (fio_binary.get("head"), fio_binary.get("tree"), fio_binary.get("binary_sha256")) != (
                campaign_fingerprint.get("fio_head"), campaign_fingerprint.get("fio_tree"), campaign_fingerprint.get("fio_binary_sha256")
            ):
                raise H3Error("tool qualification fio identity differs from H3 decision campaign")
            if (ramulator.get("head"), ramulator.get("tree")) != (
                campaign_fingerprint.get("ramulator_head"), campaign_fingerprint.get("ramulator_tree")
            ):
                raise H3Error("tool qualification Ramulator identity differs from H3 decision campaign")
        resolved[key] = {
            "artifact": str(path), "sha256": sha256_file(path), "artifact_kind": payload.get("artifact_kind")
        }
        labeled_payloads.append((f"prerequisite {key}", payload))

    concurrency_status: dict[str, Any] = {"required_for_conventional_sufficiency": True, "applied": False}
    if preliminary_verdict == "CONVENTIONAL_MEMORY_SUFFICIENT":
        spec = contract["prerequisites"].get("conditional_evidence", {}).get("storage_concurrency_feasibility")
        entry = evidence.get("storage_concurrency_feasibility")
        if not isinstance(spec, dict) or not isinstance(entry, dict):
            raise H3Error("QD32 conventional sufficiency requires workload-derived storage-concurrency evidence")
        path, payload = _resolve_bound_artifact(
            prerequisites_path, entry, str(spec["artifact_kind"]), contract_sha, "storage_concurrency_feasibility"
        )
        if payload.get("derived_pass") is not True or payload.get("evaluator") != "measured_workload_scheduler_raw_event_qd_and_prefetch_slack_feasibility":
            raise H3Error("storage-concurrency evidence does not establish realizable QD32 with measured prefetch slack")
        if payload.get("trace_manifest_sha256") != campaign_fingerprint.get("trace_manifest_sha256"):
            raise H3Error("storage-concurrency evidence uses a different source trace manifest")
        if payload.get("storage_binding_sha256") != campaign_fingerprint.get("storage_binding_sha256"):
            raise H3Error("storage-concurrency evidence uses a different storage binding")
        resolved["storage_concurrency_feasibility"] = {
            "artifact": str(path), "sha256": sha256_file(path), "artifact_kind": payload.get("artifact_kind")
        }
        labeled_payloads.append(("prerequisite storage_concurrency_feasibility", payload))
        concurrency_status = {"required_for_conventional_sufficiency": True, "applied": True, "passed": True}

    prereq_identity = require_same_execution_repository(*labeled_payloads)
    keys = ("head", "tree", "required_base_commit")
    if tuple(prereq_identity.get(k) for k in keys) != tuple(decision_identity.get(k) for k in keys):
        raise H3Error("prerequisite evidence was generated by a different H3 implementation commit")
    if not pagecache_complete:
        raise H3Error("page-cache prerequisite is derived from the decision matrix and is incomplete")
    return {
        "evidence": resolved,
        "pagecache_control": {"derived_from_decision_matrix": True, "passed": True},
        "storage_concurrency": concurrency_status,
    }


def campaign_gate(
    decision_paths: Iterable[Path],
    prerequisites_path: Path,
    contract_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    paths = list(decision_paths)
    contract = load_json(contract_path)
    contract_sha = sha256_file(contract_path)
    decisions = [load_json(path) for path in paths]
    if not decisions:
        raise H3Error("campaign gate requires at least one H3 decision")
    for row in decisions:
        if row.get("artifact_kind") != "fenix_h3_same_endpoint_decision":
            raise H3Error("campaign gate received a non-H3 decision artifact")
        require_contract_sha(row, contract_sha, "H3 decision")
        for key in ("stratum", "verdict", "capacity_gib", "policy", "queue_depth", "phase_filter"):
            if key not in row:
                raise H3Error(f"H3 decision is missing required field {key}")
    execution_identity = require_same_execution_repository(
        *[(f"decision {path}", row) for path, row in zip(paths, decisions, strict=True)]
    )

    promotion_qds = {int(value) for value in contract["storage"]["promotion_queue_depths"]}
    promotion_rows = [
        row for row in decisions
        if int(row["queue_depth"]) in promotion_qds and row.get("promotion_eligible_queue_depth") is True
    ]
    if not promotion_rows:
        raise H3Error("campaign gate has no promotion-eligible QD decision rows")
    if any(row.get("actual_lpddr_trace_calibration_used") is not True for row in promotion_rows):
        raise H3Error("paper promotion requires actual-H3 LPDDR trace calibration on every full and phase row")

    fingerprints = [row.get("campaign_fingerprint") for row in promotion_rows]
    if any(not isinstance(value, dict) for value in fingerprints):
        raise H3Error("promotion decision lacks campaign_fingerprint")
    reference_fingerprint = dict(fingerprints[0])
    if any(dict(value) != reference_fingerprint for value in fingerprints[1:]):
        raise H3Error("promotion decisions mix source/storage/tool campaign fingerprints")

    required = contract["decision_gate"]["campaign_requirements"]
    required_names = list(dict.fromkeys(
        [str(value) for value in required["required_strata"]]
        + [str(value) for value in required.get("required_ordinary_strata", [])]
    ))
    expected_policies = sorted(set(contract.get("primary_policies", [])) | set(contract.get("strong_conventional_policies", [])))
    required_phases = [str(value) for value in required.get("required_phases", [])]

    def rows_for(phase: str | None) -> list[dict[str, Any]]:
        return [row for row in promotion_rows if row.get("phase_filter") == phase]

    full_rows = rows_for(None)
    phase_rows = {phase: rows_for(phase) for phase in required_phases}

    matrix_errors: list[str] = []
    selected_full: list[dict[str, Any]] = []
    selected_phase: list[dict[str, Any]] = []
    for name in required_names:
        for policy in expected_policies:
            for qd in sorted(promotion_qds):
                matches = [
                    row for row in full_rows
                    if str(row["stratum"]) == name and str(row["policy"]) == policy and int(row["queue_depth"]) == qd
                ]
                if len(matches) != 1:
                    matrix_errors.append(f"full:{name}:{policy}:qd{qd}:count={len(matches)}")
                else:
                    selected_full.append(matches[0])
                for phase in required_phases:
                    pmatches = [
                        row for row in phase_rows[phase]
                        if str(row["stratum"]) == name and str(row["policy"]) == policy and int(row["queue_depth"]) == qd
                    ]
                    if len(pmatches) != 1:
                        matrix_errors.append(f"{phase}:{name}:{policy}:qd{qd}:count={len(pmatches)}")
                    else:
                        selected_phase.append(pmatches[0])

    capacities = {float(row["capacity_gib"]) for row in [*selected_full, *selected_phase]}
    if len(capacities) != 1:
        matrix_errors.append(f"promotion rows do not use one common capacity: {sorted(capacities)}")

    pagecache_missing: list[str] = []
    required_pagecache = set(contract.get("pagecache", {}).get("primary_strata", []))
    required_modes = set(contract.get("pagecache", {}).get("required_modes", []))
    for name in sorted(required_pagecache):
        rows = [row for row in selected_full if str(row["stratum"]) == name]
        for policy in expected_policies:
            policy_rows = [row for row in rows if str(row["policy"]) == policy]
            if not policy_rows or not all(
                row.get("pagecache_baseline_used") is True
                and required_modes.issubset(set(row.get("pagecache_modes", [])))
                for row in policy_rows
            ):
                pagecache_missing.append(f"{name}:{policy}")

    if matrix_errors:
        preliminary = "INCOMPLETE_OR_DUPLICATE_H3_MATRIX"
        combined: dict[str, list[float]] = {}
    elif pagecache_missing:
        preliminary = "INCOMPLETE_PAGECACHE_BASELINE"
        combined = {}
    else:
        sufficient = float(contract["decision_gate"]["conventional_sufficient_max_penalty_fraction"])
        gap = float(contract["decision_gate"]["memory_gap_min_penalty_fraction"])
        combined = {}
        gap_all = True
        sufficient_all = True
        for name in required_names:
            scoped_rows = [row for row in [*selected_full, *selected_phase] if str(row["stratum"]) == name]
            gap_low = min(float(row["global_penalty_fraction_bounds"][0]) for row in scoped_rows)
            policy_worst_high: dict[str, float] = {}
            for policy in expected_policies:
                policy_rows = [row for row in scoped_rows if str(row["policy"]) == policy]
                policy_worst_high[policy] = max(float(row["global_penalty_fraction_bounds"][1]) for row in policy_rows)
            realizable_high = min(policy_worst_high.values())
            combined[name] = [gap_low, realizable_high]
            gap_all = gap_all and gap_low >= gap
            sufficient_all = sufficient_all and realizable_high <= sufficient
        if sufficient_all:
            preliminary = "CONVENTIONAL_MEMORY_SUFFICIENT"
        elif gap_all:
            preliminary = "H3_MEMORY_GAP_SUPPORTED"
        else:
            preliminary = "INCONCLUSIVE_ESCALATE_INTEGRATED_TIMING"

    prerequisite_status: dict[str, Any] | None = None
    prerequisite_error: str | None = None
    if not matrix_errors and not pagecache_missing and len(capacities) == 1:
        try:
            prerequisite_status = _validate_prerequisites(
                prerequisites_path,
                contract,
                contract_sha,
                execution_identity,
                reference_fingerprint,
                next(iter(capacities)),
                pagecache_complete=True,
                preliminary_verdict=preliminary,
            )
        except H3Error as exc:
            prerequisite_error = str(exc)

    directional = preliminary in {"H3_MEMORY_GAP_SUPPORTED", "CONVENTIONAL_MEMORY_SUFFICIENT"}
    if directional and prerequisite_status is not None:
        if preliminary == "H3_MEMORY_GAP_SUPPORTED":
            promotion = "H3_PAPER_GATE_SUPPORTED"
            proceed = True
        else:
            promotion = "H3_CONVENTIONAL_SUFFICIENT_STOP_H4"
            proceed = False
    else:
        promotion = "H3_PAPER_GATE_BLOCKED"
        proceed = False

    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_campaign_gate",
        "preliminary_memory_service_verdict": preliminary,
        "paper_promotion_verdict": promotion,
        "proceed_to_h4": proceed,
        "promotion_queue_depths": sorted(promotion_qds),
        "diagnostic_queue_depths_do_not_promote": list(contract["storage"].get("diagnostic_queue_depths", [])),
        "qd32_sufficiency_requires_measured_scheduler_feasibility": True,
        "actual_lpddr_trace_required_and_verified": all(
            row.get("actual_lpddr_trace_calibration_used") is True for row in promotion_rows
        ),
        "explicit_phase_calibration_required": True,
        "required_strata": required_names,
        "required_policies": expected_policies,
        "required_phases": required_phases,
        "matrix_errors": matrix_errors,
        "missing_pagecache_baselines": pagecache_missing,
        "decision_capacity_gib": next(iter(capacities)) if len(capacities) == 1 else None,
        "combined_directional_bounds": combined,
        "combined_directional_scope": "full_plus_independent_prefill_decode; gap requires every policy/scope lower bound >= threshold; sufficiency requires one fixed policy per stratum whose worst full/prefill/decode upper bound <= threshold",
        "campaign_fingerprint": reference_fingerprint,
        "campaign_fingerprint_consistency_verified": not matrix_errors,
        "prerequisite_status": prerequisite_status,
        "prerequisite_error": prerequisite_error,
        "decision_count": len(decisions),
        "promotion_full_decision_count": len(selected_full),
        "promotion_phase_decision_count": len(selected_phase),
        "decision_sha256": {str(path): sha256_file(path) for path in paths},
        "prerequisites_sha256": sha256_file(prerequisites_path),
        "contract_sha256": contract_sha,
        "execution_repository_consistency_verified": True,
        "execution_repository": execution_identity,
    }
    write_json(out_path, result)
    return result

