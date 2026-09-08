"""H3 decision amendment constrained by measured expert-prefetch causality.

For the pinned host-driven runtime, an exact cold-expert read cannot start until
router-selected expert IDs are available.  When the causal experiment proves
that the native ID-ready-to-dispatch window cannot hide even the fastest
measured one-expert transfer, unavoidable expert storage service is on the
critical path before native expert dispatch.  This module therefore causalizes
every conventional lower bound that still contains expert misses; PLE perfect
prefetch remains an intentionally hostile sensitivity.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from analysis.h3.common import H3Error, load_json, sha256_file, write_json
from analysis.h3.decision import (
    _candidate_interval_summary,
    _service_ns,
    _service_time_optimized_offline_bound,
    decide_point,
)
from analysis.h3.storage import project_storage_bytes_ns


REMOVED_BASELINE = "PLE_and_expert_perfect_prefetch_sensitivity"
OFFLINE_BASELINE = "service_time_optimized_offline_frequency_bound"
PLE_PREFETCH_BASELINE = "deterministic_PLE_perfect_prefetch_sensitivity"
EXPLICIT_BASELINE = "explicit_trace_cache_plus_direct_storage"
CAUSAL_BELADY_BASELINE = "causal_belady_pagecache_plus_measured_storage_bandwidth"


def _load_amendment(path: Path, contract_path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("artifact_kind") != "fenix_h3_analysis_amendment_contract":
        raise H3Error("unexpected causality amendment")
    if payload.get("base_contract_sha256") != sha256_file(contract_path):
        raise H3Error("causality amendment is not bound to H3 contract")
    cfg = payload.get("causality") or {}
    if cfg.get("expert_storage_serialized_before_native_dispatch") is not True:
        raise H3Error("causality amendment does not declare serialized expert service")
    if cfg.get("ple_perfect_prefetch_retained") is not True:
        raise H3Error("causality amendment must retain PLE perfect-prefetch sensitivity")
    return payload


def _validate_causality(
    path: Path,
    amendment_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("artifact_kind") != "fenix_h3_expert_prefetch_causality":
        raise H3Error("unexpected expert-prefetch causality artifact")
    if payload.get("derived_pass") is not True or payload.get("failures"):
        raise H3Error("expert-prefetch causality gate did not pass")
    if payload.get("contract_sha256") != sha256_file(contract_path):
        raise H3Error("causality artifact uses another H3 contract")
    if payload.get("analysis_amendment_sha256") != sha256_file(amendment_path):
        raise H3Error("causality artifact uses another analysis amendment")
    if payload.get("all_expert_perfect_prefetch_falsified_for_pinned_host_driven_runtime") is not True:
        raise H3Error("causality artifact does not falsify all-expert perfect prefetch")
    if payload.get("ple_perfect_prefetch_remains_allowed") is not True:
        raise H3Error("causality artifact incorrectly removes PLE perfect prefetch")
    return payload


def _validate_pagecache_oracle(
    path: Path,
    residency: dict[str, Any],
    frozen_head: str,
) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("artifact_kind") != "fenix_h3_pagecache_oracle":
        raise H3Error("unexpected page-cache oracle artifact")
    if str(payload.get("stratum")) != str(residency.get("stratum")):
        raise H3Error("page-cache oracle stratum differs from residency")
    if float(payload.get("capacity_gib", -1)) != float(residency.get("capacity_gib", -2)):
        raise H3Error("page-cache oracle capacity differs from residency")
    source = payload.get("source_measurement_execution_repository") or {}
    if source.get("head") != frozen_head:
        raise H3Error("page-cache oracle is not bound to frozen measurement campaign")
    for key in (
        "future_knowledge",
        "ple_storage_traffic_free",
        "expert_partial_tail_free",
        "all_capacity_dedicated_to_expert_pages",
    ):
        if payload.get(key) is not True:
            raise H3Error(f"page-cache oracle lacks hostile property {key}")
    if int(payload.get("lower_tier_read_bytes", -1)) < 0:
        raise H3Error("page-cache oracle lower-tier bytes are invalid")
    return payload


def _causalize_candidate_lower(
    candidate: dict[str, Any],
    *,
    all_lpddr_ns: float,
    expert_storage_bytes: float,
    fio: dict[str, Any],
    queue_depth: int,
) -> None:
    """Add the fastest measured unavoidable expert service before dispatch.

    The original lower endpoint is retained if it is stronger.  The expert-only
    storage projection deliberately grants PLE perfect prefetch and uses the
    same optimistic QD requested by the decision.  Adding it to all-resident
    LPDDR service follows the measured source dependency: exact expert IDs ->
    required cold-expert service -> native quant_method.apply dispatch.
    """
    if expert_storage_bytes < 0:
        raise H3Error("negative expert storage bytes in causal decision")
    expert_service = project_storage_bytes_ns(
        0.0,
        float(expert_storage_bytes),
        fio,
        int(queue_depth),
    )
    old = candidate.get("service_ns_bounds")
    if not isinstance(old, list) or len(old) != 2:
        raise H3Error(f"candidate lacks service bounds: {candidate.get('baseline')}")
    causal_low = max(float(old[0]), float(all_lpddr_ns) + float(expert_service["low_ns"]))
    candidate["pre_causal_service_ns_bounds"] = list(old)
    candidate["service_ns_bounds"] = [causal_low, max(causal_low, float(old[1]))]
    candidate["causal_expert_storage_bytes"] = float(expert_storage_bytes)
    candidate["causal_expert_storage_service_ns"] = expert_service
    candidate["expert_storage_serialized_before_native_dispatch"] = True
    candidate["causal_lower_bound_semantics"] = (
        "max(original lower, all-resident LPDDR service + fastest measured "
        "QD expert-storage service); PLE storage latency may remain perfectly hidden"
    )


def decide_with_causal_prefetch(
    residency_path: Path,
    lpddr_path: Path,
    fio_path: Path,
    actual_lpddr_path: Path,
    causality_path: Path,
    contract_path: Path,
    amendment_path: Path,
    queue_depth: int,
    out_path: Path,
    pagecache_oracle_path: Path | None = None,
) -> dict[str, Any]:
    amendment = _load_amendment(amendment_path, contract_path)
    causality = _validate_causality(causality_path, amendment_path, contract_path)
    frozen_head = str(amendment["frozen_measurement_head"])

    with tempfile.TemporaryDirectory(prefix="fenix-h3-causal-decision-") as tmp:
        base_path = Path(tmp) / "base.json"
        decide_point(
            residency_path,
            lpddr_path,
            fio_path,
            contract_path,
            int(queue_depth),
            base_path,
            None,
            actual_lpddr_path,
        )
        result = load_json(base_path)

    measurement_identity = (result.get("provenance") or {}).get("execution_repository") or {}
    if measurement_identity.get("head") != frozen_head:
        raise H3Error("decision inputs are not from frozen H3 measurement campaign")

    residency = load_json(residency_path)
    fio = load_json(fio_path)
    phase = residency.get("phase_filter")
    counters = residency.get("counters") or {}
    baseline_expert_storage_bytes = float(counters.get("expert_storage_bytes", 0))
    special = set(str(v) for v in amendment["causality"]["special_full_strata"])
    needs_oracle = phase is None and str(residency.get("stratum")) in special
    if needs_oracle and pagecache_oracle_path is None:
        raise H3Error("special full-stratum causal decision requires Belady page-cache oracle")
    if phase is not None and pagecache_oracle_path is not None:
        raise H3Error("page-cache oracle is full-stratum only")

    oracle = None
    if pagecache_oracle_path is not None:
        oracle = _validate_pagecache_oracle(pagecache_oracle_path, residency, frozen_head)

    read_bytes = float(counters.get("lpddr_read_useful_bytes", 0))
    removed = 0
    causalized = 0
    for point in result["sensitivity_points"]:
        all_lpddr = float(point["all_lpddr_oracle_ns"])
        write_bw = float(point["aggregate_write_bandwidth_gb_s"])
        before = list(point["conventional_candidates"])
        kept = [row for row in before if row.get("baseline") != REMOVED_BASELINE]
        removed += len(before) - len(kept)
        point["conventional_candidates"] = kept

        by_name = {str(row.get("baseline")): row for row in kept}
        explicit = by_name.get(EXPLICIT_BASELINE)
        ple_prefetch = by_name.get(PLE_PREFETCH_BASELINE)
        offline = by_name.get(OFFLINE_BASELINE)
        if explicit is None or ple_prefetch is None:
            raise H3Error("causal decision lacks required explicit/PLE-prefetch candidates")

        _causalize_candidate_lower(
            explicit,
            all_lpddr_ns=all_lpddr,
            expert_storage_bytes=baseline_expert_storage_bytes,
            fio=fio,
            queue_depth=int(queue_depth),
        )
        causalized += 1
        _causalize_candidate_lower(
            ple_prefetch,
            all_lpddr_ns=all_lpddr,
            expert_storage_bytes=baseline_expert_storage_bytes,
            fio=fio,
            queue_depth=int(queue_depth),
        )
        causalized += 1

        if offline is not None:
            profile = residency.get("offline_static_frequency_profile")
            if not isinstance(profile, dict):
                raise H3Error("offline candidate exists without offline profile")
            bound = _service_time_optimized_offline_bound(
                profile,
                fio,
                int(queue_depth),
                int(residency.get("capacity_bytes", 0)),
                write_bw,
                str(residency.get("policy")),
                int(counters.get("ple_storage_bytes", 0)),
                int(counters.get("expert_storage_bytes", 0)),
            )
            offline["causal_offline_bound"] = bound
            _causalize_candidate_lower(
                offline,
                all_lpddr_ns=all_lpddr,
                expert_storage_bytes=float(bound.get("expert_storage_bytes", 0)),
                fio=fio,
                queue_depth=int(queue_depth),
            )
            causalized += 1

        if oracle is not None:
            miss_bytes = float(oracle["lower_tier_read_bytes"])
            storage = project_storage_bytes_ns(0.0, miss_bytes, fio, int(queue_depth))
            bandwidths = [
                float(point["aggregate_read_bandwidth_gb_s"]),
                float(point["aggregate_write_bandwidth_gb_s"]),
                float(point["aggregate_mixed_total_bandwidth_gb_s"]),
            ]
            if any(value <= 0 for value in bandwidths):
                raise H3Error("causal page-cache oracle LPDDR bandwidth is invalid")
            optimistic_bw = max(bandwidths)
            fill = _service_ns(miss_bytes, optimistic_bw)
            shared_bus = _service_ns(read_bytes + miss_bytes, optimistic_bw)
            # Belady replacement and LPDDR bandwidth remain unrealistically
            # favorable, but the unavoidable expert service is causally before
            # the native expert dispatch and cannot be hidden behind that dispatch.
            lower = max(all_lpddr + float(storage["low_ns"]), shared_bus)
            upper = max(lower, all_lpddr + fill + float(storage["high_ns"]))
            point["conventional_candidates"].append({
                "baseline": CAUSAL_BELADY_BASELINE,
                "promotion_role": "gap_falsification_only",
                "role": "future_aware_cache_with_finite_causally_serialized_expert_storage",
                "future_knowledge": True,
                "ple_storage_traffic_free": True,
                "expert_partial_tail_free": True,
                "storage_bandwidth_finite_and_measured": True,
                "expert_storage_serialized_before_native_dispatch": True,
                "storage_read_bytes": miss_bytes,
                "storage_service_ns": storage,
                "oracle_lpddr_bandwidth_gb_s": optimistic_bw,
                "lpddr_fill_service_ns": fill,
                "shared_lpddr_bus_lower_bound_ns": shared_bus,
                "service_ns_bounds": [lower, upper],
                "pagecache_oracle_sha256": sha256_file(pagecache_oracle_path),
            })
            causalized += 1

        interval = _candidate_interval_summary(
            point["conventional_candidates"], all_lpddr
        )
        point["gap_falsification_penalty_fraction_bounds"] = interval[
            "gap_falsification_penalty_fraction_bounds"
        ]
        point["realizable_penalty_fraction_bounds"] = interval[
            "realizable_penalty_fraction_bounds"
        ]
        point["penalty_fraction_bounds"] = [
            interval["gap_falsification_penalty_fraction_bounds"][0],
            interval["realizable_penalty_fraction_bounds"][1],
        ]

    if removed != len(result["sensitivity_points"]):
        raise H3Error(
            "expected exactly one all-expert perfect-prefetch candidate per sensitivity point"
        )

    global_low = min(
        float(point["gap_falsification_penalty_fraction_bounds"][0])
        for point in result["sensitivity_points"]
    )
    global_high = max(
        float(point["realizable_penalty_fraction_bounds"][1])
        for point in result["sensitivity_points"]
    )
    gap = float(result["thresholds"]["memory_gap_min_penalty_fraction"])
    sufficient = float(result["thresholds"]["conventional_sufficient_max_penalty_fraction"])
    if global_low >= gap:
        verdict = "H3_MEMORY_GAP_SUPPORTED"
    elif global_high <= sufficient:
        verdict = "INCONCLUSIVE_MEASURED_PAGECACHE_REQUIRED_FOR_SUFFICIENCY"
    else:
        verdict = "INCONCLUSIVE_ESCALATE_INTEGRATED_TIMING"

    result["global_penalty_fraction_bounds"] = [global_low, global_high]
    result["verdict"] = verdict
    result["causal_expert_prefetch_evidence_used"] = True
    result["causality_artifact_sha256"] = sha256_file(causality_path)
    result["causality_amendment_sha256"] = sha256_file(amendment_path)
    result["removed_unbounded_sensitivity_baseline"] = REMOVED_BASELINE
    result["expert_storage_lower_bounds_causalized"] = True
    result["causalized_candidate_instances"] = causalized
    result["ple_perfect_prefetch_sensitivity_retained"] = True
    result["causal_belady_pagecache_used"] = oracle is not None
    result["causal_belady_pagecache_baseline"] = CAUSAL_BELADY_BASELINE if oracle is not None else None
    result["measurement_execution_repository"] = measurement_identity
    result["causality_summary"] = {
        "maximum_native_prefetch_slack_ns": causality["maximum_native_prefetch_slack_ns"],
        "fastest_measured_one_expert_transfer_ns_lower": causality[
            "fastest_measured_one_expert_transfer_ns_lower"
        ],
        "slack_over_fastest_one_expert_service": causality[
            "slack_over_fastest_one_expert_service"
        ],
    }
    result["provenance"]["causality_sha256"] = sha256_file(causality_path)
    result["provenance"]["causality_amendment_sha256"] = sha256_file(amendment_path)
    if pagecache_oracle_path is not None:
        result["provenance"]["pagecache_oracle_sha256"] = sha256_file(pagecache_oracle_path)
    result["claim_boundary"] = amendment["claim_boundary"]
    write_json(out_path, result)
    return result
