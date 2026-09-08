"""H3 decision amendment constrained by measured expert-prefetch causality."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from analysis.h3.common import H3Error, load_json, sha256_file, write_json
from analysis.h3.decision import _candidate_interval_summary, _service_ns, decide_point
from analysis.h3.storage import project_storage_bytes_ns


REMOVED_BASELINE = "PLE_and_expert_perfect_prefetch_sensitivity"
CAUSAL_BELADY_BASELINE = "causal_belady_pagecache_plus_measured_storage_bandwidth"


def _load_amendment(path: Path, contract_path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("artifact_kind") != "fenix_h3_analysis_amendment_contract":
        raise H3Error("unexpected causality amendment")
    if payload.get("base_contract_sha256") != sha256_file(contract_path):
        raise H3Error("causality amendment is not bound to H3 contract")
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
    special = set(str(v) for v in amendment["causality"]["special_full_strata"])
    needs_oracle = phase is None and str(residency.get("stratum")) in special
    if needs_oracle and pagecache_oracle_path is None:
        raise H3Error("special full-stratum causal decision requires Belady page-cache oracle")
    if phase is not None and pagecache_oracle_path is not None:
        raise H3Error("page-cache oracle is full-stratum only")

    oracle = None
    if pagecache_oracle_path is not None:
        oracle = _validate_pagecache_oracle(pagecache_oracle_path, residency, frozen_head)

    read_bytes = float(residency["counters"]["lpddr_read_useful_bytes"])
    removed = 0
    for point in result["sensitivity_points"]:
        before = list(point["conventional_candidates"])
        kept = [row for row in before if row.get("baseline") != REMOVED_BASELINE]
        removed += len(before) - len(kept)
        point["conventional_candidates"] = kept

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
            all_lpddr = float(point["all_lpddr_oracle_ns"])
            fill = _service_ns(miss_bytes, optimistic_bw)
            shared_bus = _service_ns(read_bytes + miss_bytes, optimistic_bw)
            lower = max(all_lpddr, shared_bus, float(storage["low_ns"]))
            # Storage and LPDDR are granted perfect overlap. The upper endpoint
            # therefore uses the slower of the two finite resources rather than
            # summing them; this remains deliberately favorable to conventional memory.
            upper = max(all_lpddr + fill, shared_bus, float(storage["high_ns"]))
            point["conventional_candidates"].append({
                "baseline": CAUSAL_BELADY_BASELINE,
                "promotion_role": "gap_falsification_only",
                "role": "future_aware_cache_plus_finite_measured_storage_with_perfect_resource_overlap",
                "future_knowledge": True,
                "ple_storage_traffic_free": True,
                "expert_partial_tail_free": True,
                "storage_latency_overlap": "perfect",
                "storage_bandwidth_finite_and_measured": True,
                "storage_read_bytes": miss_bytes,
                "storage_service_ns": storage,
                "oracle_lpddr_bandwidth_gb_s": optimistic_bw,
                "lpddr_fill_service_ns": fill,
                "shared_lpddr_bus_lower_bound_ns": shared_bus,
                "service_ns_bounds": [lower, upper],
                "pagecache_oracle_sha256": sha256_file(pagecache_oracle_path),
            })

        interval = _candidate_interval_summary(
            point["conventional_candidates"], float(point["all_lpddr_oracle_ns"])
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
