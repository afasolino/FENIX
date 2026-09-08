"""Analysis-layer H3 decision/gate amendment for the hostile page-cache oracle."""
from __future__ import annotations

import tempfile
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
from analysis.h3.decision import (
    _candidate_interval_summary,
    _service_ns,
    _validate_prerequisites,
    decide_point,
)
from analysis.h3.pagecache_oracle import load_analysis_amendment


def _identity_fingerprint(identity: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(identity.get(key) for key in ("head", "tree", "required_base_commit"))


def _validate_oracle(
    oracle_path: Path,
    residency: dict[str, Any],
    measurement_identity: dict[str, Any],
    contract_path: Path,
    amendment_path: Path,
) -> dict[str, Any]:
    oracle = load_json(oracle_path)
    if oracle.get("artifact_kind") != "fenix_h3_pagecache_oracle":
        raise H3Error("unexpected page-cache oracle artifact")
    require_contract_sha(oracle, sha256_file(contract_path), "page-cache oracle")
    if oracle.get("analysis_amendment_sha256") != sha256_file(amendment_path):
        raise H3Error("page-cache oracle uses another analysis amendment")
    if str(oracle.get("stratum")) != str(residency.get("stratum")):
        raise H3Error("page-cache oracle stratum differs from residency replay")
    if float(oracle.get("capacity_gib", -1)) != float(residency.get("capacity_gib", -2)):
        raise H3Error("page-cache oracle capacity differs from residency replay")
    if residency.get("phase_filter") is not None:
        raise H3Error("page-cache oracle is defined only for full-stratum decisions")
    trace_sha = (residency.get("provenance") or {}).get("trace_manifest_sha256")
    if oracle.get("trace_manifest_sha256") != trace_sha:
        raise H3Error("page-cache oracle uses a different trace manifest")
    if int(oracle.get("logical_read_bytes", -1)) != int(
        residency.get("counters", {}).get("lpddr_read_useful_bytes", -2)
    ):
        raise H3Error("page-cache oracle logical endpoint differs from residency replay")
    source_identity = oracle.get("source_measurement_execution_repository")
    if not isinstance(source_identity, dict):
        raise H3Error("page-cache oracle lacks source measurement identity")
    if _identity_fingerprint(source_identity) != _identity_fingerprint(measurement_identity):
        raise H3Error("page-cache oracle source identity differs from measurement campaign")
    required = {
        "future_knowledge": True,
        "ple_storage_traffic_free": True,
        "expert_partial_tail_free": True,
        "all_capacity_dedicated_to_expert_pages": True,
        "storage_latency_hidden": True,
        "gap_falsification_allowed": True,
        "conventional_sufficiency_allowed": False,
    }
    for key, expected in required.items():
        if oracle.get(key) is not expected:
            raise H3Error(f"page-cache oracle lacks required dominance property {key}")
    if int(oracle.get("lower_tier_read_bytes", -1)) < 0:
        raise H3Error("page-cache oracle lower-tier bytes are invalid")
    return oracle


def decide_with_oracle(
    residency_path: Path,
    lpddr_path: Path,
    fio_path: Path,
    contract_path: Path,
    queue_depth: int,
    out_path: Path,
    amendment_path: Path,
    actual_lpddr_path: Path | None = None,
    oracle_path: Path | None = None,
    pagecache_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    amendment = load_analysis_amendment(amendment_path, contract_path)
    with tempfile.TemporaryDirectory(prefix="fenix-h3-oracle-decision-") as tmp:
        base_path = Path(tmp) / "base.json"
        decide_point(
            residency_path,
            lpddr_path,
            fio_path,
            contract_path,
            int(queue_depth),
            base_path,
            pagecache_paths,
            actual_lpddr_path,
        )
        result = load_json(base_path)

    residency = load_json(residency_path)
    measurement_identity = (result.get("provenance") or {}).get("execution_repository")
    if not isinstance(measurement_identity, dict):
        raise H3Error("base decision lacks measurement execution identity")
    if measurement_identity.get("head") != amendment["frozen_measurement_head"]:
        raise H3Error("decision measurements are not from the frozen measurement commit")

    result["analysis_amendment_sha256"] = sha256_file(amendment_path)
    result["analysis_amendment_name"] = amendment["name"]
    result["measurement_execution_repository"] = measurement_identity
    result["pagecache_oracle_used"] = False
    result["pagecache_gap_control_qualified"] = False
    result["pagecache_oracle_gap_falsification_only"] = True
    result["measured_pagecache_required_for_sufficiency"] = True

    if oracle_path is not None:
        oracle = _validate_oracle(
            oracle_path,
            residency,
            measurement_identity,
            contract_path,
            amendment_path,
        )
        read_bytes = float(residency["counters"]["lpddr_read_useful_bytes"])
        miss_bytes = float(oracle["lower_tier_read_bytes"])
        for point in result["sensitivity_points"]:
            read_bw = float(point["aggregate_read_bandwidth_gb_s"])
            write_bw = float(point["aggregate_write_bandwidth_gb_s"])
            mixed_bw = float(point["aggregate_mixed_total_bandwidth_gb_s"])
            all_lpddr = float(point["all_lpddr_oracle_ns"])
            fill = _service_ns(miss_bytes, write_bw)
            shared_bus = _service_ns(read_bytes + miss_bytes, mixed_bw)
            lower = max(all_lpddr, shared_bus)
            upper = all_lpddr + fill
            point["conventional_candidates"].append({
                "baseline": "future_aware_pagecache_dominance_oracle",
                "promotion_role": "gap_falsification_only",
                "role": "hostile_rootless_conventional_relaxation",
                "storage_latency_hidden": True,
                "ple_storage_traffic_free": True,
                "expert_partial_tail_free": True,
                "storage_read_bytes": miss_bytes,
                "lpddr_fill_service_ns": fill,
                "shared_lpddr_bus_lower_bound_ns": shared_bus,
                "service_ns_bounds": [lower, upper],
                "oracle_artifact": str(oracle_path.resolve()),
                "oracle_artifact_sha256": sha256_file(oracle_path),
            })
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

        global_low = min(
            float(point["gap_falsification_penalty_fraction_bounds"][0])
            for point in result["sensitivity_points"]
        )
        global_high = max(
            float(point["realizable_penalty_fraction_bounds"][1])
            for point in result["sensitivity_points"]
        )
        sufficient = float(result["thresholds"]["conventional_sufficient_max_penalty_fraction"])
        gap = float(result["thresholds"]["memory_gap_min_penalty_fraction"])
        if global_high <= sufficient:
            verdict = "CONVENTIONAL_MEMORY_SUFFICIENT"
        elif global_low >= gap:
            verdict = "H3_MEMORY_GAP_SUPPORTED"
        else:
            verdict = "INCONCLUSIVE_ESCALATE_INTEGRATED_TIMING"
        result["global_penalty_fraction_bounds"] = [global_low, global_high]
        result["verdict"] = verdict
        result["pagecache_oracle_used"] = True
        result["pagecache_gap_control_qualified"] = True
        result["pagecache_oracle_artifact"] = str(oracle_path.resolve())
        result["pagecache_oracle_sha256"] = sha256_file(oracle_path)
        result["provenance"]["pagecache_oracle_sha256"] = sha256_file(oracle_path)

    result["provenance"]["analysis_amendment_sha256"] = sha256_file(amendment_path)
    result["claim_boundary"] = (
        "Gap support may use the hostile future-aware page-cache oracle when "
        "measured page-cache isolation is unavailable. The oracle can only "
        "falsify a gap; conventional sufficiency still requires measured "
        "realizable evidence. No end-to-end inference or FeRAM superiority "
        "claim follows from this endpoint."
    )
    write_json(out_path, result)
    return result


def campaign_gate_with_oracle(
    decision_paths: Iterable[Path],
    prerequisites_path: Path,
    contract_path: Path,
    amendment_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    amendment = load_analysis_amendment(amendment_path, contract_path)
    paths = list(decision_paths)
    contract = load_json(contract_path)
    contract_sha = sha256_file(contract_path)
    amendment_sha = sha256_file(amendment_path)
    decisions = [load_json(path) for path in paths]
    if not decisions:
        raise H3Error("campaign gate requires at least one H3 decision")
    for row in decisions:
        if row.get("artifact_kind") != "fenix_h3_same_endpoint_decision":
            raise H3Error("campaign gate received a non-H3 decision artifact")
        require_contract_sha(row, contract_sha, "H3 decision")
        if row.get("analysis_amendment_sha256") != amendment_sha:
            raise H3Error("decision was not produced by this analysis amendment")
    analysis_identity = require_same_execution_repository(
        *[(f"decision {path}", row) for path, row in zip(paths, decisions, strict=True)]
    )

    measurement_identities = [row.get("measurement_execution_repository") for row in decisions]
    if any(not isinstance(value, dict) for value in measurement_identities):
        raise H3Error("amended decision lacks measurement execution identity")
    measurement_identity = dict(measurement_identities[0])
    if any(
        _identity_fingerprint(dict(value)) != _identity_fingerprint(measurement_identity)
        for value in measurement_identities[1:]
    ):
        raise H3Error("amended decisions mix measurement implementation commits")
    if measurement_identity.get("head") != amendment["frozen_measurement_head"]:
        raise H3Error("measurement identity differs from amendment frozen commit")

    promotion_qds = {int(value) for value in contract["storage"]["promotion_queue_depths"]}
    promotion_rows = [
        row for row in decisions
        if int(row["queue_depth"]) in promotion_qds
        and row.get("promotion_eligible_queue_depth") is True
    ]
    if not promotion_rows:
        raise H3Error("campaign gate has no promotion-eligible decisions")
    if any(row.get("actual_lpddr_trace_calibration_used") is not True for row in promotion_rows):
        raise H3Error("every promotion row requires actual-H3 LPDDR calibration")

    fingerprints = [row.get("campaign_fingerprint") for row in promotion_rows]
    if any(not isinstance(value, dict) for value in fingerprints):
        raise H3Error("promotion decision lacks campaign fingerprint")
    reference_fingerprint = dict(fingerprints[0])
    if any(dict(value) != reference_fingerprint for value in fingerprints[1:]):
        raise H3Error("promotion decisions mix source/storage/tool fingerprints")

    required = contract["decision_gate"]["campaign_requirements"]
    required_names = list(dict.fromkeys(
        [str(value) for value in required["required_strata"]]
        + [str(value) for value in required.get("required_ordinary_strata", [])]
    ))
    expected_policies = sorted(
        set(contract.get("primary_policies", []))
        | set(contract.get("strong_conventional_policies", []))
    )
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
                    if str(row["stratum"]) == name
                    and str(row["policy"]) == policy
                    and int(row["queue_depth"]) == qd
                ]
                if len(matches) != 1:
                    matrix_errors.append(f"full:{name}:{policy}:qd{qd}:count={len(matches)}")
                else:
                    selected_full.append(matches[0])
                for phase in required_phases:
                    pmatches = [
                        row for row in phase_rows[phase]
                        if str(row["stratum"]) == name
                        and str(row["policy"]) == policy
                        and int(row["queue_depth"]) == qd
                    ]
                    if len(pmatches) != 1:
                        matrix_errors.append(
                            f"{phase}:{name}:{policy}:qd{qd}:count={len(pmatches)}"
                        )
                    else:
                        selected_phase.append(pmatches[0])

    capacities = {float(row["capacity_gib"]) for row in [*selected_full, *selected_phase]}
    if len(capacities) != 1:
        matrix_errors.append(
            f"promotion rows do not use one common capacity: {sorted(capacities)}"
        )

    required_pagecache = set(contract.get("pagecache", {}).get("primary_strata", []))
    required_modes = set(contract.get("pagecache", {}).get("required_modes", []))
    measured_missing: list[str] = []
    oracle_missing: list[str] = []
    gap_control_missing: list[str] = []
    for name in sorted(required_pagecache):
        rows = [row for row in selected_full if str(row["stratum"]) == name]
        for policy in expected_policies:
            policy_rows = [row for row in rows if str(row["policy"]) == policy]
            measured_ok = bool(policy_rows) and all(
                row.get("pagecache_baseline_used") is True
                and required_modes.issubset(set(row.get("pagecache_modes", [])))
                for row in policy_rows
            )
            oracle_ok = bool(policy_rows) and all(
                row.get("pagecache_oracle_used") is True
                and row.get("pagecache_gap_control_qualified") is True
                and row.get("pagecache_oracle_gap_falsification_only") is True
                for row in policy_rows
            )
            label = f"{name}:{policy}"
            if not measured_ok:
                measured_missing.append(label)
            if not oracle_ok:
                oracle_missing.append(label)
            if not (measured_ok or oracle_ok):
                gap_control_missing.append(label)

    combined: dict[str, list[float]] = {}
    if matrix_errors:
        preliminary = "INCOMPLETE_OR_DUPLICATE_H3_MATRIX"
    elif gap_control_missing:
        preliminary = "INCOMPLETE_PAGECACHE_GAP_CONTROL"
    else:
        sufficient_threshold = float(
            contract["decision_gate"]["conventional_sufficient_max_penalty_fraction"]
        )
        gap_threshold = float(contract["decision_gate"]["memory_gap_min_penalty_fraction"])
        gap_all = True
        sufficient_all = True
        for name in required_names:
            scoped = [
                row for row in [*selected_full, *selected_phase]
                if str(row["stratum"]) == name
            ]
            gap_low = min(float(row["global_penalty_fraction_bounds"][0]) for row in scoped)
            policy_worst_high: dict[str, float] = {}
            for policy in expected_policies:
                policy_rows = [row for row in scoped if str(row["policy"]) == policy]
                policy_worst_high[policy] = max(
                    float(row["global_penalty_fraction_bounds"][1]) for row in policy_rows
                )
            realizable_high = min(policy_worst_high.values())
            combined[name] = [gap_low, realizable_high]
            gap_all = gap_all and gap_low >= gap_threshold
            sufficient_all = sufficient_all and realizable_high <= sufficient_threshold
        if sufficient_all:
            if measured_missing:
                preliminary = "INCONCLUSIVE_MEASURED_PAGECACHE_REQUIRED_FOR_SUFFICIENCY"
            else:
                preliminary = "CONVENTIONAL_MEMORY_SUFFICIENT"
        elif gap_all:
            preliminary = "H3_MEMORY_GAP_SUPPORTED"
        else:
            preliminary = "INCONCLUSIVE_ESCALATE_INTEGRATED_TIMING"

    directional = preliminary in {
        "H3_MEMORY_GAP_SUPPORTED",
        "CONVENTIONAL_MEMORY_SUFFICIENT",
    }
    prerequisite_status: dict[str, Any] | None = None
    prerequisite_error: str | None = None
    if directional and len(capacities) == 1 and not matrix_errors and not gap_control_missing:
        try:
            pagecache_complete = (
                not measured_missing
                if preliminary == "CONVENTIONAL_MEMORY_SUFFICIENT"
                else True
            )
            prerequisite_status = _validate_prerequisites(
                prerequisites_path,
                contract,
                contract_sha,
                measurement_identity,
                reference_fingerprint,
                next(iter(capacities)),
                pagecache_complete=pagecache_complete,
                preliminary_verdict=preliminary,
            )
            prerequisite_status["pagecache_control"] = {
                "derived_from_decision_matrix": True,
                "passed": True,
                "mode": (
                    "measured"
                    if not measured_missing
                    else "hostile_future_aware_oracle_gap_falsification_only"
                ),
            }
        except H3Error as exc:
            prerequisite_error = str(exc)

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
        "schema_version": 1,
        "artifact_kind": "fenix_h3_campaign_gate",
        "analysis_amendment_name": amendment["name"],
        "analysis_amendment_sha256": amendment_sha,
        "preliminary_memory_service_verdict": preliminary,
        "paper_promotion_verdict": promotion,
        "proceed_to_h4": proceed,
        "promotion_queue_depths": sorted(promotion_qds),
        "actual_lpddr_trace_required_and_verified": all(
            row.get("actual_lpddr_trace_calibration_used") is True
            for row in promotion_rows
        ),
        "explicit_phase_calibration_required": True,
        "required_strata": required_names,
        "required_policies": expected_policies,
        "required_phases": required_phases,
        "matrix_errors": matrix_errors,
        "missing_measured_pagecache_baselines": measured_missing,
        "missing_pagecache_oracles": oracle_missing,
        "missing_pagecache_gap_controls": gap_control_missing,
        "oracle_may_support_gap_only": True,
        "measured_pagecache_required_for_conventional_sufficiency": True,
        "decision_capacity_gib": next(iter(capacities)) if len(capacities) == 1 else None,
        "combined_directional_bounds": combined,
        "campaign_fingerprint": reference_fingerprint,
        "prerequisite_status": prerequisite_status,
        "prerequisite_error": prerequisite_error,
        "decision_count": len(decisions),
        "promotion_full_decision_count": len(selected_full),
        "promotion_phase_decision_count": len(selected_phase),
        "decision_sha256": {str(path): sha256_file(path) for path in paths},
        "prerequisites_sha256": sha256_file(prerequisites_path),
        "contract_sha256": contract_sha,
        "measurement_execution_repository": measurement_identity,
        "analysis_execution_repository": analysis_identity,
        "execution_repository_consistency_verified": True,
        "claim_boundary": (
            "The page-cache oracle is a hostile gap-falsification relaxation. "
            "It cannot establish conventional sufficiency. Positive H3 support "
            "remains conditional-memory-service evidence only."
        ),
    }
    write_json(out_path, result)
    return result
