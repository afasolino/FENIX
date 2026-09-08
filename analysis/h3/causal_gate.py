"""Campaign-level H3 promotion gate for the rootless causal amendment path."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis.h3.causal_decision import (
    CAUSAL_BELADY_BASELINE,
    REMOVED_BASELINE,
    _validate_causality,
)
from analysis.h3.causal_matrix import required_matrix_keys
from analysis.h3.common import (
    H3Error,
    load_json,
    require_contract_sha,
    require_same_execution_repository,
    sha256_file,
    write_json,
)
from analysis.h3.decision import _validate_prerequisites


def _load_promotion_contract(
    path: Path,
    contract_path: Path,
    amendment_path: Path,
) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("artifact_kind") != "fenix_h3_causal_promotion_contract":
        raise H3Error("unexpected causal promotion contract")
    if payload.get("base_contract_sha256") != sha256_file(contract_path):
        raise H3Error("causal promotion contract is not bound to the H3 contract")
    if payload.get("causality_amendment_sha256") != sha256_file(amendment_path):
        raise H3Error("causal promotion contract is not bound to the causal amendment")
    substitution = payload.get("pagecache_substitution") or {}
    if substitution.get("allowed_for_gap_promotion_only") is not True:
        raise H3Error("causal promotion contract does not permit gap-only oracle substitution")
    if substitution.get("measured_linux_pagecache_required_for_conventional_sufficiency") is not True:
        raise H3Error("causal promotion contract weakens conventional-sufficiency requirements")
    if substitution.get("conventional_sufficiency_may_not_be_promoted_by_this_gate") is not True:
        raise H3Error("causal promotion gate must prohibit conventional-sufficiency promotion")
    return payload


def _resolve_manifest_decisions(
    matrix_path: Path,
    matrix: dict[str, Any],
) -> list[tuple[Path, dict[str, Any]]]:
    rows = matrix.get("decisions") or []
    if not isinstance(rows, list) or not rows:
        raise H3Error("causal matrix manifest contains no decisions")
    resolved: list[tuple[Path, dict[str, Any]]] = []
    for entry in rows:
        if not isinstance(entry, dict):
            raise H3Error("causal matrix contains a non-object decision entry")
        raw = entry.get("artifact")
        expected_sha = entry.get("sha256")
        if not raw or not expected_sha:
            raise H3Error("causal matrix decision entry lacks artifact/SHA")
        path = Path(str(raw))
        if not path.is_absolute():
            path = (matrix_path.parent / path).resolve()
        if not path.is_file():
            raise H3Error(f"causal decision artifact is missing: {path}")
        if sha256_file(path) != expected_sha:
            raise H3Error(f"causal decision SHA drift: {path}")
        resolved.append((path, load_json(path)))
    return resolved


def _same_measurement_identity(decisions: list[dict[str, Any]], frozen_head: str) -> dict[str, Any]:
    identities = [row.get("measurement_execution_repository") or {} for row in decisions]
    if any(identity.get("head") != frozen_head for identity in identities):
        raise H3Error("causal decisions are not bound to the frozen measurement HEAD")
    reference = dict(identities[0])
    keys = ("head", "tree", "required_base_commit")
    if any(tuple(identity.get(k) for k in keys) != tuple(reference.get(k) for k in keys) for identity in identities[1:]):
        raise H3Error("causal decisions mix frozen measurement repository identities")
    return reference


def causal_campaign_gate(
    matrix_path: Path,
    prerequisites_path: Path,
    causality_path: Path,
    contract_path: Path,
    amendment_path: Path,
    promotion_contract_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    contract = load_json(contract_path)
    contract_sha = sha256_file(contract_path)
    amendment_sha = sha256_file(amendment_path)
    promotion_contract = _load_promotion_contract(
        promotion_contract_path, contract_path, amendment_path
    )
    causality = _validate_causality(causality_path, amendment_path, contract_path)
    causality_sha = sha256_file(causality_path)
    frozen_head = str(promotion_contract["frozen_measurement_head"])
    if causality.get("frozen_measurement_head") != frozen_head:
        raise H3Error("causal evidence and promotion contract use different frozen measurements")

    matrix = load_json(matrix_path)
    if matrix.get("artifact_kind") != "fenix_h3_causal_decision_matrix":
        raise H3Error("unexpected causal matrix manifest")
    if matrix.get("contract_sha256") != contract_sha:
        raise H3Error("causal matrix uses another H3 contract")
    if matrix.get("causality_amendment_sha256") != amendment_sha:
        raise H3Error("causal matrix uses another causal amendment")
    if matrix.get("causality_sha256") != causality_sha:
        raise H3Error("causal matrix uses another causality artifact")
    if matrix.get("exact_matrix") is not True:
        raise H3Error("causal matrix is not declared exact")

    resolved = _resolve_manifest_decisions(matrix_path, matrix)
    paths = [path for path, _ in resolved]
    decisions = [row for _, row in resolved]
    expected_keys = required_matrix_keys(contract)
    if int(matrix.get("expected_decision_count", -1)) != len(expected_keys):
        raise H3Error("causal matrix expected-decision count differs from contract")
    if len(decisions) != len(expected_keys):
        raise H3Error(f"causal matrix has {len(decisions)} decisions, expected {len(expected_keys)}")

    for row in decisions:
        if row.get("artifact_kind") != "fenix_h3_same_endpoint_decision":
            raise H3Error("causal gate received a non-H3 decision artifact")
        require_contract_sha(row, contract_sha, "causal H3 decision")
        if row.get("causal_expert_prefetch_evidence_used") is not True:
            raise H3Error("causal decision did not use expert-prefetch evidence")
        if row.get("expert_storage_lower_bounds_causalized") is not True:
            raise H3Error("causal decision did not serialize unavoidable expert storage")
        if row.get("ple_perfect_prefetch_sensitivity_retained") is not True:
            raise H3Error("causal decision incorrectly removed perfect PLE prefetch")
        if row.get("removed_unbounded_sensitivity_baseline") != REMOVED_BASELINE:
            raise H3Error("causal decision removed an unexpected sensitivity")
        if row.get("causality_artifact_sha256") != causality_sha:
            raise H3Error("causal decision uses another causality artifact")
        if row.get("causality_amendment_sha256") != amendment_sha:
            raise H3Error("causal decision uses another causality amendment")
        if row.get("actual_lpddr_trace_calibration_used") is not True:
            raise H3Error("causal promotion requires actual-H3 LPDDR calibration on every row")
        if row.get("promotion_eligible_queue_depth") is not True:
            raise H3Error("causal matrix contains a non-promotion queue depth")

    analysis_identity = require_same_execution_repository(
        *[(f"causal decision {path}", row) for path, row in resolved]
    )
    measurement_identity = _same_measurement_identity(decisions, frozen_head)

    observed: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in decisions:
        key = (str(row["stratum"]), str(row["policy"]), row.get("phase_filter"))
        if key in observed:
            duplicates.append(str(key))
        observed[key] = row
    missing = [key for key in expected_keys if key not in observed]
    extras = [key for key in observed if key not in set(expected_keys)]
    if duplicates or missing or extras:
        raise H3Error(
            f"causal matrix mismatch duplicates={duplicates} missing={missing} extras={extras}"
        )

    required_capacity = float(promotion_contract["matrix"]["capacity_gib"])
    required_qd = int(promotion_contract["matrix"]["queue_depth"])
    if any(float(row["capacity_gib"]) != required_capacity for row in decisions):
        raise H3Error("causal matrix does not use the predeclared common capacity")
    if any(int(row["queue_depth"]) != required_qd for row in decisions):
        raise H3Error("causal matrix does not use the predeclared promotion queue depth")

    fingerprints = [row.get("campaign_fingerprint") for row in decisions]
    if any(not isinstance(value, dict) for value in fingerprints):
        raise H3Error("causal decision lacks campaign_fingerprint")
    fingerprint = dict(fingerprints[0])
    if any(dict(value) != fingerprint for value in fingerprints[1:]):
        raise H3Error("causal decisions mix source/storage/tool campaign fingerprints")

    special = set(str(v) for v in promotion_contract["matrix"]["special_full_strata"])
    required_special_baseline = str(
        promotion_contract["matrix"]["required_special_full_baseline"]
    )
    special_errors: list[str] = []
    for (stratum, policy, phase), row in observed.items():
        should_use = phase is None and stratum in special
        used = row.get("causal_belady_pagecache_used") is True
        if should_use and not used:
            special_errors.append(f"missing_causal_belady:{stratum}:{policy}")
        if not should_use and used:
            special_errors.append(f"unexpected_causal_belady:{stratum}:{policy}:{phase}")
        if should_use and row.get("causal_belady_pagecache_baseline") != required_special_baseline:
            special_errors.append(f"wrong_causal_belady:{stratum}:{policy}")
    if special_errors:
        raise H3Error("; ".join(special_errors))

    gap = float(contract["decision_gate"]["memory_gap_min_penalty_fraction"])
    sufficient = float(contract["decision_gate"]["conventional_sufficient_max_penalty_fraction"])
    required = contract["decision_gate"]["campaign_requirements"]
    required_names = list(dict.fromkeys(
        [str(v) for v in required["required_strata"]]
        + [str(v) for v in required.get("required_ordinary_strata", [])]
    ))
    policies = sorted(
        set(str(v) for v in contract.get("primary_policies", []))
        | set(str(v) for v in contract.get("strong_conventional_policies", []))
    )

    combined: dict[str, list[float]] = {}
    gap_all = True
    sufficient_all = True
    for stratum in required_names:
        scoped = [row for (s, _, _), row in observed.items() if s == stratum]
        gap_low = min(float(row["global_penalty_fraction_bounds"][0]) for row in scoped)
        policy_worst_high: dict[str, float] = {}
        for policy in policies:
            policy_rows = [
                row for (s, p, _), row in observed.items()
                if s == stratum and p == policy
            ]
            policy_worst_high[policy] = max(
                float(row["global_penalty_fraction_bounds"][1]) for row in policy_rows
            )
        realizable_high = min(policy_worst_high.values())
        combined[stratum] = [gap_low, realizable_high]
        gap_all = gap_all and gap_low >= gap
        sufficient_all = sufficient_all and realizable_high <= sufficient

    if gap_all:
        preliminary = "H3_MEMORY_GAP_SUPPORTED"
    elif sufficient_all:
        preliminary = "INCONCLUSIVE_MEASURED_PAGECACHE_REQUIRED_FOR_SUFFICIENCY"
    else:
        preliminary = "INCONCLUSIVE_ESCALATE_INTEGRATED_TIMING"

    prerequisite_status: dict[str, Any] | None = None
    prerequisite_error: str | None = None
    try:
        prerequisite_status = _validate_prerequisites(
            prerequisites_path,
            contract,
            contract_sha,
            measurement_identity,
            fingerprint,
            required_capacity,
            pagecache_complete=True,
            preliminary_verdict=preliminary,
        )
        prerequisite_status = dict(prerequisite_status)
        prerequisite_status["pagecache_control"] = {
            "passed": True,
            "measured_linux_pagecache_used": False,
            "substitution": "future_aware_causal_belady_upper_bound",
            "gap_promotion_only": True,
            "special_full_strata": sorted(special),
            "measured_linux_pagecache_still_required_for_conventional_sufficiency": True,
        }
    except H3Error as exc:
        prerequisite_error = str(exc)

    supported = preliminary == "H3_MEMORY_GAP_SUPPORTED" and prerequisite_status is not None
    paper_verdict = (
        str(promotion_contract["promotion"]["supported_verdict"])
        if supported
        else str(promotion_contract["promotion"]["blocked_verdict"])
    )
    proceed_to_h4 = bool(
        supported and promotion_contract["promotion"].get("proceed_to_h4_on_supported_gap")
    )

    limiting_key, limiting_row = min(
        observed.items(), key=lambda item: float(item[1]["global_penalty_fraction_bounds"][0])
    )
    result = {
        "schema_version": 1,
        "artifact_kind": "fenix_h3_causal_campaign_gate",
        "scope": "conditional_memory_service_layer",
        "preliminary_memory_service_verdict": preliminary,
        "paper_promotion_verdict": paper_verdict,
        "proceed_to_h4": proceed_to_h4,
        "decision_count": len(decisions),
        "exact_matrix_verified": True,
        "capacity_gib": required_capacity,
        "queue_depth": required_qd,
        "actual_lpddr_trace_required_and_verified": True,
        "causal_expert_prefetch_required_and_verified": True,
        "ple_perfect_prefetch_retained": True,
        "future_aware_causal_belady_gap_control_verified": True,
        "measured_linux_pagecache_used": False,
        "measured_linux_pagecache_required_for_conventional_sufficiency": True,
        "conventional_sufficiency_promotion_allowed": False,
        "combined_directional_bounds": combined,
        "memory_gap_threshold": gap,
        "conventional_sufficiency_threshold": sufficient,
        "limiting_matrix_row": {
            "stratum": limiting_key[0],
            "policy": limiting_key[1],
            "phase_filter": limiting_key[2],
            "global_penalty_fraction_bounds": limiting_row["global_penalty_fraction_bounds"],
            "artifact_sha256": sha256_file(paths[decisions.index(limiting_row)]),
        },
        "campaign_fingerprint": fingerprint,
        "frozen_measurement_execution_repository": measurement_identity,
        "analysis_execution_repository": analysis_identity,
        "causality_artifact": str(causality_path.resolve()),
        "causality_sha256": causality_sha,
        "causality_amendment_sha256": amendment_sha,
        "causal_promotion_contract_sha256": sha256_file(promotion_contract_path),
        "matrix_manifest_sha256": sha256_file(matrix_path),
        "prerequisites_sha256": sha256_file(prerequisites_path),
        "prerequisite_status": prerequisite_status,
        "prerequisite_error": prerequisite_error,
        "claim_boundary": promotion_contract["claim_boundary"],
        "not_end_to_end_inference_latency": True,
        "energy_superiority_claim_allowed": False,
    }
    write_json(out_path, result)
    return result
