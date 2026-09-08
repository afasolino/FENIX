"""Exact analysis-only H3 causal re-decision matrix over frozen measurements."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis.h3.causal_decision import decide_with_causal_prefetch
from analysis.h3.common import H3Error, load_json, sha256_file, write_json


def required_matrix_keys(contract: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    required = contract["decision_gate"]["campaign_requirements"]
    strata = list(dict.fromkeys(
        [str(v) for v in required["required_strata"]]
        + [str(v) for v in required.get("required_ordinary_strata", [])]
    ))
    policies = sorted(
        set(str(v) for v in contract.get("primary_policies", []))
        | set(str(v) for v in contract.get("strong_conventional_policies", []))
    )
    phases: list[str | None] = [None] + [
        str(v) for v in required.get("required_phases", [])
    ]
    return [(stratum, policy, phase) for stratum in strata for policy in policies for phase in phases]


def _phase_label(phase: str | None) -> str:
    return "full" if phase is None else str(phase)


def _index_residencies(root: Path, capacity_gib: float) -> dict[tuple[str, str, str | None], Path]:
    indexed: dict[tuple[str, str, str | None], Path] = {}
    replay_root = root / "replay"
    if not replay_root.is_dir():
        raise H3Error(f"frozen H3 replay root is missing: {replay_root}")
    for path in replay_root.rglob("summary.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if payload.get("artifact_kind") != "fenix_h3_residency_replay":
            continue
        if float(payload.get("capacity_gib", -1)) != float(capacity_gib):
            continue
        key = (
            str(payload.get("stratum")),
            str(payload.get("policy")),
            payload.get("phase_filter"),
        )
        if key in indexed:
            raise H3Error(f"duplicate frozen residency for {key}: {indexed[key]} and {path}")
        indexed[key] = path
    return indexed


def _index_fio(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    storage_root = root / "storage"
    if not storage_root.is_dir():
        raise H3Error(f"frozen H3 storage root is missing: {storage_root}")
    for path in storage_root.rglob("fio-summary.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if payload.get("artifact_kind") != "fenix_h3_fio_calibration":
            continue
        miss_sha = str((payload.get("provenance") or {}).get("miss_stream_sha256") or "")
        if not miss_sha:
            continue
        if miss_sha in indexed:
            raise H3Error(f"duplicate fio calibration for miss stream {miss_sha}")
        indexed[miss_sha] = path
    return indexed


def _index_actual_lpddr(root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    actual_root = root / "lpddr-actual"
    if not actual_root.is_dir():
        raise H3Error(f"frozen actual-H3 LPDDR root is missing: {actual_root}")
    for path in actual_root.rglob("*.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if payload.get("artifact_kind") != "fenix_h3_lpddr_actual_trace_calibration":
            continue
        residency_sha = str((payload.get("provenance") or {}).get("residency_sha256") or "")
        if not residency_sha:
            continue
        if residency_sha in indexed:
            raise H3Error(f"duplicate actual-H3 LPDDR artifact for residency {residency_sha}")
        indexed[residency_sha] = path
    return indexed


def build_causal_matrix(
    measurement_root: Path,
    oracle_root: Path,
    causality_path: Path,
    contract_path: Path,
    amendment_path: Path,
    capacity_gib: float,
    queue_depth: int,
    out_root: Path,
) -> dict[str, Any]:
    measurement_root = measurement_root.resolve()
    oracle_root = oracle_root.resolve()
    causality_path = causality_path.resolve()
    out_root = out_root.resolve()
    contract = load_json(contract_path)
    contract_sha = sha256_file(contract_path)
    amendment_sha = sha256_file(amendment_path)

    if not causality_path.is_file():
        raise H3Error(f"causality artifact is missing: {causality_path}")
    causality = load_json(causality_path)
    if causality.get("artifact_kind") != "fenix_h3_expert_prefetch_causality":
        raise H3Error("unexpected causal evidence artifact")
    if causality.get("derived_pass") is not True or causality.get("failures"):
        raise H3Error("causal expert-prefetch evidence did not pass")
    if causality.get("contract_sha256") != contract_sha:
        raise H3Error("causal evidence uses another H3 contract")
    if causality.get("analysis_amendment_sha256") != amendment_sha:
        raise H3Error("causal evidence uses another H3 analysis amendment")

    generic_lpddr = measurement_root / "lpddr-generic.json"
    if not generic_lpddr.is_file():
        raise H3Error(f"generic LPDDR artifact is missing: {generic_lpddr}")

    required = required_matrix_keys(contract)
    expected_capacity = float(capacity_gib)
    expected_qds = {int(v) for v in contract["storage"]["promotion_queue_depths"]}
    if int(queue_depth) not in expected_qds:
        raise H3Error(f"queue depth {queue_depth} is not promotion eligible: {sorted(expected_qds)}")

    residencies = _index_residencies(measurement_root, expected_capacity)
    fio_by_miss = _index_fio(measurement_root)
    actual_by_residency = _index_actual_lpddr(measurement_root)
    special = set(str(v) for v in load_json(amendment_path)["causality"]["special_full_strata"])

    missing = [key for key in required if key not in residencies]
    extras = [key for key in residencies if key not in set(required)]
    if missing:
        raise H3Error(f"frozen H3 matrix is missing residency rows: {missing}")
    if extras:
        # Extra rows at the same capacity are not consumed, but recording them
        # silently would make the exact-matrix claim ambiguous.
        raise H3Error(f"unexpected extra frozen residency rows at {capacity_gib} GiB: {extras}")

    out_root.mkdir(parents=True, exist_ok=True)
    decision_root = out_root / "decisions"
    rows: list[dict[str, Any]] = []

    for ordinal, (stratum, policy, phase) in enumerate(required, 1):
        residency_path = residencies[(stratum, policy, phase)]
        residency = load_json(residency_path)
        miss_sha = str((residency.get("miss_stream") or {}).get("sha256") or "")
        if miss_sha not in fio_by_miss:
            raise H3Error(f"no fio calibration bound to {stratum}/{policy}/{_phase_label(phase)}")
        fio_path = fio_by_miss[miss_sha]

        residency_sha = sha256_file(residency_path)
        if residency_sha not in actual_by_residency:
            raise H3Error(f"no actual-H3 LPDDR calibration bound to {stratum}/{policy}/{_phase_label(phase)}")
        actual_path = actual_by_residency[residency_sha]

        oracle_path: Path | None = None
        if phase is None and stratum in special:
            oracle_path = oracle_root / "pagecache-oracle" / f"{stratum}.json"
            if not oracle_path.is_file():
                raise H3Error(f"required hostile cache oracle is missing: {oracle_path}")

        output = decision_root / stratum / policy / _phase_label(phase) / f"decision-q{int(queue_depth)}.json"
        result = decide_with_causal_prefetch(
            residency_path=residency_path,
            lpddr_path=generic_lpddr,
            fio_path=fio_path,
            actual_lpddr_path=actual_path,
            causality_path=causality_path,
            contract_path=contract_path,
            amendment_path=amendment_path,
            queue_depth=int(queue_depth),
            out_path=output,
            pagecache_oracle_path=oracle_path,
        )
        if (
            str(result.get("stratum")) != stratum
            or str(result.get("policy")) != policy
            or result.get("phase_filter") != phase
        ):
            raise H3Error("causal decision identity does not match requested matrix row")
        rows.append({
            "ordinal": ordinal,
            "stratum": stratum,
            "policy": policy,
            "phase_filter": phase,
            "scope": _phase_label(phase),
            "artifact": str(output),
            "sha256": sha256_file(output),
            "verdict": result.get("verdict"),
            "global_penalty_fraction_bounds": result.get("global_penalty_fraction_bounds"),
            "causal_belady_pagecache_used": result.get("causal_belady_pagecache_used"),
            "fio_artifact": str(fio_path),
            "actual_lpddr_artifact": str(actual_path),
        })

    if len(rows) != len(required):
        raise H3Error(f"causal matrix emitted {len(rows)} rows, expected {len(required)}")

    manifest = {
        "schema_version": 1,
        "artifact_kind": "fenix_h3_causal_decision_matrix",
        "scope": "conditional_memory_service_layer",
        "capacity_gib": expected_capacity,
        "queue_depth": int(queue_depth),
        "exact_matrix": True,
        "expected_decision_count": len(required),
        "decision_count": len(rows),
        "measurement_root": str(measurement_root),
        "oracle_root": str(oracle_root),
        "causality_artifact": str(causality_path),
        "causality_sha256": sha256_file(causality_path),
        "contract_sha256": contract_sha,
        "causality_amendment_sha256": amendment_sha,
        "frozen_measurement_head": causality.get("frozen_measurement_head"),
        "decisions": rows,
    }
    manifest_path = out_root / "causal-matrix.json"
    write_json(manifest_path, manifest)
    return manifest
