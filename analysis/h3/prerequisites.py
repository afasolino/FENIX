"""Derived, evidence-bound H3 promotion prerequisites."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from analysis.h3.common import H3Error, iter_jsonl, load_json, sha256_file, write_json


def _resolve(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()




def _canonical_digest(rows: list[Any]) -> str:
    # Placement invariance is order-sensitive: the same selected objects in a
    # different request/token order can change cache locality.  Canonicalize
    # field ordering, but preserve the trace sequence exactly.
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _placement_semantic_sha(path: Path, field: str) -> str:
    if field == "prompts":
        payload = load_json(path)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if field == "ple_normalized":
        rows = []
        for row in iter_jsonl(path):
            rows.append({
                "request_id": str(row["request_id"]),
                "token_position": int(row["token_position"]),
                "ple_head": int(row["ple_head"]),
                "physical_row_id": int(row["physical_row_id"]),
                "bytes": int(row["bytes"]),
                "phase": str(row.get("phase", "unknown")),
            })
        return _canonical_digest(rows)
    if field == "moe_normalized":
        rows = []
        for row in iter_jsonl(path):
            rows.append({
                "request_id": str(row["request_id"]),
                "layer": str(row["layer"]),
                "selected_expert_ids": [int(value) for value in row.get("selected_expert_ids", [])],
                "phase": str(row.get("phase", "unknown")),
                "trace_scope": row.get("trace_scope"),
            })
        return _canonical_digest(rows)
    raise H3Error(f"unknown placement semantic field {field}")

def build_capacity_budget(input_path: Path, contract_path: Path, out_path: Path) -> dict[str, Any]:
    payload = load_json(input_path)
    contract = load_json(contract_path)
    physical = float(contract["lpddr"]["platform_capacity_gib"])
    declared_physical = payload.get("physical_memory_gib")
    if declared_physical is not None and abs(float(declared_physical) - physical) > 1e-12:
        raise H3Error(
            "primary H3 capacity budget must use contract platform_capacity_gib; "
            "a different physical capacity requires a separate sensitivity contract"
        )
    if str(payload.get("platform_mode", "primary")) != "primary":
        raise H3Error("paper-promotion capacity budget must use platform_mode=primary")

    required = {
        "os_and_services",
        "runtime",
        "nonconditional_model",
        "kv_cache",
        "activations_buffers",
        "safety_headroom",
    }
    components = payload.get("reserved_components") or {}
    missing = sorted(required - set(components))
    if missing:
        raise H3Error(f"capacity budget missing components: {missing}")
    resolved_components: dict[str, Any] = {}
    total = 0.0
    for name in sorted(required):
        node = components[name]
        if not isinstance(node, dict):
            raise H3Error(f"capacity component {name} must bind value and evidence")
        gib = float(node.get("gib", float("nan")))
        if not math.isfinite(gib) or gib < 0:
            raise H3Error(f"capacity component {name} must be finite and non-negative")
        artifact_raw = node.get("artifact")
        expected_sha = node.get("sha256")
        if not artifact_raw or not expected_sha:
            raise H3Error(f"capacity component {name} lacks evidence artifact/SHA256")
        artifact = _resolve(str(artifact_raw), input_path.parent)
        if not artifact.is_file():
            raise H3Error(f"capacity component {name} evidence does not exist: {artifact}")
        observed_sha = sha256_file(artifact)
        if observed_sha != str(expected_sha):
            raise H3Error(f"capacity component {name} evidence SHA mismatch")
        resolved_components[name] = {
            "gib": gib,
            "artifact": str(artifact),
            "sha256": observed_sha,
        }
        total += gib
    available = physical - total
    if available < 0:
        raise H3Error("reserved memory exceeds contract physical memory")
    candidates = [
        float(value) for value in contract["capacity_gib"] if float(value) <= available + 1e-12
    ]
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_capacity_budget",
        "platform_mode": "primary",
        "physical_memory_source": "contract.lpddr.platform_capacity_gib",
        "physical_memory_gib": physical,
        "reserved_components": resolved_components,
        "reserved_total_gib": total,
        "conditional_state_available_gib": available,
        "candidate_budgets_that_fit_gib": candidates,
        "derived": True,
        "contract_sha256": sha256_file(contract_path),
        "input_sha256": sha256_file(input_path),
    }
    write_json(out_path, result)
    return result


def build_placement_invariance(input_path: Path, contract_path: Path, out_path: Path) -> dict[str, Any]:
    payload = load_json(input_path)
    contract = load_json(contract_path)
    spec = contract["prerequisites"]["placement_invariance"]
    placements = payload.get("placements") or []
    minimum = int(spec.get("minimum_placements", 2))
    required_strata = [str(value) for value in spec.get("required_strata", [])]
    if len(placements) < minimum:
        raise H3Error(f"placement invariance requires at least {minimum} placement controls")

    resolved: list[dict[str, Any]] = []
    for placement in placements:
        name = str(placement.get("name", ""))
        cases = placement.get("cases") or {}
        if not name:
            raise H3Error("placement control lacks a name")
        if sorted(cases) != sorted(required_strata):
            raise H3Error(
                f"placement {name} case set {sorted(cases)} != required {sorted(required_strata)}"
            )
        case_rows: dict[str, Any] = {}
        for stratum in required_strata:
            node = cases[stratum]
            raw_hashes: dict[str, str] = {}
            semantic_hashes: dict[str, str] = {}
            paths: dict[str, str] = {}
            for field in ("prompts", "ple_normalized", "moe_normalized"):
                raw = node.get(field)
                if not raw:
                    raise H3Error(f"placement {name}/{stratum} lacks {field}")
                path = _resolve(str(raw), input_path.parent)
                if not path.is_file():
                    raise H3Error(f"placement {name}/{stratum} file missing: {path}")
                paths[field] = str(path)
                raw_hashes[field] = sha256_file(path)
                semantic_hashes[field] = _placement_semantic_sha(path, field)
            case_rows[stratum] = {
                "paths": paths,
                "raw_sha256": raw_hashes,
                "semantic_sha256": semantic_hashes,
            }
        resolved.append({"name": name, "cases": case_rows})

    reference = resolved[0]
    mismatches: list[dict[str, Any]] = []
    for placement in resolved[1:]:
        for stratum in required_strata:
            for field in ("prompts", "ple_normalized", "moe_normalized"):
                expected = reference["cases"][stratum]["semantic_sha256"][field]
                observed = placement["cases"][stratum]["semantic_sha256"][field]
                if observed != expected:
                    mismatches.append({
                        "placement": placement["name"],
                        "stratum": stratum,
                        "field": field,
                        "reference_sha256": expected,
                        "observed_sha256": observed,
                    })
    derived_pass = not mismatches
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_placement_invariance_evidence",
        "evaluator": "ordered_routing_and_ple_semantic_identity_across_physical_placements",
        "derived_pass": derived_pass,
        "minimum_placements": minimum,
        "required_strata": required_strata,
        "placements": resolved,
        "mismatches": mismatches,
        "contract_sha256": sha256_file(contract_path),
        "input_sha256": sha256_file(input_path),
    }
    write_json(out_path, result)
    return result


def build_long_context_scaling(input_path: Path, contract_path: Path, out_path: Path) -> dict[str, Any]:
    payload = load_json(input_path)
    contract = load_json(contract_path)
    spec = contract["prerequisites"]["long_context_beyond_8k"]
    case_dirs = payload.get("case_dirs") or []
    minimum = int(spec.get("minimum_points", 3))
    if len(case_dirs) < minimum:
        raise H3Error(f"long-context scaling evidence requires at least {minimum} case directories")

    points: list[dict[str, Any]] = []
    for raw in case_dirs:
        case_dir = _resolve(str(raw), input_path.parent)
        evidence_path = case_dir / "evidence.json"
        client_path = case_dir / "client.jsonl"
        ple_path = case_dir / "ple_normalized.jsonl"
        moe_path = case_dir / "moe_normalized.jsonl"
        for path in (evidence_path, client_path, ple_path, moe_path):
            if not path.is_file():
                raise H3Error(f"long-context evidence file missing: {path}")
        evidence = load_json(evidence_path)
        case = evidence.get("case") or {}
        if evidence.get("trace_valid") is not True:
            raise H3Error(f"long-context case is not trace-valid: {case_dir}")
        if int(case.get("concurrency", 0)) != int(contract["source_trace"]["concurrency"]):
            raise H3Error("long-context control concurrency differs from H3 source contract")
        if str(case.get("correlation_mode")) != str(contract["source_trace"]["correlation_mode"]):
            raise H3Error("long-context control correlation mode differs from H3 source contract")

        clients = [row for row in iter_jsonl(client_path) if "error" not in row]
        if not clients:
            raise H3Error(f"long-context case has no successful requests: {case_dir}")
        max_prompt = max(int(row["prompt_tokens"]) for row in clients)
        model_tokens = sum(
            int(row["prompt_tokens"]) + max(int(row["completion_tokens"]) - 1, 0)
            for row in clients
        )
        ple_rows = list(iter_jsonl(ple_path))
        moe_rows = list(iter_jsonl(moe_path))
        ple_unique = {int(row["physical_row_id"]) for row in ple_rows}
        expert_pairs: set[tuple[str, int]] = set()
        selected_count = 0
        for row in moe_rows:
            layer = str(row["layer"])
            selected = [int(value) for value in row.get("selected_expert_ids", [])]
            selected_count += len(selected)
            expert_pairs.update((layer, value) for value in selected)
        if model_tokens <= 0 or not ple_rows or not moe_rows:
            raise H3Error(f"long-context case lacks conditional-state observations: {case_dir}")
        points.append({
            "case_dir": str(case_dir),
            "stratum": case.get("stratum"),
            "request_count": len(clients),
            "max_prompt_tokens": max_prompt,
            "model_token_observations": model_tokens,
            "conditional_state_metrics": {
                "ple_row_records": len(ple_rows),
                "ple_unique_rows": len(ple_unique),
                "ple_row_records_per_model_token": len(ple_rows) / model_tokens,
                "moe_trace_records": len(moe_rows),
                "moe_selected_expert_ids": selected_count,
                "moe_unique_layer_expert_pairs": len(expert_pairs),
                "moe_selected_expert_ids_per_model_token": selected_count / model_tokens,
            },
            "sha256": {
                "evidence": sha256_file(evidence_path),
                "client": sha256_file(client_path),
                "ple_normalized": sha256_file(ple_path),
                "moe_normalized": sha256_file(moe_path),
            },
        })

    points.sort(key=lambda row: int(row["max_prompt_tokens"]))
    bands = list(spec.get("required_context_bands") or [])
    coverage: dict[str, dict[str, Any]] = {}
    used_cases: set[str] = set()
    for band in bands:
        name = str(band["name"])
        lo = int(band["min_prompt_tokens"])
        raw_hi = band.get("max_prompt_tokens_exclusive")
        hi = int(raw_hi) if raw_hi is not None else None
        matches = [
            row for row in points
            if int(row["max_prompt_tokens"]) >= lo
            and (hi is None or int(row["max_prompt_tokens"]) < hi)
        ]
        if matches:
            chosen = min(matches, key=lambda row: int(row["max_prompt_tokens"]))
            used_cases.add(str(chosen["case_dir"]))
            coverage[name] = {
                "present": True,
                "min_prompt_tokens": lo,
                "max_prompt_tokens_exclusive": hi,
                "case_dir": chosen["case_dir"],
                "observed_max_prompt_tokens": chosen["max_prompt_tokens"],
            }
        else:
            coverage[name] = {
                "present": False,
                "min_prompt_tokens": lo,
                "max_prompt_tokens_exclusive": hi,
            }
    derived_pass = bool(bands) and all(node["present"] for node in coverage.values()) and len(used_cases) == len(bands)
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_long_context_scaling_evidence",
        "evaluator": "trace_validated_multi_length_context_scaling_characterization",
        "derived_pass": derived_pass,
        "required_context_bands": bands,
        "coverage_by_band": coverage,
        "distinct_band_cases": len(used_cases),
        "maximum_demonstrated_prompt_tokens": max(int(row["max_prompt_tokens"]) for row in points),
        "points": points,
        "claim_boundary": str(spec.get("claim_boundary", "multi-length conditional-state characterization only")),
        "contract_sha256": sha256_file(contract_path),
        "input_sha256": sha256_file(input_path),
    }
    write_json(out_path, result)
    return result

def build_storage_concurrency_evidence(
    measurement_path: Path, contract_path: Path, out_path: Path
) -> dict[str, Any]:
    """Derive QD32 scheduler feasibility from raw per-object workload events."""
    measurement = load_json(measurement_path)
    contract = load_json(contract_path)
    spec = contract["prerequisites"]["conditional_evidence"]["storage_concurrency_feasibility"]
    expected_kind = str(spec["measurement_artifact_kind"])
    if measurement.get("artifact_kind") != expected_kind:
        raise H3Error(f"storage concurrency measurement must be {expected_kind}")
    if measurement.get("measured") is not True or measurement.get("workload_derived_schedule") is not True:
        raise H3Error("storage scheduler measurement must be measured and workload-derived")
    trace_sha = str(measurement.get("trace_manifest_sha256", ""))
    binding_sha = str(measurement.get("storage_binding_sha256", ""))
    if not trace_sha or not binding_sha:
        raise H3Error("storage scheduler measurement must bind trace_manifest_sha256 and storage_binding_sha256")

    required_qd = int(spec["required_queue_depth"])
    minimum_achieved = float(spec["minimum_achieved_fraction"])
    minimum_overlap = float(spec.get("minimum_usable_overlap_fraction", 0.75))
    minimum_events = int(spec.get("minimum_events_per_stratum_phase", 32))
    minimum_requests = int(spec.get("minimum_unique_requests_per_stratum_phase", 4))
    req = contract["decision_gate"]["campaign_requirements"]
    strata = list(dict.fromkeys([str(x) for x in req["required_strata"]] + [str(x) for x in req.get("required_ordinary_strata", [])]))
    phases = [str(x) for x in req.get("required_phases", [])]
    required_cells = {(stratum, phase) for stratum in strata for phase in phases}

    events = measurement.get("events") or []
    if not isinstance(events, list):
        raise H3Error("storage scheduler measurement events must be a list")
    normalized: list[dict[str, Any]] = []
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, raw in enumerate(events):
        if not isinstance(raw, dict):
            raise H3Error(f"storage scheduler event {index} is not an object")
        request_id = str(raw.get("request_id", "")); object_id = str(raw.get("object_id", ""))
        stratum = str(raw.get("stratum", "")); phase = str(raw.get("phase", ""))
        if not request_id or not object_id or (stratum, phase) not in required_cells:
            raise H3Error(f"storage scheduler event {index} lacks a valid request/object/stratum/phase identity")
        try:
            outstanding = float(raw["observed_outstanding_io"])
            slack = float(raw["available_prefetch_slack_ns"])
            completion = float(raw["storage_completion_latency_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise H3Error(f"storage scheduler event {index} lacks numeric measured fields") from exc
        if not all(math.isfinite(v) for v in (outstanding, slack, completion)) or min(outstanding, slack, completion) < 0:
            raise H3Error(f"storage scheduler event {index} contains invalid measured values")
        node = {
            "request_id": request_id, "object_id": object_id, "stratum": stratum, "phase": phase,
            "observed_outstanding_io": outstanding, "available_prefetch_slack_ns": slack,
            "storage_completion_latency_ns": completion,
            "completion_within_prefetch_slack": completion <= slack,
        }
        normalized.append(node); cells.setdefault((stratum, phase), []).append(node)

    cell_results: dict[str, Any] = {}; passed = True
    for stratum, phase in sorted(required_cells):
        rows = cells.get((stratum, phase), [])
        unique_requests = {r["request_id"] for r in rows}
        achieved = (sum(min(float(r["observed_outstanding_io"]) / required_qd, 1.0) for r in rows) / len(rows)) if rows else 0.0
        overlap = (sum(bool(r["completion_within_prefetch_slack"]) for r in rows) / len(rows)) if rows else 0.0
        cell_pass = len(rows) >= minimum_events and len(unique_requests) >= minimum_requests and achieved >= minimum_achieved and overlap >= minimum_overlap
        passed = passed and cell_pass
        cell_results[f"{stratum}:{phase}"] = {
            "event_count": len(rows), "unique_request_count": len(unique_requests),
            "achieved_queue_depth_fraction": achieved,
            "storage_completions_within_available_prefetch_slack_fraction": overlap,
            "derived_pass": cell_pass,
        }
    result = {
        "schema_version": 6, "artifact_kind": "fenix_h3_storage_concurrency_evidence",
        "evaluator": "measured_workload_scheduler_raw_event_qd_and_prefetch_slack_feasibility",
        "derived_pass": passed, "required_queue_depth": required_qd,
        "minimum_achieved_fraction": minimum_achieved, "minimum_usable_overlap_fraction": minimum_overlap,
        "minimum_events_per_stratum_phase": minimum_events,
        "minimum_unique_requests_per_stratum_phase": minimum_requests,
        "required_strata": strata, "required_phases": phases, "cell_results": cell_results,
        "observed_measured_event_count": len(normalized),
        "raw_event_fields_derived_not_user_aggregated": True,
        "object_specific_prefetch_slack_measured": True,
        "trace_manifest_sha256": trace_sha, "storage_binding_sha256": binding_sha,
        "source_measurement": str(measurement_path.resolve()),
        "source_measurement_sha256": sha256_file(measurement_path),
        "contract_sha256": sha256_file(contract_path),
    }
    write_json(out_path, result)
    return result
