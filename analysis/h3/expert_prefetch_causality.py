"""Rootless causal qualification of exact expert prefetch for H3.

The experiment measures the native software interval between exact routed expert
IDs becoming available on the host and the runtime being ready to dispatch the
expert backend. The comparison deliberately uses the *fastest* QD32 expert
storage coefficient observed anywhere in the frozen H3 campaign. If even that
one-expert transfer cannot fit in the native interval, arbitrary exact expert
prefetch cannot be treated as fully hidden for this pinned host-driven runtime.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from analysis.h3.common import H3Error, iter_jsonl, load_json, parse_layer_id, sha256_file, write_json


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        raise H3Error("cannot compute percentile of empty sample")
    ordered = sorted(int(v) for v in values)
    index = int(round((len(ordered) - 1) * float(fraction)))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _load_amendment(path: Path, contract_path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("artifact_kind") != "fenix_h3_analysis_amendment_contract":
        raise H3Error("unexpected H3 causality amendment artifact")
    if payload.get("base_contract_sha256") != sha256_file(contract_path):
        raise H3Error("causality amendment is not bound to the supplied H3 contract")
    cfg = payload.get("causality") or {}
    if cfg.get("all_expert_perfect_prefetch_removed_only_if_causality_passes") is not True:
        raise H3Error("causality amendment does not fail closed")
    if cfg.get("gap_falsification_only") is not True:
        raise H3Error("causality amendment must be gap-falsification only")
    return payload


def _fastest_frozen_expert_qd32(
    fio_root: Path,
    frozen_head: str,
) -> dict[str, Any]:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for path in sorted(fio_root.rglob("fio-summary.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if payload.get("artifact_kind") != "fenix_h3_fio_calibration":
            continue
        identity = payload.get("execution_repository") or {}
        if identity.get("head") != frozen_head:
            continue
        node = (payload.get("class_coefficients") or {}).get("expert:qd32")
        if not isinstance(node, dict):
            continue
        ci = node.get("ns_per_byte_ci95")
        if not isinstance(ci, list) or len(ci) != 2:
            continue
        low = float(ci[0])
        if low <= 0:
            continue
        candidates.append((low, path, payload))
    if not candidates:
        raise H3Error("no frozen e08 expert:qd32 fio calibration found")
    low, path, payload = min(candidates, key=lambda item: item[0])
    return {
        "ns_per_byte_ci95_lower": low,
        "artifact": str(path.resolve()),
        "sha256": sha256_file(path),
        "qualified_fio_summary_count": len(candidates),
        "fio_tool": payload.get("fio_tool"),
        "campaign_fingerprint": {
            "storage_binding_sha256": (payload.get("provenance") or {}).get("storage_binding_sha256"),
            "device_major_minor": (payload.get("provenance") or {}).get("device_major_minor"),
        },
    }


def build_expert_prefetch_causality(
    normalized_moe_paths: Iterable[Path],
    fio_root: Path,
    contract_path: Path,
    amendment_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    contract = load_json(contract_path)
    amendment = _load_amendment(amendment_path, contract_path)
    cfg = amendment["causality"]
    frozen_head = str(amendment["frozen_measurement_head"])
    required_phases = [str(value) for value in cfg["required_phases"]]
    required_layers = int(cfg["required_layers"])
    min_events = int(cfg["minimum_events_per_phase"])

    paths = [Path(path).resolve() for path in normalized_moe_paths]
    if not paths:
        raise H3Error("causality experiment requires at least one normalized MoE trace")

    slacks_by_phase: dict[str, list[int]] = defaultdict(list)
    layers_by_phase: dict[str, set[int]] = defaultdict(set)
    records = 0
    source_hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise H3Error(f"normalized MoE trace is missing: {path}")
        source_hashes[str(path)] = sha256_file(path)
        for row in iter_jsonl(path):
            if row.get("trace_scope") != "router_all_layers":
                raise H3Error(f"{path}: causality trace is not router_all_layers")
            if row.get("fenix_h3_prefetch_semantics") != (
                "exact_expert_ids_host_ready_to_native_dispatch_ready_no_prediction"
            ):
                raise H3Error(f"{path}: missing H3 causal instrumentation marker")
            phase = str(row.get("phase"))
            if phase not in required_phases:
                continue
            try:
                ready = int(row["fenix_h3_host_ids_ready_ns"])
                dispatch = int(row["fenix_h3_dispatch_call_ready_ns"])
            except (KeyError, TypeError, ValueError) as exc:
                raise H3Error(f"{path}: invalid H3 causal timestamps") from exc
            if dispatch < ready:
                raise H3Error(f"{path}: dispatch-ready precedes host expert-ID readiness")
            slack = dispatch - ready
            slacks_by_phase[phase].append(slack)
            layers_by_phase[phase].add(parse_layer_id(row["layer"]))
            records += 1

    fio = _fastest_frozen_expert_qd32(fio_root.resolve(), frozen_head)
    geometry = contract["geometry"]
    expert_transfer_bytes = int(geometry["expert_storage_stride_bytes"])
    fastest_one_expert_ns = expert_transfer_bytes * float(fio["ns_per_byte_ci95_lower"])

    phase_stats: dict[str, Any] = {}
    failures: list[str] = []
    all_slacks: list[int] = []
    expected_layers = set(range(required_layers))
    for phase in required_phases:
        values = slacks_by_phase.get(phase, [])
        layers = layers_by_phase.get(phase, set())
        if len(values) < min_events:
            failures.append(f"{phase}:events={len(values)}<{min_events}")
        if layers != expected_layers:
            failures.append(
                f"{phase}:layers={len(layers)}/{required_layers}"
            )
        if values:
            phase_stats[phase] = {
                "events": len(values),
                "layers": sorted(layers),
                "minimum_slack_ns": min(values),
                "median_slack_ns": median(values),
                "p95_slack_ns": _percentile(values, 0.95),
                "maximum_slack_ns": max(values),
            }
            all_slacks.extend(values)
        else:
            phase_stats[phase] = {"events": 0, "layers": []}

    max_slack = max(all_slacks) if all_slacks else None
    if max_slack is None:
        failures.append("no_required_phase_events")
    elif float(max_slack) >= fastest_one_expert_ns:
        failures.append(
            "native_host_ready_to_dispatch_window_can_fit_fastest_measured_one_expert_transfer"
        )

    result = {
        "schema_version": 1,
        "artifact_kind": "fenix_h3_expert_prefetch_causality",
        "derived_pass": not failures,
        "evaluator": "measured_host_expert_id_ready_to_native_dispatch_vs_fastest_frozen_qd32_expert_transfer",
        "failures": failures,
        "records_used": records,
        "required_phases": required_phases,
        "required_layers": required_layers,
        "phase_stats": phase_stats,
        "maximum_native_prefetch_slack_ns": max_slack,
        "fastest_measured_one_expert_transfer_ns_lower": fastest_one_expert_ns,
        "slack_over_fastest_one_expert_service": (
            float(max_slack) / fastest_one_expert_ns
            if max_slack is not None and fastest_one_expert_ns > 0 else None
        ),
        "expert_transfer_bytes": expert_transfer_bytes,
        "storage_queue_depth": 32,
        "fastest_frozen_expert_fio": fio,
        "exact_expert_prefetch_before_host_ids_requires_prediction": True,
        "prediction_or_speculation_measured": False,
        "ple_perfect_prefetch_remains_allowed": True,
        "all_expert_perfect_prefetch_falsified_for_pinned_host_driven_runtime": not failures,
        "source_normalized_moe_sha256": source_hashes,
        "contract_sha256": sha256_file(contract_path),
        "analysis_amendment_sha256": sha256_file(amendment_path),
        "frozen_measurement_head": frozen_head,
        "runtime_revision": amendment["runtime_revision"],
        "claim_boundary": amendment["claim_boundary"],
    }
    write_json(out_path, result)
    return result
