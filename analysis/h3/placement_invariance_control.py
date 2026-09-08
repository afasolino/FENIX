"""H3 placement-invariance control with run-local identity canonicalization.

This evaluator is analysis-only. It compares ordered prompt, PLE-row, and MoE
routing semantics across the qualified resident and mmap PLE placements while
removing run-local request UUIDs, endpoint URLs, timing fields, and MoE event
batching from the semantic identity. Physical placement itself is bound to the
captured server launch preamble.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from analysis.h3.common import H3Error, iter_jsonl, load_json, parse_layer_id, sha256_file, write_json

PLACEMENT_MODES = ("resident", "mmap")
_MODE_PATTERNS = (
    re.compile(r'"FENIX_PLE_STORAGE_MODE"\s*:\s*"(resident|mmap)"'),
    re.compile(r"\bFENIX_PLE_STORAGE_MODE=(resident|mmap)\b"),
)
_BANK_PATTERNS = (
    re.compile(r'"FENIX_PLE_BANK_MANIFEST"\s*:\s*"([^"]+)"'),
    re.compile(r"\bFENIX_PLE_BANK_MANIFEST=([^\s]+)"),
)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _prompt_semantics(path: Path) -> tuple[str, int]:
    payload = load_json(path)
    rows = payload.get("prompts")
    if not isinstance(rows, list) or not rows:
        raise H3Error(f"{path}: prompt manifest contains no prompts")

    canonical: list[dict[str, Any]] = []
    observed_ordinals: list[int] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise H3Error(f"{path}: prompt manifest contains a non-object row")
        ordinal = int(raw["ordinal"])
        observed_ordinals.append(ordinal)
        declared = raw.get("rendered_prompt_sha256") or raw.get("sha256")
        relative = raw.get("path")
        if relative:
            prompt_path = _resolve(str(relative), path.parent)
            if not prompt_path.is_file():
                raise H3Error(f"{path}: prompt file is missing: {prompt_path}")
            actual = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
            if declared is not None and str(declared) != actual:
                raise H3Error(f"{path}: prompt SHA mismatch for ordinal {ordinal}")
            prompt_sha = actual
        elif declared:
            prompt_sha = str(declared)
        else:
            raise H3Error(f"{path}: prompt ordinal {ordinal} lacks path/SHA")
        token_count = raw.get("rendered_prompt_tokens")
        canonical.append(
            {
                "ordinal": ordinal,
                "prompt_sha256": prompt_sha,
                "prompt_tokens": int(token_count) if token_count is not None else None,
            }
        )

    if sorted(observed_ordinals) != list(range(len(rows))):
        raise H3Error(f"{path}: prompt ordinals are not exactly 0..{len(rows)-1}")
    canonical.sort(key=lambda row: int(row["ordinal"]))
    return _digest(canonical), len(canonical)


def _request_ordinal(mapping: dict[str, int], request_id: str) -> int:
    if request_id not in mapping:
        mapping[request_id] = len(mapping)
    return mapping[request_id]


def _ple_semantics(path: Path) -> tuple[str, int]:
    request_map: dict[str, int] = {}
    canonical: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        canonical.append(
            {
                "request_ordinal": _request_ordinal(request_map, request_id),
                "token_position": int(row["token_position"]),
                "ple_head": int(row["ple_head"]),
                "physical_row_id": int(row["physical_row_id"]),
                "bytes": int(row["bytes"]),
                "phase": str(row.get("phase", "unknown")),
            }
        )
    if not canonical:
        raise H3Error(f"{path}: PLE normalized trace is empty")
    return _digest(canonical), len(request_map)


def _moe_semantics(path: Path, topk: int) -> tuple[str, int]:
    request_map: dict[str, int] = {}
    canonical: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        selected = [int(value) for value in row.get("selected_expert_ids", [])]
        if not selected or len(selected) % int(topk):
            raise H3Error(f"{path}: selected_expert_ids is not a non-empty multiple of top-k")
        request_ordinal = _request_ordinal(request_map, str(row["request_id"]))
        layer = parse_layer_id(row["layer"])
        for begin in range(0, len(selected), int(topk)):
            chunk = selected[begin : begin + int(topk)]
            canonical.append(
                {
                    "request_ordinal": request_ordinal,
                    "layer": layer,
                    "selected_expert_ids": chunk,
                    "phase": str(row.get("phase", "unknown")),
                    "trace_scope": row.get("trace_scope"),
                }
            )
    if not canonical:
        raise H3Error(f"{path}: MoE normalized trace is empty")
    return _digest(canonical), len(request_map)


def _launch_binding(server_log: Path, declared_mode: str) -> dict[str, Any]:
    if declared_mode not in PLACEMENT_MODES:
        raise H3Error(f"unsupported PLE placement mode {declared_mode!r}")
    if not server_log.is_file():
        raise H3Error(f"placement server log is missing: {server_log}")
    text = server_log.read_text(errors="replace")
    observed_modes: set[str] = set()
    for pattern in _MODE_PATTERNS:
        observed_modes.update(pattern.findall(text))
    if observed_modes != {declared_mode}:
        raise H3Error(
            f"server log placement mode mismatch for {server_log}: "
            f"observed={sorted(observed_modes)} declared={declared_mode}"
        )
    banks: set[str] = set()
    for pattern in _BANK_PATTERNS:
        banks.update(pattern.findall(text))
    if declared_mode == "mmap" and not banks:
        raise H3Error("mmap placement server log does not bind FENIX_PLE_BANK_MANIFEST")
    if declared_mode == "resident" and banks:
        raise H3Error("resident placement unexpectedly declares FENIX_PLE_BANK_MANIFEST")
    return {
        "mode": declared_mode,
        "server_log": str(server_log),
        "server_log_sha256": sha256_file(server_log),
        "ple_bank_manifests": sorted(banks),
    }


def _validate_case_evidence(
    evidence_path: Path,
    stratum: str,
    prompts: Path,
    ple: Path,
    moe: Path,
) -> dict[str, Any]:
    evidence = load_json(evidence_path)
    if evidence.get("trace_valid") is not True or evidence.get("repository_clean") is not True:
        raise H3Error(f"{evidence_path}: placement case is not clean trace-valid evidence")
    case = evidence.get("case") or {}
    if str(case.get("stratum")) != stratum:
        raise H3Error(f"{evidence_path}: placement case stratum mismatch")
    if int(case.get("concurrency", 0)) != 1:
        raise H3Error(f"{evidence_path}: placement control requires concurrency=1")
    hashes = evidence.get("artifacts_sha256") or {}
    for path in (prompts, ple, moe):
        if path.parent != evidence_path.parent:
            raise H3Error("placement case semantic files must belong to the evidence case directory")
        expected = hashes.get(path.name)
        if expected is None:
            raise H3Error(f"{evidence_path}: evidence does not hash-bind {path.name}")
        if sha256_file(path) != str(expected):
            raise H3Error(f"{evidence_path}: hash drift for {path.name}")
    launch = evidence.get("launch") or {}
    runtime_lane = evidence.get("runtime_lane") or {}
    return {
        "repository_commit": str(evidence.get("repository_commit")),
        "runtime_image_id": str(launch.get("runtime_image_id")),
        "runtime_lane": runtime_lane,
        "source_manifest_sha256": evidence.get("source_manifest_sha256"),
        "frozen_corpus_sha256": evidence.get("frozen_corpus_sha256"),
        "evidence_sha256": sha256_file(evidence_path),
    }


def build_placement_invariance_control(
    input_path: Path,
    contract_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    payload = load_json(input_path)
    contract = load_json(contract_path)
    spec = contract["prerequisites"]["placement_invariance"]
    required_strata = [str(value) for value in spec.get("required_strata", [])]
    placements = payload.get("placements") or []
    if len(placements) < int(spec.get("minimum_placements", 2)):
        raise H3Error("placement invariance has too few physical controls")

    declared_modes = [str(node.get("ple_storage_mode", "")) for node in placements]
    if set(declared_modes) != set(PLACEMENT_MODES):
        raise H3Error(
            "placement control must contain exactly the resident and mmap PLE modes; "
            f"observed={declared_modes}"
        )
    if len(declared_modes) != len(set(declared_modes)):
        raise H3Error("placement control contains duplicate physical modes")

    topk = int(contract["geometry"]["experts_per_token"])
    resolved: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for placement in placements:
        name = str(placement.get("name", ""))
        mode = str(placement.get("ple_storage_mode", ""))
        server_log_raw = placement.get("server_log")
        cases = placement.get("cases") or {}
        if not name or not server_log_raw:
            raise H3Error("placement control lacks name/server_log")
        if sorted(cases) != sorted(required_strata):
            raise H3Error(f"placement {name} case set differs from contract-required strata")
        launch = _launch_binding(_resolve(str(server_log_raw), input_path.parent), mode)

        case_rows: dict[str, Any] = {}
        for stratum in required_strata:
            node = cases[stratum]
            paths: dict[str, Path] = {}
            for field in ("prompts", "ple_normalized", "moe_normalized", "evidence"):
                raw = node.get(field)
                if not raw:
                    raise H3Error(f"placement {name}/{stratum} lacks {field}")
                path = _resolve(str(raw), input_path.parent)
                if not path.is_file():
                    raise H3Error(f"placement {name}/{stratum} file missing: {path}")
                paths[field] = path

            provenance = _validate_case_evidence(
                paths["evidence"], stratum, paths["prompts"], paths["ple_normalized"], paths["moe_normalized"]
            )
            prompt_sha, prompt_requests = _prompt_semantics(paths["prompts"])
            ple_sha, ple_requests = _ple_semantics(paths["ple_normalized"])
            moe_sha, moe_requests = _moe_semantics(paths["moe_normalized"], topk)
            if len({prompt_requests, ple_requests, moe_requests}) != 1:
                raise H3Error(
                    f"placement {name}/{stratum} request-count mismatch: "
                    f"prompts={prompt_requests} ple={ple_requests} moe={moe_requests}"
                )
            case_rows[stratum] = {
                "paths": {key: str(value) for key, value in paths.items()},
                "raw_sha256": {key: sha256_file(value) for key, value in paths.items()},
                "semantic_sha256": {
                    "prompts": prompt_sha,
                    "ple_normalized": ple_sha,
                    "moe_normalized": moe_sha,
                },
                "request_count": prompt_requests,
                "source_provenance": provenance,
            }
            provenance_rows.append(provenance)
        resolved.append({"name": name, "ple_storage_mode": mode, "launch_binding": launch, "cases": case_rows})

    # Both controls must execute the same source/runtime stack. The only intended
    # difference is the resident versus mmap PLE placement.
    invariant_keys = ("repository_commit", "runtime_image_id", "runtime_lane", "source_manifest_sha256", "frozen_corpus_sha256")
    reference_provenance = provenance_rows[0]
    for row in provenance_rows[1:]:
        for key in invariant_keys:
            if row.get(key) != reference_provenance.get(key):
                raise H3Error(f"placement controls differ in non-placement provenance field {key}")

    reference = resolved[0]
    mismatches: list[dict[str, Any]] = []
    for placement in resolved[1:]:
        for stratum in required_strata:
            for field in ("prompts", "ple_normalized", "moe_normalized"):
                expected = reference["cases"][stratum]["semantic_sha256"][field]
                observed = placement["cases"][stratum]["semantic_sha256"][field]
                if observed != expected:
                    mismatches.append(
                        {
                            "placement": placement["name"],
                            "stratum": stratum,
                            "field": field,
                            "reference_sha256": expected,
                            "observed_sha256": observed,
                        }
                    )

    result = {
        "schema_version": 1,
        "artifact_kind": "fenix_h3_placement_invariance_evidence",
        "evaluator": "ordered_routing_and_ple_semantic_identity_across_physical_placements",
        "derived_pass": not mismatches,
        "minimum_placements": int(spec.get("minimum_placements", 2)),
        "required_strata": required_strata,
        "required_physical_modes": list(PLACEMENT_MODES),
        "request_identity_semantics": "run-local request UUID canonicalized to first-seen request ordinal",
        "prompt_semantics": "ordered prompt bytes/token counts; endpoint/tokenize URL metadata excluded",
        "moe_semantics": "ordered atomic top-k routing selections; runtime event batching and timing excluded",
        "physical_placement_binding": "server launch preamble FENIX_PLE_STORAGE_MODE; mmap additionally binds FENIX_PLE_BANK_MANIFEST",
        "source_runtime_provenance_invariant": True,
        "placements": resolved,
        "mismatches": mismatches,
        "contract_sha256": sha256_file(contract_path),
        "input_sha256": sha256_file(input_path),
    }
    write_json(out_path, result)
    return result
