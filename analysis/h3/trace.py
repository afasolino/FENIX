"""Validate and lazily merge the authoritative no-prefix H1 traces for H3."""
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from analysis.h3.common import H3Error, iter_jsonl, load_json, parse_layer_id, sha256_file, write_json


@dataclass(frozen=True)
class Geometry:
    layers: int
    experts: int
    topk: int
    ple_rows: int
    ple_heads: int
    ple_row_bytes: int
    expert_bytes: int
    storage_page_bytes: int
    expert_stride_bytes: int


@dataclass(frozen=True)
class TraceEvent:
    timestamp_ns: int  # ordering only; never an edge-platform timing measurement
    source_order: int
    request_id: str
    phase: str
    kind: str
    layer: int | None
    token_position: int
    object_ids: tuple[int, ...]


@dataclass(frozen=True)
class TraceEpoch:
    timestamp_ns: int
    events: tuple[TraceEvent, ...]


def geometry_from_files(campaign_path: Path, h3_contract_path: Path) -> Geometry:
    model = load_json(campaign_path).get("model")
    if not isinstance(model, dict):
        raise H3Error("campaign model geometry missing")
    spec = load_json(h3_contract_path)["geometry"]
    geometry = Geometry(
        layers=int(model["num_hidden_layers"]),
        experts=int(model["num_experts"]),
        topk=int(model["experts_per_token"]),
        ple_rows=int(model["ple_addressable_rows"]),
        ple_heads=int(spec["ple_heads"]),
        ple_row_bytes=int(spec["ple_row_bytes"]),
        expert_bytes=int(spec["expert_useful_bytes"]),
        storage_page_bytes=int(spec["storage_page_bytes"]),
        expert_stride_bytes=int(spec["expert_storage_stride_bytes"]),
    )
    expected = {
        "layers": geometry.layers,
        "experts_per_layer": geometry.experts,
        "experts_per_token": geometry.topk,
        "ple_addressable_rows": geometry.ple_rows,
    }
    for key, observed in expected.items():
        if int(spec[key]) != observed:
            raise H3Error(f"geometry drift for {key}: contract={spec[key]} campaign={observed}")
    expected_ple_bytes = geometry.ple_rows * geometry.ple_row_bytes
    if int(spec.get("ple_data_bytes", expected_ple_bytes)) != expected_ple_bytes:
        raise H3Error("PLE checkpoint byte geometry is inconsistent")
    return geometry


def _clients(case_dir: Path) -> list[dict[str, Any]]:
    records = [row for row in iter_jsonl(case_dir / "client.jsonl") if "error" not in row]
    if not records:
        raise H3Error(f"{case_dir}: no successful client records")
    records.sort(key=lambda row: int(row["ordinal"]))
    ordinals = [int(row["ordinal"]) for row in records]
    if len(ordinals) != len(set(ordinals)):
        raise H3Error(f"{case_dir}: duplicate client ordinal")
    request_ids = [str(row["request_id"]) for row in records]
    if len(request_ids) != len(set(request_ids)):
        raise H3Error(f"{case_dir}: duplicate request_id")
    return records


def _expected_tokens(clients: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(row["request_id"]): int(row["prompt_tokens"]) + max(int(row["completion_tokens"]) - 1, 0)
        for row in clients
    }


def _validate_evidence(
    case_dir: Path,
    stratum: str,
    expected_repository_commit: str,
    expected_concurrency: int,
    expected_correlation: str,
) -> dict[str, Any]:
    evidence = load_json(case_dir / "evidence.json")
    if evidence.get("trace_valid") is not True:
        raise H3Error(f"{case_dir.name}: trace_valid is not true")
    if evidence.get("repository_clean") is not True:
        raise H3Error(f"{case_dir.name}: source repository was not clean")
    if str(evidence.get("repository_commit")) != expected_repository_commit:
        raise H3Error(
            f"{case_dir.name}: source repository commit drift: "
            f"{evidence.get('repository_commit')} != {expected_repository_commit}"
        )
    case = evidence.get("case") or {}
    if case.get("stratum") != stratum:
        raise H3Error(f"{case_dir.name}: stratum mismatch")
    if int(case.get("concurrency", 0)) != expected_concurrency:
        raise H3Error(f"{case_dir.name}: H3 source requires concurrency={expected_concurrency}")
    if case.get("correlation_mode") != expected_correlation:
        raise H3Error(f"{case_dir.name}: {expected_correlation} required")
    if not evidence.get("source_manifest_sha256"):
        raise H3Error(f"{case_dir.name}: source manifest provenance is missing")
    if not (evidence.get("launch") or {}).get("runtime_image_id"):
        raise H3Error(f"{case_dir.name}: runtime image provenance is missing")
    hashes = evidence.get("artifacts_sha256") or {}
    required = {"client.jsonl", "ple_normalized.jsonl", "moe_normalized.jsonl"}
    if not required.issubset(set(hashes)):
        raise H3Error(f"{case_dir.name}: evidence artifact hash set is incomplete")
    for relative, expected in hashes.items():
        path = case_dir / relative
        if not path.is_file():
            raise H3Error(f"{case_dir.name}: missing hashed source artifact {relative}")
        if sha256_file(path) != str(expected):
            raise H3Error(f"{case_dir.name}: source SHA drift for {relative}")
    return evidence


def _validate_ple(case_dir: Path, expected: dict[str, int], geometry: Geometry) -> dict[str, Any]:
    per_request_tokens: dict[str, int] = defaultdict(int)
    seen_positions: dict[str, set[int]] = defaultdict(set)
    current: tuple[str, int] | None = None
    heads: set[int] = set()
    rows = 0
    last_ts = -1

    def finish() -> None:
        if current is None:
            return
        if heads != set(range(geometry.ple_heads)):
            raise H3Error(f"{case_dir.name}:{current}: PLE heads are not exactly 0..{geometry.ple_heads-1}")
        request_id, token_position = current
        if token_position in seen_positions[request_id]:
            raise H3Error(f"{case_dir.name}:{current}: duplicate PLE token position")
        seen_positions[request_id].add(token_position)
        per_request_tokens[request_id] += 1

    for record in iter_jsonl(case_dir / "ple_normalized.jsonl"):
        request_id = str(record["request_id"])
        if request_id not in expected:
            raise H3Error(f"{case_dir.name}: PLE request outside client stream")
        timestamp = int(record["address_known_ns"])
        if timestamp < last_ts:
            raise H3Error(f"{case_dir.name}: PLE trace is not timestamp monotonic")
        last_ts = timestamp
        key = (request_id, int(record["token_position"]))
        if current is not None and key != current:
            finish()
            heads = set()
        current = key
        head = int(record["ple_head"])
        if head in heads:
            raise H3Error(f"{case_dir.name}:{key}: duplicate PLE head {head}")
        heads.add(head)
        row = int(record["physical_row_id"])
        if not 0 <= row < geometry.ple_rows:
            raise H3Error(f"{case_dir.name}: PLE row outside geometry")
        if int(record["bytes"]) != geometry.ple_row_bytes:
            raise H3Error(f"{case_dir.name}: PLE row width drift")
        rows += 1
    finish()
    for request_id, count in expected.items():
        if per_request_tokens[request_id] != count:
            raise H3Error(
                f"{case_dir.name}:{request_id}: PLE autoregressive coverage "
                f"{per_request_tokens[request_id]} != {count}"
            )
        if seen_positions[request_id] != set(range(count)):
            raise H3Error(
                f"{case_dir.name}:{request_id}: PLE token positions are not exactly 0..{count-1}"
            )
    return {"row_records": rows, "model_tokens": sum(per_request_tokens.values())}


def _validate_moe(case_dir: Path, expected: dict[str, int], geometry: Geometry) -> dict[str, Any]:
    tokens: dict[str, list[int]] = {request_id: [0] * geometry.layers for request_id in expected}
    last_ts = -1
    runtime_records = 0
    selections = 0
    scopes: set[str] = set()
    for record in iter_jsonl(case_dir / "moe_normalized.jsonl"):
        request_id = str(record["request_id"])
        if request_id not in expected:
            raise H3Error(f"{case_dir.name}: MoE request outside client stream")
        timestamp = int(record["timestamp_ns"])
        if timestamp < last_ts:
            raise H3Error(f"{case_dir.name}: MoE trace is not timestamp monotonic")
        last_ts = timestamp
        layer = parse_layer_id(record["layer"])
        if not 0 <= layer < geometry.layers:
            raise H3Error(f"{case_dir.name}: layer outside geometry")
        selected = [int(value) for value in record.get("selected_expert_ids", [])]
        if not selected or len(selected) % geometry.topk:
            raise H3Error(f"{case_dir.name}: selected-expert width is not a multiple of top-k")
        for begin in range(0, len(selected), geometry.topk):
            chunk = selected[begin : begin + geometry.topk]
            if len(chunk) != geometry.topk or len(set(chunk)) != geometry.topk:
                raise H3Error(f"{case_dir.name}: malformed atomic top-k expert selection")
            if any(not 0 <= expert < geometry.experts for expert in chunk):
                raise H3Error(f"{case_dir.name}: expert outside geometry")
            tokens[request_id][layer] += 1
        selections += len(selected)
        runtime_records += 1
        if record.get("trace_scope") is not None:
            scopes.add(str(record["trace_scope"]))
    if scopes and scopes != {"router_all_layers"}:
        raise H3Error(f"{case_dir.name}: unexpected trace scope {sorted(scopes)}")
    for request_id, expected_count in expected.items():
        bad = [layer for layer, count in enumerate(tokens[request_id]) if count != expected_count]
        if bad:
            raise H3Error(f"{case_dir.name}:{request_id}: MoE coverage mismatch in layers {bad[:8]}")
    return {"runtime_records": runtime_records, "expert_selections": selections}


def build_manifest(
    trace_root: Path,
    robustness_contract_path: Path,
    campaign_path: Path,
    h3_contract_path: Path,
    out_path: Path,
    execution_verification_path: Path | None = None,
) -> dict[str, Any]:
    robustness = load_json(robustness_contract_path)
    if robustness.get("artifact_kind") != "fenix_h1_h2_workload_robustness_contract":
        raise H3Error("unexpected H1/H2 robustness-contract artifact kind")
    h3_contract = load_json(h3_contract_path)
    source = h3_contract["source_trace"]
    order = list(robustness["trace"]["strata_order"])
    expected_order = list(h3_contract["workload_policy"]["strata"])
    if order != expected_order:
        raise H3Error(f"H3 strata must match the predeclared authoritative set/order: {order} != {expected_order}")
    geometry = geometry_from_files(campaign_path, h3_contract_path)
    if execution_verification_path is None:
        raise H3Error("H3 primary trace requires an explicit execution-policy artifact")
    execution = load_json(execution_verification_path)
    if execution.get("artifact_kind") != "fenix_h1_h2_trace_execution_policy":
        raise H3Error("unexpected H1/H2 trace-execution-policy artifact kind")
    if str(execution.get("prefix_caching")).lower() != str(source["prefix_caching"]).lower():
        raise H3Error("H3 primary trace requires prefix caching disabled")
    if int(execution.get("concurrency", -1)) != int(source["concurrency"]):
        raise H3Error("H3 execution policy concurrency drift")

    cases = []
    provenance_fingerprint: set[tuple[Any, Any, Any]] = set()
    for stratum in order:
        case_dir = trace_root / f"s-{stratum}-r01"
        if not case_dir.is_dir():
            raise H3Error(f"missing source case {case_dir}")
        evidence = _validate_evidence(
            case_dir,
            stratum,
            str(source["repository_commit"]),
            int(source["concurrency"]),
            str(source["correlation_mode"]),
        )
        clients = _clients(case_dir)
        stratum_contract = (robustness.get("strata") or {}).get(stratum) or {}
        declared_requests = stratum_contract.get("requests")
        if declared_requests is None:
            raise H3Error(f"{stratum}: authoritative robustness contract lacks request count")
        if len(clients) != int(declared_requests):
            raise H3Error(
                f"{stratum}: successful request count {len(clients)} != declared {declared_requests}"
            )
        expected = _expected_tokens(clients)
        ple = _validate_ple(case_dir, expected, geometry)
        moe = _validate_moe(case_dir, expected, geometry)
        provenance_fingerprint.add(
            (
                evidence.get("repository_commit"),
                (evidence.get("launch") or {}).get("runtime_image_id"),
                evidence.get("source_manifest_sha256"),
            )
        )
        cases.append(
            {
                "stratum": stratum,
                "case_dir": str(case_dir.resolve()),
                "requests": len(clients),
                "model_tokens": ple["model_tokens"],
                "ple": ple,
                "moe": moe,
                "source_sha256": {
                    "evidence.json": sha256_file(case_dir / "evidence.json"),
                    "client.jsonl": sha256_file(case_dir / "client.jsonl"),
                    "ple_normalized.jsonl": sha256_file(case_dir / "ple_normalized.jsonl"),
                    "moe_normalized.jsonl": sha256_file(case_dir / "moe_normalized.jsonl"),
                },
            }
        )
    if len(provenance_fingerprint) != 1:
        raise H3Error("cross-stratum source provenance differs")
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_trace_manifest",
        "source_timestamps_semantics": "ordering_only_never_edge_timing",
        "primary_replay_policy": "per_stratum_cold_start",
        "source_trace_contract": source,
        "geometry": geometry.__dict__,
        "cases": cases,
        "provenance_fingerprint": [list(item) for item in provenance_fingerprint],
        "inputs": {
            "robustness_contract_sha256": sha256_file(robustness_contract_path),
            "campaign_sha256": sha256_file(campaign_path),
            "h3_contract_sha256": sha256_file(h3_contract_path),
            "execution_verification_sha256": sha256_file(execution_verification_path),
        },
    }
    write_json(out_path, result)
    return result


def _iter_ple_events(case_dir: Path, geometry: Geometry) -> Iterator[TraceEvent]:
    """Yield one PLE-row event per normalized record, preserving its own timestamp.

    The previous candidate incorrectly packed all 16 heads of a token at the
    first head timestamp. H3 atomicity is defined only by *equal* source
    timestamps, so each head must retain address_known_ns independently.
    """
    source_order = 0
    last_ts = -1
    for record in iter_jsonl(case_dir / "ple_normalized.jsonl"):
        timestamp = int(record["address_known_ns"])
        if timestamp < last_ts:
            raise H3Error(f"{case_dir.name}: PLE event stream is not monotonic")
        last_ts = timestamp
        row = int(record["physical_row_id"])
        if not 0 <= row < geometry.ple_rows:
            raise H3Error(f"{case_dir.name}: PLE event row outside geometry")
        yield TraceEvent(
            timestamp,
            source_order,
            str(record["request_id"]),
            str(record.get("phase", "unknown")),
            "ple",
            None,
            int(record["token_position"]),
            (row,),
        )
        source_order += 1


def _iter_moe_events(case_dir: Path, geometry: Geometry) -> Iterator[TraceEvent]:
    positions: dict[tuple[str, int], int] = defaultdict(int)
    source_order = 1_000_000_000
    last_ts = -1
    for record in iter_jsonl(case_dir / "moe_normalized.jsonl"):
        timestamp = int(record["timestamp_ns"])
        if timestamp < last_ts:
            raise H3Error(f"{case_dir.name}: MoE event stream is not monotonic")
        last_ts = timestamp
        request_id = str(record["request_id"])
        layer = parse_layer_id(record["layer"])
        selected = [int(value) for value in record["selected_expert_ids"]]
        for begin in range(0, len(selected), geometry.topk):
            chunk = tuple(selected[begin : begin + geometry.topk])
            if len(chunk) != geometry.topk:
                raise H3Error(f"{case_dir.name}: malformed MoE chunk during replay")
            key = (request_id, layer)
            position = positions[key]
            positions[key] += 1
            yield TraceEvent(
                timestamp,
                source_order,
                request_id,
                str(record.get("phase", "unknown")),
                "expert",
                layer,
                position,
                chunk,
            )
            source_order += 1


def iter_case_epochs(case_dir: Path, geometry: Geometry, phase_filter: str | None = None) -> Iterator[TraceEpoch]:
    merged = heapq.merge(
        _iter_ple_events(case_dir, geometry),
        _iter_moe_events(case_dir, geometry),
        key=lambda event: (event.timestamp_ns, event.source_order),
    )
    timestamp: int | None = None
    bucket: list[TraceEvent] = []
    for event in merged:
        if phase_filter is not None and event.phase != phase_filter:
            continue
        if timestamp is not None and event.timestamp_ns != timestamp:
            yield TraceEpoch(timestamp, tuple(bucket))
            bucket = []
        timestamp = event.timestamp_ns
        bucket.append(event)
    if timestamp is not None:
        yield TraceEpoch(timestamp, tuple(bucket))
