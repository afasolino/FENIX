#!/usr/bin/env python3
"""H1: characterize measured Qwen3.8 conditional-state working sets.

This analysis consumes only exact C=1 trace cases.  It is deliberately
fail-closed about MoE coverage: each request must expose the same routed-token
count at every transformer layer, and that count must equal the number of PLE
token positions observed for the same request.  This catches the historical
decode-only MoE instrumentation path before any H1 conclusion is drawn.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from analysis.expert_locality import parse_layer_id
from analysis.process_ple_trace import load_jsonl


class H1AnalysisError(ValueError):
    """Raised when a trace case cannot support H1 analysis."""


@dataclass(frozen=True)
class CaseGeometry:
    num_hidden_layers: int
    num_experts: int
    experts_per_token: int
    ple_addressable_rows: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_campaign_geometry(path: Path) -> CaseGeometry:
    payload = json.loads(path.read_text())
    model = payload.get("model")
    if not isinstance(model, dict):
        raise H1AnalysisError("campaign model geometry is missing")
    try:
        geometry = CaseGeometry(
            num_hidden_layers=int(model["num_hidden_layers"]),
            num_experts=int(model["num_experts"]),
            experts_per_token=int(model["experts_per_token"]),
            ple_addressable_rows=int(model["ple_addressable_rows"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise H1AnalysisError("campaign model geometry is incomplete") from exc
    if (
        geometry.num_hidden_layers <= 0
        or geometry.num_experts <= 0
        or geometry.experts_per_token <= 0
        or geometry.ple_addressable_rows <= 0
    ):
        raise H1AnalysisError("campaign model geometry must be positive")
    return geometry


def derive_expert_slot_bytes(
    moe_records: Iterable[dict[str, Any]],
    *,
    explicit_expert_slot_bytes: int | None = None,
) -> tuple[int, str]:
    """Return one complete layer-expert slot size.

    The runtime trace records bytes copied when a GPU hot-cache slot is filled.
    Every transferred expert in the pinned WNA16 representation must have one
    identical complete slot size.  Selection-only prefill events carry zero
    transfer bytes and are intentionally ignored here.
    """

    if explicit_expert_slot_bytes is not None:
        if explicit_expert_slot_bytes <= 0:
            raise H1AnalysisError("explicit expert slot bytes must be positive")
        return explicit_expert_slot_bytes, "explicit_override"

    observed: set[int] = set()
    for record in moe_records:
        transferred = record.get("transfer_expert_ids", [])
        if not isinstance(transferred, list) or not transferred:
            continue
        raw_bytes = record.get("transfer_bytes")
        if (
            not isinstance(raw_bytes, int)
            or isinstance(raw_bytes, bool)
            or raw_bytes <= 0
            or raw_bytes % len(transferred)
        ):
            raise H1AnalysisError(
                "MoE transfer trace cannot derive a uniform expert slot size"
            )
        observed.add(raw_bytes // len(transferred))
    if len(observed) != 1:
        raise H1AnalysisError(
            "expected exactly one measured expert slot size; "
            f"observed={sorted(observed)}"
        )
    return next(iter(observed)), "measured_gpu_hot_cache_fill"


def _uniform_row_bytes(ple_records: list[dict[str, Any]]) -> int:
    observed = {
        int(record["bytes"])
        for record in ple_records
        if record.get("bytes") is not None
    }
    if len(observed) != 1:
        raise H1AnalysisError(
            f"expected one measured PLE row width; observed={sorted(observed)}"
        )
    row_bytes = next(iter(observed))
    if row_bytes <= 0:
        raise H1AnalysisError("PLE row width must be positive")
    return row_bytes


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise H1AnalysisError(f"{path}: expected JSON object")
    return value


def _expert_concentration(
    per_layer_counts: list[collections.Counter[int]],
    topk_values: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topk in topk_values:
        layer_fractions: list[float] = []
        for counts in per_layer_counts:
            total = sum(counts.values())
            if total <= 0:
                continue
            selected = sum(value for _, value in counts.most_common(topk))
            layer_fractions.append(selected / total)
        rows.append(
            {
                "topk_experts_per_layer": topk,
                "mean_selection_fraction": (
                    sum(layer_fractions) / len(layer_fractions)
                    if layer_fractions
                    else None
                ),
                "min_selection_fraction": min(layer_fractions)
                if layer_fractions
                else None,
                "max_selection_fraction": max(layer_fractions)
                if layer_fractions
                else None,
            }
        )
    return rows


def analyze_case(
    case_dir: Path,
    campaign_path: Path,
    *,
    topk_values: list[int] | None = None,
    explicit_expert_slot_bytes: int | None = None,
) -> dict[str, Any]:
    geometry = _load_campaign_geometry(campaign_path)
    if topk_values is None:
        topk_values = [16, 32, 64, 128, 256]
    if any(value <= 0 or value > geometry.num_experts for value in topk_values):
        raise H1AnalysisError("expert concentration top-k is outside model geometry")

    evidence_path = case_dir / "evidence.json"
    client_path = case_dir / "client.jsonl"
    ple_path = case_dir / "ple_normalized.jsonl"
    moe_path = case_dir / "moe_normalized.jsonl"
    for path in (evidence_path, client_path, ple_path, moe_path):
        if not path.is_file():
            raise H1AnalysisError(f"required exact-trace artifact is missing: {path}")

    evidence = json.loads(evidence_path.read_text())
    if evidence.get("trace_valid") is not True:
        raise H1AnalysisError("trace case is not marked valid")
    case = evidence.get("case")
    if not isinstance(case, dict):
        raise H1AnalysisError("trace evidence case metadata is missing")
    if case.get("correlation_mode") != "exact_request_correlation":
        raise H1AnalysisError("H1 requires exact request correlation")
    if int(case.get("concurrency", 0)) != 1:
        raise H1AnalysisError("H1 requires concurrency=1")

    clients = [
        record for record in load_jsonl(client_path) if "error" not in record
    ]
    ple_records = load_jsonl(ple_path)
    moe_records = load_jsonl(moe_path)
    if not clients or not ple_records or not moe_records:
        raise H1AnalysisError("exact trace case contains an empty required stream")

    successful_ids = {str(record["request_id"]) for record in clients}
    ple_ids = {str(record["request_id"]) for record in ple_records}
    moe_ids = {str(record["request_id"]) for record in moe_records}
    if ple_ids != successful_ids or moe_ids != successful_ids:
        raise H1AnalysisError(
            "client/PLE/MoE request sets differ: "
            f"client={len(successful_ids)} ple={len(ple_ids)} moe={len(moe_ids)}"
        )

    row_bytes = _uniform_row_bytes(ple_records)
    expert_slot_bytes, expert_bytes_source = derive_expert_slot_bytes(
        moe_records,
        explicit_expert_slot_bytes=explicit_expert_slot_bytes,
    )

    ple_tokens_by_request: dict[str, set[int]] = collections.defaultdict(set)
    ple_rows = collections.Counter()
    ple_phase_accesses: collections.Counter[str] = collections.Counter()
    for record in ple_records:
        request_id = str(record["request_id"])
        try:
            position = int(record["token_position"])
            row = int(record["physical_row_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise H1AnalysisError("normalized PLE record is missing token/row data") from exc
        if row < 0 or row >= geometry.ple_addressable_rows:
            raise H1AnalysisError(
                f"PLE row {row} outside 0..{geometry.ple_addressable_rows - 1}"
            )
        ple_tokens_by_request[request_id].add(position)
        ple_rows[row] += 1
        ple_phase_accesses[str(record.get("phase", "unknown"))] += 1

    expert_counts = [
        collections.Counter() for _ in range(geometry.num_hidden_layers)
    ]
    routed_tokens: dict[str, list[int]] = {
        request_id: [0] * geometry.num_hidden_layers for request_id in successful_ids
    }
    moe_phase_selections: collections.Counter[str] = collections.Counter()
    expert_selections = 0
    expert_last_ordinal: dict[tuple[int, int], int] = {}
    expert_reuse_gap_histogram: collections.Counter[int] = collections.Counter()
    expert_reused_selections = 0
    expert_cold_selections = 0
    expert_reuse_gap_sum = 0
    expert_reuse_gap_max = 0

    def record_expert_reuse(key: tuple[int, int], ordinal: int) -> None:
        nonlocal expert_reused_selections
        nonlocal expert_cold_selections
        nonlocal expert_reuse_gap_sum
        nonlocal expert_reuse_gap_max
        previous = expert_last_ordinal.get(key)
        if previous is None:
            expert_cold_selections += 1
        else:
            gap = ordinal - previous - 1
            expert_reused_selections += 1
            expert_reuse_gap_sum += gap
            expert_reuse_gap_max = max(expert_reuse_gap_max, gap)
            bucket = 0 if gap == 0 else gap.bit_length()
            expert_reuse_gap_histogram[bucket] += 1
        expert_last_ordinal[key] = ordinal

    for record in moe_records:
        request_id = str(record["request_id"])
        layer = parse_layer_id(record["layer"])
        if not 0 <= layer < geometry.num_hidden_layers:
            raise H1AnalysisError(f"MoE layer {layer} outside model geometry")
        selected = [int(value) for value in record.get("selected_expert_ids", [])]
        if not selected or len(selected) % geometry.experts_per_token:
            raise H1AnalysisError(
                "selected expert count is not a positive multiple of experts_per_token"
            )
        for expert in selected:
            if not 0 <= expert < geometry.num_experts:
                raise H1AnalysisError(
                    f"expert {expert} outside 0..{geometry.num_experts - 1}"
                )
            expert_counts[layer][expert] += 1
            record_expert_reuse((layer, expert), expert_selections)
            expert_selections += 1
        token_equivalents = len(selected) // geometry.experts_per_token
        routed_tokens[request_id][layer] += token_equivalents
        moe_phase_selections[str(record.get("phase", "unknown"))] += len(selected)

    coverage_rows: list[dict[str, Any]] = []
    coverage_failures: list[str] = []
    client_by_id = {str(record["request_id"]): record for record in clients}
    for request_id in sorted(successful_ids):
        ple_token_count = len(ple_tokens_by_request[request_id])
        layer_counts = routed_tokens[request_id]
        missing_layers = [
            layer for layer, value in enumerate(layer_counts) if value == 0
        ]
        mismatched_layers = [
            layer
            for layer, value in enumerate(layer_counts)
            if value != ple_token_count
        ]
        client = client_by_id[request_id]
        prompt_tokens = int(client.get("prompt_tokens", 0))
        completion_tokens = int(client.get("completion_tokens", 0))
        autoregressive_expected = prompt_tokens + max(completion_tokens - 1, 0)
        if missing_layers:
            coverage_failures.append(
                f"{request_id}:missing_layers={missing_layers}"
            )
        if mismatched_layers:
            coverage_failures.append(
                f"{request_id}:moe_ple_token_mismatch_layers={mismatched_layers}"
            )
        coverage_rows.append(
            {
                "request_id": request_id,
                "ple_model_tokens": ple_token_count,
                "moe_tokens_per_layer_min": min(layer_counts),
                "moe_tokens_per_layer_max": max(layer_counts),
                "all_layers_present": not missing_layers,
                "moe_matches_ple_all_layers": not mismatched_layers,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "autoregressive_expected_model_tokens": autoregressive_expected,
                "ple_minus_autoregressive_expected": (
                    ple_token_count - autoregressive_expected
                ),
            }
        )

    if coverage_failures:
        preview = "; ".join(coverage_failures[:4])
        if len(coverage_failures) > 4:
            preview += f"; ... ({len(coverage_failures)} failures)"
        raise H1AnalysisError(
            "MoE coverage is incomplete relative to PLE; refusing H1 evidence: "
            + preview
        )

    model_tokens = sum(len(values) for values in ple_tokens_by_request.values())
    unique_layer_experts = sum(len(counts) for counts in expert_counts)
    ple_unique_bytes = len(ple_rows) * row_bytes
    expert_unique_bytes = unique_layer_experts * expert_slot_bytes
    ple_requested_bytes = len(ple_records) * row_bytes
    expert_requested_bytes = expert_selections * expert_slot_bytes
    if model_tokens <= 0:
        raise H1AnalysisError("no model-token observations found")

    per_layer_unique = [len(counts) for counts in expert_counts]
    result = {
        "schema_version": 1,
        "evidence_kind": "local_measured_trace_analysis",
        "hypothesis": "H1_working_set_sparsity",
        "performance_eligible": False,
        "can_establish_edge_performance": False,
        "h1_coverage_complete": True,
        "case_id": str(case.get("case_id", case_dir.name)),
        "case": {
            "input_tokens": int(case["input_tokens"]),
            "output_tokens": int(case["output_tokens"]),
            "requests": int(case["requests"]),
            "concurrency": int(case["concurrency"]),
            "correlation_mode": str(case["correlation_mode"]),
        },
        "source": {
            "repository_commit": evidence.get("repository_commit"),
            "runtime_image": evidence.get("launch", {}).get("runtime_image"),
            "runtime_image_id": evidence.get("launch", {}).get("runtime_image_id"),
            "artifacts": {
                "evidence.json": sha256_file(evidence_path),
                "client.jsonl": sha256_file(client_path),
                "ple_normalized.jsonl": sha256_file(ple_path),
                "moe_normalized.jsonl": sha256_file(moe_path),
            },
        },
        "geometry": {
            "num_hidden_layers": geometry.num_hidden_layers,
            "num_experts": geometry.num_experts,
            "experts_per_token": geometry.experts_per_token,
            "ple_addressable_rows": geometry.ple_addressable_rows,
            "ple_row_bytes": row_bytes,
            "expert_slot_bytes": expert_slot_bytes,
            "expert_slot_bytes_source": expert_bytes_source,
        },
        "coverage": {
            "request_count": len(successful_ids),
            "requests": coverage_rows,
        },
        "measured_working_set": {
            "model_token_observations": model_tokens,
            "ple_accesses": len(ple_records),
            "ple_unique_rows": len(ple_rows),
            "ple_unique_bytes": ple_unique_bytes,
            "expert_selections": expert_selections,
            "unique_layer_experts": unique_layer_experts,
            "expert_unique_bytes": expert_unique_bytes,
            "conditional_unique_bytes": ple_unique_bytes + expert_unique_bytes,
            "ple_requested_bytes": ple_requested_bytes,
            "expert_requested_bytes": expert_requested_bytes,
            "conditional_requested_bytes": (
                ple_requested_bytes + expert_requested_bytes
            ),
            "ple_requested_bytes_per_model_token": (
                ple_requested_bytes / model_tokens
            ),
            "expert_requested_bytes_per_model_token": (
                expert_requested_bytes / model_tokens
            ),
            "conditional_requested_bytes_per_model_token": (
                (ple_requested_bytes + expert_requested_bytes) / model_tokens
            ),
        },
        "expert_reuse_gap": {
            "definition": (
                "number of routed-expert selection references between consecutive "
                "uses of the same (layer, expert); streaming locality metric"
            ),
            "cold_selections": expert_cold_selections,
            "reused_selections": expert_reused_selections,
            "mean_selection_gap": (
                expert_reuse_gap_sum / expert_reused_selections
                if expert_reused_selections
                else None
            ),
            "max_selection_gap": (
                expert_reuse_gap_max if expert_reused_selections else None
            ),
            "log2_histogram": [
                {
                    "min": 0 if bucket == 0 else 1 << (bucket - 1),
                    "max": 0 if bucket == 0 else (1 << bucket) - 1,
                    "count": count,
                }
                for bucket, count in sorted(expert_reuse_gap_histogram.items())
            ],
        },
        "expert_layer_working_set": {
            "unique_experts_per_layer_min": min(per_layer_unique),
            "unique_experts_per_layer_mean": (
                sum(per_layer_unique) / len(per_layer_unique)
            ),
            "unique_experts_per_layer_max": max(per_layer_unique),
            "concentration": _expert_concentration(
                expert_counts, topk_values
            ),
        },
        "phase": {
            "ple_row_accesses": dict(sorted(ple_phase_accesses.items())),
            "expert_selections": dict(sorted(moe_phase_selections.items())),
        },
    }

    ple_locality = _load_optional_json(case_dir / "ple_locality.json")
    if ple_locality is not None:
        result["ple_locality"] = {
            "accesses": ple_locality.get("accesses"),
            "unique_rows": ple_locality.get("unique_rows"),
            "working_set_bytes": ple_locality.get("working_set_bytes"),
            "inter_request_reused_rows": ple_locality.get(
                "inter_request_reused_rows"
            ),
            "reuse_distance": ple_locality.get("reuse_distance"),
        }
    expert_locality = _load_optional_json(case_dir / "expert_locality.json")
    if expert_locality is not None:
        result["expert_locality"] = {
            "expert_selections": expert_locality.get("expert_selections"),
            "unique_layer_experts": expert_locality.get("unique_layer_experts"),
            "reuse_distance": expert_locality.get("reuse_distance"),
        }

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument(
        "--campaign", type=Path, default=Path("configs/campaign.json")
    )
    parser.add_argument("--expert-slot-bytes", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = analyze_case(
            args.case_dir,
            args.campaign,
            explicit_expert_slot_bytes=args.expert_slot_bytes,
        )
    except (H1AnalysisError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
