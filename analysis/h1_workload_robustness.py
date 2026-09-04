#!/usr/bin/env python3
"""H1 robustness analysis across natural Qwen3.8 workload strata."""

from __future__ import annotations

import argparse
import collections
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from analysis.expert_locality import parse_layer_id
from analysis.process_ple_trace import load_jsonl


class H1RobustnessError(ValueError):
    """Raised when a robustness trace cannot support H1."""


ExpertKey = tuple[int, int]


@dataclass
class StratumInternal:
    public: dict[str, Any]
    ple_rows: set[int]
    expert_keys: set[ExpertKey]
    expert_top128: set[ExpertKey]
    expert_counts: collections.Counter[ExpertKey]
    request_ple: dict[str, set[int]]
    request_experts: dict[str, set[ExpertKey]]
    request_meta: dict[str, dict[str, Any]]


def jaccard(left: set[Any], right: set[Any]) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def js_divergence(
    left: Mapping[Any, int],
    right: Mapping[Any, int],
) -> float | None:
    """Jensen-Shannon divergence in bits, bounded in [0, 1]."""
    lsum = sum(int(value) for value in left.values())
    rsum = sum(int(value) for value in right.values())
    if lsum <= 0 or rsum <= 0:
        return None
    keys = set(left) | set(right)
    total = 0.0
    for key in keys:
        p = int(left.get(key, 0)) / lsum
        q = int(right.get(key, 0)) / rsum
        m = 0.5 * (p + q)
        if p:
            total += 0.5 * p * math.log2(p / m)
        if q:
            total += 0.5 * q * math.log2(q / m)
    return total


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("artifact_kind") != "fenix_h1_h2_workload_robustness_contract":
        raise H1RobustnessError("unexpected robustness contract")
    return payload


def _load_geometry(path: Path) -> tuple[int, int, int, int]:
    payload = json.loads(path.read_text())
    model = payload.get("model")
    if not isinstance(model, dict):
        raise H1RobustnessError("model campaign geometry is missing")
    try:
        return (
            int(model["num_hidden_layers"]),
            int(model["num_experts"]),
            int(model["experts_per_token"]),
            int(model["ple_addressable_rows"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise H1RobustnessError("model campaign geometry is incomplete") from exc


def _prompt_metadata(case_dir: Path, clients: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    prompts_payload = json.loads((case_dir / "prompts.json").read_text())
    prompts = prompts_payload.get("prompts")
    if not isinstance(prompts, list):
        raise H1RobustnessError(f"{case_dir}: prompts.json is invalid")
    by_ordinal = {}
    for item in prompts:
        if not isinstance(item, dict):
            raise H1RobustnessError("prompt metadata entry is not an object")
        by_ordinal[int(item["ordinal"])] = item
    output = {}
    for client in clients:
        ordinal = int(client["ordinal"])
        if ordinal not in by_ordinal:
            raise H1RobustnessError(f"missing prompt metadata for ordinal {ordinal}")
        meta = dict(by_ordinal[ordinal])
        meta["request_id"] = str(client["request_id"])
        meta["prompt_tokens"] = int(client["prompt_tokens"])
        meta["completion_tokens"] = int(client["completion_tokens"])
        output[str(client["request_id"])] = meta
    return output


def _concentration(
    layer_counts: list[collections.Counter[int]],
    topk_values: Iterable[int],
) -> list[dict[str, Any]]:
    rows = []
    for topk in topk_values:
        fractions = []
        for counts in layer_counts:
            total = sum(counts.values())
            if total:
                fractions.append(
                    sum(value for _, value in counts.most_common(int(topk))) / total
                )
        rows.append(
            {
                "topk_experts_per_layer": int(topk),
                "mean_selection_fraction": sum(fractions) / len(fractions) if fractions else None,
                "min_selection_fraction": min(fractions) if fractions else None,
                "max_selection_fraction": max(fractions) if fractions else None,
            }
        )
    return rows


def _topk_keys(
    layer_counts: list[collections.Counter[int]],
    topk: int,
) -> set[ExpertKey]:
    output: set[ExpertKey] = set()
    for layer, counts in enumerate(layer_counts):
        output.update((layer, expert) for expert, _ in counts.most_common(topk))
    return output


def _normalized_entropy(counts: collections.Counter[int], universe: int) -> float | None:
    total = sum(counts.values())
    if total <= 0 or universe <= 1:
        return None
    entropy = 0.0
    for value in counts.values():
        p = value / total
        entropy -= p * math.log2(p)
    return entropy / math.log2(universe)


def _growth(
    request_order: list[str],
    per_request: Mapping[str, set[Any]],
    fractions: Iterable[float],
) -> list[dict[str, Any]]:
    requested = sorted(set(float(value) for value in fractions))
    if any(not 0 < value <= 1 for value in requested):
        raise H1RobustnessError("working-set growth fractions must be in (0, 1]")
    checkpoints = {
        max(1, math.ceil(value * len(request_order))): value for value in requested
    }
    seen: set[Any] = set()
    rows = []
    for index, request_id in enumerate(request_order, start=1):
        seen.update(per_request.get(request_id, set()))
        if index in checkpoints:
            rows.append(
                {
                    "request_fraction": checkpoints[index],
                    "requests_observed": index,
                    "unique_objects": len(seen),
                }
            )
    return rows


def analyze_stratum_internal(
    case_dir: Path,
    contract_path: Path,
    model_campaign_path: Path,
) -> StratumInternal:
    contract = _load_contract(contract_path)
    num_layers, num_experts, experts_per_token, ple_addressable_rows = _load_geometry(
        model_campaign_path
    )
    h1 = contract["h1"]
    topk_values = [int(value) for value in h1["expert_concentration_topk"]]

    evidence = json.loads((case_dir / "evidence.json").read_text())
    if evidence.get("trace_valid") is not True:
        raise H1RobustnessError(f"{case_dir.name}: trace_valid is not true")
    case = evidence.get("case")
    if not isinstance(case, dict):
        raise H1RobustnessError("trace case metadata missing")
    if case.get("correlation_mode") != "exact_request_correlation":
        raise H1RobustnessError("H1 robustness requires exact request correlation")
    if int(case.get("concurrency", 0)) != 1:
        raise H1RobustnessError("H1 robustness requires concurrency=1")

    clients = [r for r in load_jsonl(case_dir / "client.jsonl") if "error" not in r]
    ple = load_jsonl(case_dir / "ple_normalized.jsonl")
    moe = load_jsonl(case_dir / "moe_normalized.jsonl")
    if not clients or not ple or not moe:
        raise H1RobustnessError(f"{case_dir.name}: required trace stream is empty")
    request_order = [
        str(record["request_id"]) for record in sorted(clients, key=lambda item: int(item["ordinal"]))
    ]
    request_meta = _prompt_metadata(case_dir, clients)
    request_ids = set(request_order)

    row_widths = {int(record["bytes"]) for record in ple if record.get("bytes") is not None}
    if len(row_widths) != 1:
        raise H1RobustnessError("PLE row width is not uniform")
    row_bytes = next(iter(row_widths))

    ple_tokens: dict[str, set[int]] = collections.defaultdict(set)
    ple_tokens_phase: dict[str, dict[str, set[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    request_ple: dict[str, set[int]] = collections.defaultdict(set)
    phase_ple_rows: dict[str, set[int]] = collections.defaultdict(set)
    phase_ple_accesses: collections.Counter[str] = collections.Counter()
    all_ple_rows: set[int] = set()
    for record in ple:
        request_id = str(record["request_id"])
        if request_id not in request_ids:
            raise H1RobustnessError("PLE request ID outside client stream")
        row = int(record["physical_row_id"])
        if not 0 <= row < ple_addressable_rows:
            raise H1RobustnessError(f"PLE row outside geometry: {row}")
        position = int(record["token_position"])
        phase = str(record.get("phase", "unknown"))
        ple_tokens[request_id].add(position)
        ple_tokens_phase[request_id][phase].add(position)
        request_ple[request_id].add(row)
        phase_ple_rows[phase].add(row)
        phase_ple_accesses[phase] += 1
        all_ple_rows.add(row)

    layer_counts = [collections.Counter() for _ in range(num_layers)]
    phase_layer_counts: dict[str, list[collections.Counter[int]]] = collections.defaultdict(
        lambda: [collections.Counter() for _ in range(num_layers)]
    )
    request_experts: dict[str, set[ExpertKey]] = collections.defaultdict(set)
    routed_tokens: dict[str, list[int]] = {
        request_id: [0] * num_layers for request_id in request_ids
    }
    expert_counts: collections.Counter[ExpertKey] = collections.Counter()
    phase_expert_selections: collections.Counter[str] = collections.Counter()

    for record in moe:
        request_id = str(record["request_id"])
        if request_id not in request_ids:
            raise H1RobustnessError("MoE request ID outside client stream")
        layer = parse_layer_id(record["layer"])
        if not 0 <= layer < num_layers:
            raise H1RobustnessError(f"MoE layer outside geometry: {layer}")
        selected = [int(value) for value in record.get("selected_expert_ids", [])]
        if not selected or len(selected) % experts_per_token:
            raise H1RobustnessError("invalid selected-expert width")
        phase = str(record.get("phase", "unknown"))
        for expert in selected:
            if not 0 <= expert < num_experts:
                raise H1RobustnessError(f"expert outside geometry: {expert}")
            key = (layer, expert)
            layer_counts[layer][expert] += 1
            phase_layer_counts[phase][layer][expert] += 1
            expert_counts[key] += 1
            request_experts[request_id].add(key)
        routed_tokens[request_id][layer] += len(selected) // experts_per_token
        phase_expert_selections[phase] += len(selected)

    coverage = []
    failures = []
    client_by_id = {str(record["request_id"]): record for record in clients}
    for request_id in request_order:
        ple_count = len(ple_tokens[request_id])
        layer_token_counts = routed_tokens[request_id]
        missing = [layer for layer, value in enumerate(layer_token_counts) if value == 0]
        mismatched = [
            layer for layer, value in enumerate(layer_token_counts) if value != ple_count
        ]
        client = client_by_id[request_id]
        autoregressive_expected = int(client["prompt_tokens"]) + max(
            int(client["completion_tokens"]) - 1, 0
        )
        if missing:
            failures.append(f"{request_id}:missing_layers={missing}")
        if mismatched:
            failures.append(f"{request_id}:moe_ple_mismatch={mismatched}")
        if h1.get("require_ple_autoregressive_token_equivalence") and ple_count != autoregressive_expected:
            failures.append(
                f"{request_id}:ple_autoregressive_mismatch={ple_count}!={autoregressive_expected}"
            )
        coverage.append(
            {
                "request_id": request_id,
                "ple_model_tokens": ple_count,
                "moe_tokens_per_layer_min": min(layer_token_counts),
                "moe_tokens_per_layer_max": max(layer_token_counts),
                "all_layers_present": not missing,
                "moe_matches_ple_all_layers": not mismatched,
                "autoregressive_expected_model_tokens": autoregressive_expected,
                "ple_matches_autoregressive_expected": ple_count == autoregressive_expected,
            }
        )
    if failures:
        preview = "; ".join(failures[:6])
        if len(failures) > 6:
            preview += f"; ... ({len(failures)} failures)"
        raise H1RobustnessError("H1 coverage failed: " + preview)

    total_model_tokens = sum(len(value) for value in ple_tokens.values())
    unique_experts = set(expert_counts)
    top128 = _topk_keys(layer_counts, 128)
    phase_rows = {}
    for phase in h1["report_phases"]:
        phase_name = str(phase)
        counts = phase_layer_counts.get(
            phase_name, [collections.Counter() for _ in range(num_layers)]
        )
        token_count = sum(
            len(phases.get(phase_name, set()))
            for phases in ple_tokens_phase.values()
        )
        phase_rows[phase_name] = {
            "model_tokens": token_count,
            "ple_accesses": int(phase_ple_accesses.get(phase_name, 0)),
            "ple_unique_rows": len(phase_ple_rows.get(phase_name, set())),
            "expert_selections": int(phase_expert_selections.get(phase_name, 0)),
            "expert_unique_layer_objects": sum(len(counter) for counter in counts),
            "expert_concentration": _concentration(counts, topk_values),
        }

    entropy_values = [
        value
        for value in (_normalized_entropy(counts, num_experts) for counts in layer_counts)
        if value is not None
    ]
    per_layer_unique = [len(counts) for counts in layer_counts]

    subgroup_rows = {}
    subgroup_values = sorted(
        {
            str(meta.get("language"))
            for meta in request_meta.values()
            if meta.get("language") is not None
        }
    )
    if len(subgroup_values) > 1:
        for subgroup in subgroup_values:
            ids = {
                request_id
                for request_id, meta in request_meta.items()
                if str(meta.get("language")) == subgroup
            }
            subgroup_ple = set().union(*(request_ple[rid] for rid in ids)) if ids else set()
            subgroup_experts = (
                set().union(*(request_experts[rid] for rid in ids)) if ids else set()
            )
            subgroup_rows[subgroup] = {
                "requests": len(ids),
                "ple_unique_rows": len(subgroup_ple),
                "expert_unique_layer_objects": len(subgroup_experts),
            }

    session_metrics = []
    sessions: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for request_id, meta in request_meta.items():
        session_id = meta.get("session_id")
        turn_index = meta.get("turn_index")
        if session_id is not None and turn_index is not None:
            sessions[str(session_id)].append((int(turn_index), request_id))
    for session_id, members in sorted(sessions.items()):
        members.sort()
        for (left_turn, left_id), (right_turn, right_id) in zip(members, members[1:]):
            session_metrics.append(
                {
                    "session_id": session_id,
                    "from_turn": left_turn,
                    "to_turn": right_turn,
                    "ple_jaccard": jaccard(request_ple[left_id], request_ple[right_id]),
                    "expert_jaccard": jaccard(
                        request_experts[left_id], request_experts[right_id]
                    ),
                }
            )

    public = {
        "schema_version": 1,
        "artifact_kind": "fenix_h1_workload_robustness_stratum",
        "evidence_kind": "local_measured_trace_analysis",
        "hypothesis": "H1_working_set_sparsity_and_stability",
        "stratum": str(case["stratum"]),
        "case_id": str(case["case_id"]),
        "request_count": len(request_order),
        "model_token_observations": total_model_tokens,
        "coverage_complete": True,
        "coverage": coverage,
        "ple": {
            "addressable_rows": ple_addressable_rows,
            "row_bytes": row_bytes,
            "accesses": len(ple),
            "unique_rows": len(all_ple_rows),
            "unique_row_fraction_of_table": len(all_ple_rows) / ple_addressable_rows,
            "unique_bytes": len(all_ple_rows) * row_bytes,
            "working_set_growth": _growth(
                request_order,
                request_ple,
                h1["working_set_growth_request_fractions"],
            ),
        },
        "experts": {
            "num_layers": num_layers,
            "experts_per_layer": num_experts,
            "selected_per_token": experts_per_token,
            "selections": sum(expert_counts.values()),
            "unique_layer_objects": len(unique_experts),
            "unique_fraction_of_all_layer_experts": len(unique_experts)
            / (num_layers * num_experts),
            "per_layer_unique_min": min(per_layer_unique),
            "per_layer_unique_mean": sum(per_layer_unique) / len(per_layer_unique),
            "per_layer_unique_max": max(per_layer_unique),
            "mean_normalized_selection_entropy": (
                sum(entropy_values) / len(entropy_values) if entropy_values else None
            ),
            "concentration": _concentration(layer_counts, topk_values),
            "working_set_growth": _growth(
                request_order,
                request_experts,
                h1["working_set_growth_request_fractions"],
            ),
        },
        "phase": phase_rows,
        "language_subgroups": subgroup_rows,
        "session_consecutive_turn_overlap": {
            "pair_count": len(session_metrics),
            "rows": session_metrics,
            "mean_ple_jaccard": (
                sum(row["ple_jaccard"] for row in session_metrics if row["ple_jaccard"] is not None)
                / sum(1 for row in session_metrics if row["ple_jaccard"] is not None)
                if any(row["ple_jaccard"] is not None for row in session_metrics)
                else None
            ),
            "mean_expert_jaccard": (
                sum(row["expert_jaccard"] for row in session_metrics if row["expert_jaccard"] is not None)
                / sum(1 for row in session_metrics if row["expert_jaccard"] is not None)
                if any(row["expert_jaccard"] is not None for row in session_metrics)
                else None
            ),
        },
        "source": {
            "repository_commit": evidence.get("repository_commit"),
            "runtime_image_id": evidence.get("launch", {}).get("runtime_image_id"),
            "frozen_corpus_sha256": evidence.get("frozen_corpus_sha256"),
        },
    }
    return StratumInternal(
        public=public,
        ple_rows=all_ple_rows,
        expert_keys=unique_experts,
        expert_top128=top128,
        expert_counts=expert_counts,
        request_ple=dict(request_ple),
        request_experts=dict(request_experts),
        request_meta=request_meta,
    )


def summarize_cross_stratum(
    analyses: Mapping[str, StratumInternal],
) -> dict[str, Any]:
    names = sorted(analyses)
    pairs = []
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            left = analyses[left_name]
            right = analyses[right_name]
            pairs.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "ple_jaccard": jaccard(left.ple_rows, right.ple_rows),
                    "expert_jaccard": jaccard(left.expert_keys, right.expert_keys),
                    "expert_top128_per_layer_jaccard": jaccard(
                        left.expert_top128, right.expert_top128
                    ),
                    "expert_js_divergence_bits": js_divergence(
                        left.expert_counts, right.expert_counts
                    ),
                }
            )
    return {
        "schema_version": 1,
        "artifact_kind": "fenix_h1_cross_stratum_robustness",
        "strata": names,
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/h1_h2_workload_robustness_v1.json"),
    )
    parser.add_argument(
        "--model-campaign", type=Path, default=Path("configs/campaign.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        analyses = {
            str(
                json.loads((case_dir / "evidence.json").read_text())["case"][
                    "stratum"
                ]
            ): analyze_stratum_internal(
                case_dir, args.contract, args.model_campaign
            )
            for case_dir in args.case_dir
        }
        result = {
            "strata": {
                name: analysis.public for name, analysis in analyses.items()
            },
            "cross_stratum": summarize_cross_stratum(analyses),
        }
    except (H1RobustnessError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
