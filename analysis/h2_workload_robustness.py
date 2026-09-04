#!/usr/bin/env python3
"""H2 robustness replay over natural-domain Qwen3.8 conditional-state traces.

The replay is deliberately limited to capacity and logical lower-tier traffic.
It does not apply LPDDR/UFS/NVMe timing, bandwidth, queueing, transaction
amplification, or energy; those remain H3.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from analysis.expert_locality import parse_layer_id
from analysis.process_ple_trace import load_jsonl


class H2RobustnessError(ValueError):
    """Raised when robustness traces cannot support H2 replay."""


ExpertKey = tuple[int, int]


@dataclass
class RequestDemand:
    request_id: str
    stratum: str
    ordinal: int
    model_tokens: int
    ple: collections.Counter[int]
    experts: collections.Counter[ExpertKey]
    metadata: dict[str, Any]


@dataclass
class StaticPolicy:
    budget_gib: float
    budget_bytes: int
    row_bytes: int
    expert_slot_bytes: int
    ple_rows: tuple[int, ...]
    expert_objects: tuple[ExpertKey, ...]
    training_saved_bytes: int

    @property
    def used_bytes(self) -> int:
        return (
            len(self.ple_rows) * self.row_bytes
            + len(self.expert_objects) * self.expert_slot_bytes
        )


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("artifact_kind") != "fenix_h1_h2_workload_robustness_contract":
        raise H2RobustnessError("unexpected robustness contract")
    return payload


def _load_geometry(path: Path) -> tuple[int, int, int, int]:
    payload = json.loads(path.read_text())
    model = payload.get("model")
    if not isinstance(model, dict):
        raise H2RobustnessError("model geometry missing")
    return (
        int(model["num_hidden_layers"]),
        int(model["num_experts"]),
        int(model["experts_per_token"]),
        int(model["ple_addressable_rows"]),
    )


def _prompt_meta(case_dir: Path, clients: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    payload = json.loads((case_dir / "prompts.json").read_text())
    rows = payload.get("prompts")
    if not isinstance(rows, list):
        raise H2RobustnessError("invalid prompts.json")
    by_ordinal = {int(row["ordinal"]): dict(row) for row in rows}
    output = {}
    for client in clients:
        ordinal = int(client["ordinal"])
        if ordinal not in by_ordinal:
            raise H2RobustnessError(f"missing prompt metadata ordinal={ordinal}")
        output[str(client["request_id"])] = by_ordinal[ordinal]
    return output


def collect_case_requests(
    case_dir: Path,
    contract_path: Path,
    model_campaign_path: Path,
) -> tuple[list[RequestDemand], int, int]:
    contract = _load_contract(contract_path)
    num_layers, num_experts, experts_per_token, ple_addressable_rows = _load_geometry(
        model_campaign_path
    )
    expert_slot_bytes = int(contract["measured_geometry"]["expert_slot_bytes"])

    evidence = json.loads((case_dir / "evidence.json").read_text())
    if evidence.get("trace_valid") is not True:
        raise H2RobustnessError(f"{case_dir.name}: trace_valid is not true")
    case = evidence.get("case", {})
    if case.get("correlation_mode") != "exact_request_correlation":
        raise H2RobustnessError("H2 robustness requires exact request correlation")
    if int(case.get("concurrency", 0)) != 1:
        raise H2RobustnessError("H2 robustness requires concurrency=1")
    stratum = str(case["stratum"])

    clients = [record for record in load_jsonl(case_dir / "client.jsonl") if "error" not in record]
    ple_records = load_jsonl(case_dir / "ple_normalized.jsonl")
    moe_records = load_jsonl(case_dir / "moe_normalized.jsonl")
    request_ids = {str(record["request_id"]) for record in clients}
    metadata = _prompt_meta(case_dir, clients)

    row_widths = {int(record["bytes"]) for record in ple_records if record.get("bytes") is not None}
    if len(row_widths) != 1:
        raise H2RobustnessError(f"{case_dir.name}: PLE row width not uniform")
    row_bytes = next(iter(row_widths))
    expected_row_bytes = int(contract["measured_geometry"]["ple_row_bytes_expected"])
    if row_bytes != expected_row_bytes:
        raise H2RobustnessError(
            f"{case_dir.name}: PLE row width drift {row_bytes}!={expected_row_bytes}"
        )

    ple_counts: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    ple_tokens: dict[str, set[int]] = collections.defaultdict(set)
    for record in ple_records:
        request_id = str(record["request_id"])
        if request_id not in request_ids:
            raise H2RobustnessError("PLE request outside client set")
        row = int(record["physical_row_id"])
        if not 0 <= row < ple_addressable_rows:
            raise H2RobustnessError(f"PLE row outside geometry: {row}")
        ple_counts[request_id][row] += 1
        ple_tokens[request_id].add(int(record["token_position"]))

    expert_counts: dict[str, collections.Counter[ExpertKey]] = collections.defaultdict(
        collections.Counter
    )
    routed_tokens: dict[str, list[int]] = {
        request_id: [0] * num_layers for request_id in request_ids
    }
    for record in moe_records:
        request_id = str(record["request_id"])
        if request_id not in request_ids:
            raise H2RobustnessError("MoE request outside client set")
        layer = parse_layer_id(record["layer"])
        selected = [int(value) for value in record.get("selected_expert_ids", [])]
        if not 0 <= layer < num_layers:
            raise H2RobustnessError("MoE layer outside geometry")
        if not selected or len(selected) % experts_per_token:
            raise H2RobustnessError("invalid MoE selection width")
        for expert in selected:
            if not 0 <= expert < num_experts:
                raise H2RobustnessError("expert outside geometry")
            expert_counts[request_id][(layer, expert)] += 1
        routed_tokens[request_id][layer] += len(selected) // experts_per_token

    client_by_id = {str(record["request_id"]): record for record in clients}
    requests = []
    for client in sorted(clients, key=lambda record: int(record["ordinal"])):
        request_id = str(client["request_id"])
        model_tokens = len(ple_tokens[request_id])
        if any(value != model_tokens for value in routed_tokens[request_id]):
            raise H2RobustnessError(
                f"{case_dir.name}:{request_id}: incomplete MoE coverage"
            )
        autoregressive = int(client["prompt_tokens"]) + max(
            int(client["completion_tokens"]) - 1, 0
        )
        if model_tokens != autoregressive:
            raise H2RobustnessError(
                f"{case_dir.name}:{request_id}: PLE/model-token mismatch"
            )
        requests.append(
            RequestDemand(
                request_id=request_id,
                stratum=stratum,
                ordinal=int(client["ordinal"]),
                model_tokens=model_tokens,
                ple=ple_counts[request_id],
                experts=expert_counts[request_id],
                metadata=metadata[request_id],
            )
        )
    return requests, row_bytes, expert_slot_bytes


def aggregate_requests(
    requests: Iterable[RequestDemand],
) -> tuple[collections.Counter[int], collections.Counter[ExpertKey], int]:
    ple: collections.Counter[int] = collections.Counter()
    experts: collections.Counter[ExpertKey] = collections.Counter()
    model_tokens = 0
    for request in requests:
        ple.update(request.ple)
        experts.update(request.experts)
        model_tokens += request.model_tokens
    return ple, experts, model_tokens


def _rank(counter: Mapping[Any, int]) -> list[tuple[Any, int]]:
    return sorted(
        ((key, int(count)) for key, count in counter.items() if int(count) > 1),
        key=lambda item: (-item[1], item[0]),
    )


def _prefix_saved(ranked: Sequence[tuple[Any, int]], object_bytes: int) -> list[int]:
    prefix = [0]
    running = 0
    for _, count in ranked:
        running += (count - 1) * object_bytes
        prefix.append(running)
    return prefix


def optimize_policy(
    ple: Mapping[int, int],
    experts: Mapping[ExpertKey, int],
    *,
    budget_gib: float,
    row_bytes: int,
    expert_slot_bytes: int,
) -> StaticPolicy:
    if budget_gib <= 0:
        raise H2RobustnessError("cache budget must be positive")
    budget_bytes = int(float(budget_gib) * 1024**3)
    ranked_ple = _rank(ple)
    ranked_experts = _rank(experts)
    ple_prefix = _prefix_saved(ranked_ple, row_bytes)
    expert_prefix = _prefix_saved(ranked_experts, expert_slot_bytes)

    max_experts = min(len(ranked_experts), budget_bytes // expert_slot_bytes)
    best: tuple[int, int, int, int] | None = None
    # saved bytes, -used bytes, -expert count, ple count
    for expert_count in range(max_experts + 1):
        expert_bytes = expert_count * expert_slot_bytes
        remaining = budget_bytes - expert_bytes
        ple_count = min(len(ranked_ple), remaining // row_bytes)
        used = expert_bytes + ple_count * row_bytes
        saved = expert_prefix[expert_count] + ple_prefix[ple_count]
        candidate = (saved, -used, -expert_count, ple_count)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    saved, neg_used, neg_expert_count, ple_count = best
    expert_count = -neg_expert_count
    policy = StaticPolicy(
        budget_gib=float(budget_gib),
        budget_bytes=budget_bytes,
        row_bytes=row_bytes,
        expert_slot_bytes=expert_slot_bytes,
        ple_rows=tuple(key for key, _ in ranked_ple[:ple_count]),
        expert_objects=tuple(key for key, _ in ranked_experts[:expert_count]),
        training_saved_bytes=saved,
    )
    if policy.used_bytes != -neg_used or policy.used_bytes > budget_bytes:
        raise AssertionError("policy byte accounting mismatch")
    return policy


def _class_static_metrics(
    counter: Mapping[Any, int],
    selected: set[Any],
    object_bytes: int,
) -> dict[str, Any]:
    accesses = sum(int(value) for value in counter.values())
    selected_accesses = sum(int(counter.get(key, 0)) for key in selected)
    selected_observed = sum(1 for key in selected if int(counter.get(key, 0)) > 0)
    hits = selected_accesses - selected_observed
    misses = accesses - hits
    return {
        "accesses": accesses,
        "unique_objects": len(counter),
        "selected_objects": len(selected),
        "selected_objects_observed": selected_observed,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / accesses if accesses else None,
        "logical_requested_bytes": accesses * object_bytes,
        "lower_tier_bytes": misses * object_bytes,
    }


def evaluate_static_policy(
    requests: Sequence[RequestDemand],
    policy: StaticPolicy,
    *,
    role: str,
) -> dict[str, Any]:
    ple, experts, model_tokens = aggregate_requests(requests)
    ple_metrics = _class_static_metrics(ple, set(policy.ple_rows), policy.row_bytes)
    expert_metrics = _class_static_metrics(
        experts, set(policy.expert_objects), policy.expert_slot_bytes
    )
    requested = ple_metrics["logical_requested_bytes"] + expert_metrics["logical_requested_bytes"]
    lower = ple_metrics["lower_tier_bytes"] + expert_metrics["lower_tier_bytes"]
    return {
        "evaluation_role": role,
        "request_count": len(requests),
        "model_token_observations": model_tokens,
        "budget_gib": policy.budget_gib,
        "cache_used_bytes": policy.used_bytes,
        "selected_ple_rows": len(policy.ple_rows),
        "selected_expert_objects": len(policy.expert_objects),
        "ple": ple_metrics,
        "experts": expert_metrics,
        "conditional_requested_bytes": requested,
        "conditional_lower_tier_bytes": lower,
        "conditional_lower_tier_bytes_per_model_token": (
            lower / model_tokens if model_tokens else None
        ),
        "conditional_lower_tier_bytes_reduction_fraction": (
            (requested - lower) / requested if requested else None
        ),
    }


def _adaptive_class(
    counter: Mapping[Any, int],
    selected: set[Any],
    resident: set[Any],
    object_bytes: int,
) -> tuple[int, int, set[Any]]:
    accesses = sum(int(value) for value in counter.values())
    hits = 0
    accessed_selected: set[Any] = set()
    for key in selected:
        count = int(counter.get(key, 0))
        if not count:
            continue
        accessed_selected.add(key)
        if key in resident:
            hits += count
        else:
            hits += max(0, count - 1)
    misses = accesses - hits
    next_resident = selected & (resident | accessed_selected)
    return hits, misses * object_bytes, next_resident


def adaptive_request_epoch_lfu(
    requests: Sequence[RequestDemand],
    *,
    budget_gib: float,
    row_bytes: int,
    expert_slot_bytes: int,
) -> dict[str, Any]:
    history_ple: collections.Counter[int] = collections.Counter()
    history_experts: collections.Counter[ExpertKey] = collections.Counter()
    selected_ple: set[int] = set()
    selected_experts: set[ExpertKey] = set()
    resident_ple: set[int] = set()
    resident_experts: set[ExpertKey] = set()

    requested_bytes = 0
    lower_bytes = 0
    model_tokens = 0
    request_rows = []
    for request_index, request in enumerate(requests):
        ple_accesses = sum(request.ple.values())
        expert_accesses = sum(request.experts.values())
        requested = (
            ple_accesses * row_bytes + expert_accesses * expert_slot_bytes
        )
        _, ple_lower, next_resident_ple = _adaptive_class(
            request.ple, selected_ple, resident_ple, row_bytes
        )
        _, expert_lower, next_resident_experts = _adaptive_class(
            request.experts,
            selected_experts,
            resident_experts,
            expert_slot_bytes,
        )
        lower = ple_lower + expert_lower
        requested_bytes += requested
        lower_bytes += lower
        model_tokens += request.model_tokens

        history_ple.update(request.ple)
        history_experts.update(request.experts)
        policy = optimize_policy(
            history_ple,
            history_experts,
            budget_gib=budget_gib,
            row_bytes=row_bytes,
            expert_slot_bytes=expert_slot_bytes,
        )
        selected_ple = set(policy.ple_rows)
        selected_experts = set(policy.expert_objects)
        resident_ple = next_resident_ple & selected_ple
        resident_experts = next_resident_experts & selected_experts

        request_rows.append(
            {
                "sequence_index": request_index,
                "request_id": request.request_id,
                "stratum": request.stratum,
                "ordinal": request.ordinal,
                "model_tokens": request.model_tokens,
                "lower_tier_bytes": lower,
                "requested_bytes": requested,
                "reduction_fraction": (
                    (requested - lower) / requested if requested else None
                ),
                "selected_ple_rows_for_next_request": len(selected_ple),
                "selected_expert_objects_for_next_request": len(selected_experts),
            }
        )

    return {
        "policy": "adaptive_request_epoch_lfu",
        "budget_gib": float(budget_gib),
        "request_count": len(requests),
        "model_token_observations": model_tokens,
        "conditional_requested_bytes": requested_bytes,
        "conditional_lower_tier_bytes": lower_bytes,
        "conditional_lower_tier_bytes_per_model_token": (
            lower_bytes / model_tokens if model_tokens else None
        ),
        "conditional_lower_tier_bytes_reduction_fraction": (
            (requested_bytes - lower_bytes) / requested_bytes
            if requested_bytes
            else None
        ),
        "request_rows": request_rows,
    }


def _ordering(
    by_stratum: Mapping[str, Sequence[RequestDemand]],
    strata: Sequence[str],
    mode: str,
    seed: int,
) -> list[RequestDemand]:
    sorted_groups = {
        name: sorted(by_stratum[name], key=lambda request: request.ordinal)
        for name in strata
    }
    if mode == "domain_blocked":
        return [request for name in strata for request in sorted_groups[name]]
    if mode == "round_robin":
        output = []
        maximum = max(len(sorted_groups[name]) for name in strata)
        for index in range(maximum):
            for name in strata:
                if index < len(sorted_groups[name]):
                    output.append(sorted_groups[name][index])
        return output
    if mode == "seeded_random":
        output = [request for name in strata for request in sorted_groups[name]]
        random.Random(seed).shuffle(output)
        return output
    raise H2RobustnessError(f"unsupported mixed-online ordering: {mode}")


def replay_robustness(
    case_dirs: Sequence[Path],
    contract_path: Path,
    model_campaign_path: Path,
) -> dict[str, Any]:
    contract = _load_contract(contract_path)
    h2 = contract["h2"]
    budgets = [float(value) for value in h2["volatile_cache_budgets_gib"]]
    by_stratum: dict[str, list[RequestDemand]] = {}
    row_widths = set()
    expert_widths = set()
    for case_dir in case_dirs:
        requests, row_bytes, expert_slot_bytes = collect_case_requests(
            case_dir, contract_path, model_campaign_path
        )
        if not requests:
            raise H2RobustnessError(f"{case_dir.name}: no requests")
        stratum = requests[0].stratum
        if stratum in by_stratum:
            raise H2RobustnessError(f"duplicate stratum case: {stratum}")
        by_stratum[stratum] = requests
        row_widths.add(row_bytes)
        expert_widths.add(expert_slot_bytes)

    expected = set(contract["trace"]["strata_order"])
    if set(by_stratum) != expected:
        raise H2RobustnessError(
            f"trace strata differ: observed={sorted(by_stratum)} expected={sorted(expected)}"
        )
    if len(row_widths) != 1 or len(expert_widths) != 1:
        raise H2RobustnessError("object geometry differs across strata")
    row_bytes = next(iter(row_widths))
    expert_slot_bytes = next(iter(expert_widths))

    in_domain_rows = []
    for stratum in h2["cross_domain_strata"]:
        requests = sorted(by_stratum[stratum], key=lambda request: request.ordinal)
        train = [request for request in requests if request.ordinal % 2 == 0]
        holdout = [request for request in requests if request.ordinal % 2 == 1]
        if not train or not holdout:
            raise H2RobustnessError(f"{stratum}: in-domain split is empty")
        train_ple, train_experts, _ = aggregate_requests(train)
        holdout_ple, holdout_experts, _ = aggregate_requests(holdout)
        for budget in budgets:
            policy = optimize_policy(
                train_ple,
                train_experts,
                budget_gib=budget,
                row_bytes=row_bytes,
                expert_slot_bytes=expert_slot_bytes,
            )
            measured = evaluate_static_policy(
                holdout, policy, role="in_domain_holdout"
            )
            oracle = optimize_policy(
                holdout_ple,
                holdout_experts,
                budget_gib=budget,
                row_bytes=row_bytes,
                expert_slot_bytes=expert_slot_bytes,
            )
            oracle_eval = evaluate_static_policy(
                holdout, oracle, role="oracle_same_holdout"
            )
            in_domain_rows.append(
                {
                    "stratum": stratum,
                    "budget_gib": budget,
                    "static_frequency": measured,
                    "oracle_frequency": oracle_eval,
                    "oracle_gap_reduction_fraction": (
                        oracle_eval["conditional_lower_tier_bytes_reduction_fraction"]
                        - measured["conditional_lower_tier_bytes_reduction_fraction"]
                    ),
                }
            )

    leave_one_out_rows = []
    cross_domain = list(h2["cross_domain_strata"])
    for held_out in cross_domain:
        training = [
            request
            for stratum in cross_domain
            if stratum != held_out
            for request in by_stratum[stratum]
        ]
        evaluation = list(by_stratum[held_out])
        train_ple, train_experts, _ = aggregate_requests(training)
        eval_ple, eval_experts, _ = aggregate_requests(evaluation)
        for budget in budgets:
            policy = optimize_policy(
                train_ple,
                train_experts,
                budget_gib=budget,
                row_bytes=row_bytes,
                expert_slot_bytes=expert_slot_bytes,
            )
            measured = evaluate_static_policy(
                evaluation, policy, role="leave_one_domain_out"
            )
            oracle = optimize_policy(
                eval_ple,
                eval_experts,
                budget_gib=budget,
                row_bytes=row_bytes,
                expert_slot_bytes=expert_slot_bytes,
            )
            oracle_eval = evaluate_static_policy(
                evaluation, oracle, role="oracle_same_domain"
            )
            leave_one_out_rows.append(
                {
                    "held_out_stratum": held_out,
                    "training_strata": [name for name in cross_domain if name != held_out],
                    "budget_gib": budget,
                    "static_frequency": measured,
                    "oracle_frequency": oracle_eval,
                }
            )

    all_domain_training = [
        request for stratum in cross_domain for request in by_stratum[stratum]
    ]
    train_ple, train_experts, _ = aggregate_requests(all_domain_training)
    structural_rows = []
    for held_out in h2["structural_holdouts"]:
        evaluation = list(by_stratum[held_out])
        eval_ple, eval_experts, _ = aggregate_requests(evaluation)
        for budget in budgets:
            policy = optimize_policy(
                train_ple,
                train_experts,
                budget_gib=budget,
                row_bytes=row_bytes,
                expert_slot_bytes=expert_slot_bytes,
            )
            measured = evaluate_static_policy(
                evaluation, policy, role="structural_holdout"
            )
            oracle = optimize_policy(
                eval_ple,
                eval_experts,
                budget_gib=budget,
                row_bytes=row_bytes,
                expert_slot_bytes=expert_slot_bytes,
            )
            structural_rows.append(
                {
                    "held_out_stratum": held_out,
                    "training_strata": cross_domain,
                    "budget_gib": budget,
                    "static_frequency": measured,
                    "oracle_frequency": evaluate_static_policy(
                        evaluation, oracle, role="oracle_structural_holdout"
                    ),
                }
            )

    mixed_online = []
    seed = int(contract["seed"])
    for ordering_name in h2["mixed_online_orderings"]:
        sequence = _ordering(by_stratum, cross_domain, ordering_name, seed)
        for budget in budgets:
            row = adaptive_request_epoch_lfu(
                sequence,
                budget_gib=budget,
                row_bytes=row_bytes,
                expert_slot_bytes=expert_slot_bytes,
            )
            row["ordering"] = ordering_name
            mixed_online.append(row)

    # Compact summaries used for the paper-level gate.
    domain_summary = []
    for budget in budgets:
        selected = [
            row for row in leave_one_out_rows if row["budget_gib"] == budget
        ]
        reductions = [
            row["static_frequency"][
                "conditional_lower_tier_bytes_reduction_fraction"
            ]
            for row in selected
        ]
        domain_summary.append(
            {
                "budget_gib": budget,
                "domain_count": len(selected),
                "mean_leave_one_domain_out_reduction_fraction": (
                    sum(reductions) / len(reductions)
                ),
                "min_leave_one_domain_out_reduction_fraction": min(reductions),
                "max_leave_one_domain_out_reduction_fraction": max(reductions),
            }
        )

    online_summary = []
    for budget in budgets:
        selected = [row for row in mixed_online if row["budget_gib"] == budget]
        reductions = [
            row["conditional_lower_tier_bytes_reduction_fraction"]
            for row in selected
        ]
        online_summary.append(
            {
                "budget_gib": budget,
                "ordering_count": len(selected),
                "mean_adaptive_reduction_fraction": sum(reductions) / len(reductions),
                "min_adaptive_reduction_fraction": min(reductions),
                "max_adaptive_reduction_fraction": max(reductions),
            }
        )

    return {
        "schema_version": 1,
        "artifact_kind": "fenix_h2_workload_robustness_replay",
        "evidence_kind": "trace_projection",
        "hypothesis": "H2_small_volatile_memory_cache_effectiveness",
        "can_establish_h3": False,
        "can_establish_edge_latency": False,
        "can_establish_edge_energy": False,
        "budget_scope": contract["scientific_scope"]["budget_scope"],
        "object_geometry": {
            "ple_row_bytes": row_bytes,
            "expert_slot_bytes": expert_slot_bytes,
        },
        "policies": {
            "static_frequency": {
                "first_touch": "demand_fill_compulsory_miss",
                "selection_objective": "maximize_training_avoided_logical_lower_tier_bytes",
            },
            "adaptive_request_epoch_lfu": {
                "update_granularity": "after_each_complete_request",
                "newly_selected_objects": "demand_fill_on_first_future_access",
                "retained_objects": "remain_resident_if_still_selected",
            },
            "oracle_frequency": {
                "role": "same_evaluation_trace_upper_bound_for_static_frequency",
                "not_deployable": True,
            },
        },
        "in_domain": in_domain_rows,
        "leave_one_domain_out": leave_one_out_rows,
        "structural_holdouts": structural_rows,
        "mixed_online": mixed_online,
        "leave_one_domain_out_summary": domain_summary,
        "mixed_online_summary": online_summary,
        "interpretation": (
            "capacity/logical-traffic evidence only; no storage transaction size, "
            "latency, bandwidth, queueing, or energy model is applied"
        ),
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
        result = replay_robustness(
            args.case_dir, args.contract, args.model_campaign
        )
    except (H2RobustnessError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
