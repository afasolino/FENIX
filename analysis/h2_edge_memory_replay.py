#!/usr/bin/env python3
"""H2: replay measured Qwen conditional-state demand into small volatile caches.

The policy is intentionally limited to H2.  It models capacity and traffic,
not LPDDR/UFS timing, bandwidth, queueing, energy, or accelerator performance.

A 1024-token exact C=1 case trains a static hot-set policy.  Policies are then
evaluated both in-sample and on the 128/4096-token holdouts.  Selected objects
are demand-filled on first use, so their compulsory first touch still reaches
the lower tier.  This avoids pretending that a cache is magically preloaded.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.expert_locality import parse_layer_id
from analysis.h1_working_set import (
    H1AnalysisError,
    derive_expert_slot_bytes,
)
from analysis.process_ple_trace import load_jsonl


class H2ReplayError(ValueError):
    """Raised when a case or replay contract cannot support H2."""


ExpertKey = tuple[int, int]


@dataclass
class CaseCounts:
    case_id: str
    input_tokens: int
    model_tokens: int
    row_bytes: int
    expert_slot_bytes: int
    ple: collections.Counter[int]
    experts: collections.Counter[ExpertKey]

    @property
    def ple_accesses(self) -> int:
        return sum(self.ple.values())

    @property
    def expert_accesses(self) -> int:
        return sum(self.experts.values())

    @property
    def no_cache_bytes(self) -> int:
        return (
            self.ple_accesses * self.row_bytes
            + self.expert_accesses * self.expert_slot_bytes
        )


@dataclass(frozen=True)
class StaticPolicy:
    budget_gib: float
    budget_bytes: int
    expert_slot_bytes: int
    row_bytes: int
    expert_objects: tuple[ExpertKey, ...]
    ple_rows: tuple[int, ...]
    training_saved_bytes: int

    @property
    def used_bytes(self) -> int:
        return (
            len(self.expert_objects) * self.expert_slot_bytes
            + len(self.ple_rows) * self.row_bytes
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_replay_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise H2ReplayError("unsupported edge replay contract schema")
    if payload.get("artifact_kind") != "fenix_h1_h2_edge_replay_contract":
        raise H2ReplayError("unexpected edge replay contract artifact kind")
    h2 = payload.get("h2")
    if not isinstance(h2, dict):
        raise H2ReplayError("H2 replay contract is missing")
    budgets = h2.get("volatile_cache_budgets_gib")
    if not isinstance(budgets, list) or not budgets:
        raise H2ReplayError("H2 volatile-cache budget list is missing")
    normalized = [float(value) for value in budgets]
    if any(value <= 0 for value in normalized):
        raise H2ReplayError("H2 volatile-cache budgets must be positive")
    if len(set(normalized)) != len(normalized):
        raise H2ReplayError("H2 volatile-cache budgets must be unique")
    if normalized != sorted(normalized):
        raise H2ReplayError("H2 volatile-cache budgets must be sorted")
    if h2.get("policy") != "cross_case_static_hotset_demand_fill":
        raise H2ReplayError("unsupported H2 cache policy")
    return payload


def _load_campaign(path: Path) -> tuple[int, int, int, int]:
    payload = json.loads(path.read_text())
    model = payload.get("model")
    if not isinstance(model, dict):
        raise H2ReplayError("campaign model geometry is missing")
    try:
        return (
            int(model["num_hidden_layers"]),
            int(model["num_experts"]),
            int(model["experts_per_token"]),
            int(model["ple_addressable_rows"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise H2ReplayError("campaign model geometry is incomplete") from exc


def collect_case_counts(
    case_dir: Path,
    campaign_path: Path,
    *,
    explicit_expert_slot_bytes: int | None = None,
) -> CaseCounts:
    evidence_path = case_dir / "evidence.json"
    ple_path = case_dir / "ple_normalized.jsonl"
    moe_path = case_dir / "moe_normalized.jsonl"
    if not evidence_path.is_file() or not ple_path.is_file() or not moe_path.is_file():
        raise H2ReplayError(f"case is missing exact normalized traces: {case_dir}")

    evidence = json.loads(evidence_path.read_text())
    case = evidence.get("case", {})
    if evidence.get("trace_valid") is not True:
        raise H2ReplayError(f"{case_dir.name}: trace_valid is not true")
    if case.get("correlation_mode") != "exact_request_correlation":
        raise H2ReplayError(f"{case_dir.name}: exact correlation is required")
    if int(case.get("concurrency", 0)) != 1:
        raise H2ReplayError(f"{case_dir.name}: H2 requires C=1")

    num_layers, num_experts, experts_per_token, ple_addressable_rows = (
        _load_campaign(campaign_path)
    )
    ple_records = load_jsonl(ple_path)
    moe_records = load_jsonl(moe_path)
    if not ple_records or not moe_records:
        raise H2ReplayError(f"{case_dir.name}: normalized trace stream is empty")

    row_widths = {
        int(record["bytes"])
        for record in ple_records
        if record.get("bytes") is not None
    }
    if len(row_widths) != 1:
        raise H2ReplayError(
            f"{case_dir.name}: expected one measured PLE row width"
        )
    row_bytes = next(iter(row_widths))
    if row_bytes <= 0:
        raise H2ReplayError("PLE row width must be positive")

    try:
        expert_slot_bytes, _ = derive_expert_slot_bytes(
            moe_records,
            explicit_expert_slot_bytes=explicit_expert_slot_bytes,
        )
    except H1AnalysisError as exc:
        raise H2ReplayError(str(exc)) from exc

    ple = collections.Counter()
    token_positions: dict[str, set[int]] = collections.defaultdict(set)
    for record in ple_records:
        row = int(record["physical_row_id"])
        if not 0 <= row < ple_addressable_rows:
            raise H2ReplayError(f"PLE row {row} outside checkpoint geometry")
        ple[row] += 1
        token_positions[str(record["request_id"])].add(int(record["token_position"]))

    experts: collections.Counter[ExpertKey] = collections.Counter()
    per_request_layer_tokens: dict[str, list[int]] = collections.defaultdict(
        lambda: [0] * num_layers
    )
    for record in moe_records:
        layer = parse_layer_id(record["layer"])
        if not 0 <= layer < num_layers:
            raise H2ReplayError(f"MoE layer {layer} outside model geometry")
        selected = [int(value) for value in record.get("selected_expert_ids", [])]
        if not selected or len(selected) % experts_per_token:
            raise H2ReplayError("invalid routed-expert selection width")
        for expert in selected:
            if not 0 <= expert < num_experts:
                raise H2ReplayError(f"expert {expert} outside model geometry")
            experts[(layer, expert)] += 1
        per_request_layer_tokens[str(record["request_id"])][layer] += (
            len(selected) // experts_per_token
        )

    # H2 is not allowed to proceed on the historical decode-only MoE stream.
    for request_id, ple_positions in token_positions.items():
        expected = len(ple_positions)
        layer_counts = per_request_layer_tokens.get(request_id)
        if layer_counts is None or any(value != expected for value in layer_counts):
            raise H2ReplayError(
                f"{case_dir.name}: incomplete MoE coverage for request {request_id}; "
                "run the H1/H2 trace instrumentation before replay"
            )

    return CaseCounts(
        case_id=str(case.get("case_id", case_dir.name)),
        input_tokens=int(case["input_tokens"]),
        model_tokens=sum(len(value) for value in token_positions.values()),
        row_bytes=row_bytes,
        expert_slot_bytes=expert_slot_bytes,
        ple=ple,
        experts=experts,
    )


def _rank_experts(counter: collections.Counter[ExpertKey]) -> list[tuple[ExpertKey, int]]:
    return sorted(
        ((key, count) for key, count in counter.items() if count > 1),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )


def _rank_ple(counter: collections.Counter[int]) -> list[tuple[int, int]]:
    return sorted(
        ((row, count) for row, count in counter.items() if count > 1),
        key=lambda item: (-item[1], item[0]),
    )


def _prefix_saved(
    ranked: list[tuple[Any, int]],
    object_bytes: int,
) -> list[int]:
    prefix = [0]
    running = 0
    for _, count in ranked:
        running += (count - 1) * object_bytes
        prefix.append(running)
    return prefix


def _prepare_training_rankings(
    training: CaseCounts,
) -> tuple[
    list[tuple[ExpertKey, int]],
    list[tuple[int, int]],
    list[int],
    list[int],
]:
    ranked_experts = _rank_experts(training.experts)
    ranked_ple = _rank_ple(training.ple)
    return (
        ranked_experts,
        ranked_ple,
        _prefix_saved(ranked_experts, training.expert_slot_bytes),
        _prefix_saved(ranked_ple, training.row_bytes),
    )


def _optimize_static_policy_from_rankings(
    training: CaseCounts,
    budget_gib: float,
    ranked_experts: list[tuple[ExpertKey, int]],
    ranked_ple: list[tuple[int, int]],
    expert_prefix: list[int],
    ple_prefix: list[int],
) -> StaticPolicy:
    if budget_gib <= 0:
        raise H2ReplayError("cache budget must be positive")
    budget_bytes = int(budget_gib * 1024**3)
    max_experts = min(
        len(ranked_experts),
        budget_bytes // training.expert_slot_bytes,
    )
    best: tuple[int, int, int, int] | None = None
    # tuple fields: saved_bytes, -used_bytes, -expert_count, ple_count
    for expert_count in range(max_experts + 1):
        expert_bytes = expert_count * training.expert_slot_bytes
        remaining = budget_bytes - expert_bytes
        ple_count = min(len(ranked_ple), remaining // training.row_bytes)
        used = expert_bytes + ple_count * training.row_bytes
        saved = expert_prefix[expert_count] + ple_prefix[ple_count]
        candidate = (saved, -used, -expert_count, ple_count)
        if best is None or candidate > best:
            best = candidate

    assert best is not None
    saved_bytes, neg_used, neg_expert_count, ple_count = best
    expert_count = -neg_expert_count
    policy = StaticPolicy(
        budget_gib=budget_gib,
        budget_bytes=budget_bytes,
        expert_slot_bytes=training.expert_slot_bytes,
        row_bytes=training.row_bytes,
        expert_objects=tuple(
            key for key, _ in ranked_experts[:expert_count]
        ),
        ple_rows=tuple(row for row, _ in ranked_ple[:ple_count]),
        training_saved_bytes=saved_bytes,
    )
    if policy.used_bytes != -neg_used:
        raise AssertionError("static-policy byte accounting mismatch")
    if policy.used_bytes > budget_bytes:
        raise AssertionError("static policy exceeds cache budget")
    return policy


def optimize_static_policy(
    training: CaseCounts,
    budget_gib: float,
) -> StaticPolicy:
    """Optimize one static hot set; primarily a unit/API convenience wrapper."""

    rankings = _prepare_training_rankings(training)
    return _optimize_static_policy_from_rankings(
        training,
        budget_gib,
        *rankings,
    )


def _class_metrics(
    counter: collections.Counter[Any],
    selected: set[Any],
    object_bytes: int,
) -> dict[str, Any]:
    accesses = sum(counter.values())
    logical_bytes = accesses * object_bytes
    selected_accesses = 0
    selected_unique_accessed = 0
    for key in selected:
        count = counter.get(key, 0)
        if count:
            selected_accesses += count
            selected_unique_accessed += 1
    # Demand-fill: first access to a selected object is a compulsory miss.
    hits = selected_accesses - selected_unique_accessed
    misses = accesses - hits
    lower_tier_bytes = misses * object_bytes
    return {
        "accesses": accesses,
        "unique_objects": len(counter),
        "selected_objects": len(selected),
        "selected_objects_observed_in_case": selected_unique_accessed,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / accesses if accesses else None,
        "logical_requested_bytes": logical_bytes,
        "lower_tier_bytes": lower_tier_bytes,
        "lower_tier_bytes_reduction_fraction": (
            (logical_bytes - lower_tier_bytes) / logical_bytes
            if logical_bytes
            else None
        ),
    }


def evaluate_policy(
    case: CaseCounts,
    policy: StaticPolicy,
    *,
    is_training_case: bool,
) -> dict[str, Any]:
    if (
        case.row_bytes != policy.row_bytes
        or case.expert_slot_bytes != policy.expert_slot_bytes
    ):
        raise H2ReplayError("training/evaluation object geometry differs")
    ple_selected = set(policy.ple_rows)
    expert_selected = set(policy.expert_objects)
    ple = _class_metrics(case.ple, ple_selected, case.row_bytes)
    expert = _class_metrics(
        case.experts, expert_selected, case.expert_slot_bytes
    )
    total_requested = (
        ple["logical_requested_bytes"] + expert["logical_requested_bytes"]
    )
    total_lower = ple["lower_tier_bytes"] + expert["lower_tier_bytes"]
    return {
        "case_id": case.case_id,
        "input_tokens": case.input_tokens,
        "evaluation_role": "training_in_sample"
        if is_training_case
        else "cross_workload_holdout",
        "model_token_observations": case.model_tokens,
        "budget_gib": policy.budget_gib,
        "budget_bytes": policy.budget_bytes,
        "cache_used_bytes": policy.used_bytes,
        "cache_utilization_fraction": policy.used_bytes / policy.budget_bytes,
        "selected_expert_objects": len(policy.expert_objects),
        "selected_ple_rows": len(policy.ple_rows),
        "ple": ple,
        "experts": expert,
        "conditional_requested_bytes": total_requested,
        "conditional_lower_tier_bytes": total_lower,
        "conditional_lower_tier_bytes_per_model_token": (
            total_lower / case.model_tokens
        ),
        "conditional_lower_tier_bytes_reduction_fraction": (
            (total_requested - total_lower) / total_requested
            if total_requested
            else None
        ),
    }


def replay(
    cases: list[CaseCounts],
    replay_contract_path: Path,
) -> dict[str, Any]:
    if not cases:
        raise H2ReplayError("no exact trace cases supplied")
    contract = load_replay_contract(replay_contract_path)
    h2 = contract["h2"]
    training_input = int(h2["training_input_tokens"])
    training_candidates = [
        case for case in cases if case.input_tokens == training_input
    ]
    if len(training_candidates) != 1:
        raise H2ReplayError(
            f"expected exactly one {training_input}-token training case; "
            f"found={len(training_candidates)}"
        )
    training = training_candidates[0]

    if len({case.row_bytes for case in cases}) != 1:
        raise H2ReplayError("PLE row width differs across cases")
    if len({case.expert_slot_bytes for case in cases}) != 1:
        raise H2ReplayError("expert slot width differs across cases")

    budgets = [float(value) for value in h2["volatile_cache_budgets_gib"]]
    rows = []
    policies = []
    training_rankings = _prepare_training_rankings(training)
    for budget in budgets:
        policy = _optimize_static_policy_from_rankings(
            training,
            budget,
            *training_rankings,
        )
        policies.append(
            {
                "budget_gib": budget,
                "budget_bytes": policy.budget_bytes,
                "used_bytes": policy.used_bytes,
                "selected_expert_objects": len(policy.expert_objects),
                "selected_ple_rows": len(policy.ple_rows),
                "training_saved_bytes_after_compulsory_fills": (
                    policy.training_saved_bytes
                ),
                # Full ID lists are intentionally not emitted: the selected PLE
                # set can be large.  Reproduction derives it deterministically
                # from the hashed training traces and contract.
            }
        )
        for case in sorted(cases, key=lambda value: value.input_tokens):
            rows.append(
                evaluate_policy(
                    case,
                    policy,
                    is_training_case=(case.case_id == training.case_id),
                )
            )

    holdout_rows = [
        row for row in rows if row["evaluation_role"] == "cross_workload_holdout"
    ]
    holdout_summary = []
    for budget in budgets:
        selected = [row for row in holdout_rows if row["budget_gib"] == budget]
        reductions = [
            float(row["conditional_lower_tier_bytes_reduction_fraction"])
            for row in selected
        ]
        bytes_per_token = [
            float(row["conditional_lower_tier_bytes_per_model_token"])
            for row in selected
        ]
        holdout_summary.append(
            {
                "budget_gib": budget,
                "holdout_case_count": len(selected),
                "mean_conditional_lower_tier_bytes_reduction_fraction": (
                    sum(reductions) / len(reductions) if reductions else None
                ),
                "min_conditional_lower_tier_bytes_reduction_fraction": (
                    min(reductions) if reductions else None
                ),
                "max_conditional_lower_tier_bytes_reduction_fraction": (
                    max(reductions) if reductions else None
                ),
                "mean_conditional_lower_tier_bytes_per_model_token": (
                    sum(bytes_per_token) / len(bytes_per_token)
                    if bytes_per_token
                    else None
                ),
            }
        )

    return {
        "schema_version": 1,
        "evidence_kind": "trace_projection",
        "hypothesis": "H2_small_volatile_memory_cache_effectiveness",
        "can_establish_h3": False,
        "can_establish_edge_latency": False,
        "can_establish_edge_energy": False,
        "budget_scope": contract["scientific_scope"]["budget_scope"],
        "policy": {
            "name": h2["policy"],
            "training_input_tokens": training_input,
            "training_case_id": training.case_id,
            "objective": h2["policy_objective"],
            "first_touch": "demand_fill_compulsory_miss",
            "selection_is_in_sample_for_training_case": True,
            "holdouts_are_cross_workload": True,
            "interpretation": (
                "capacity/traffic screening only; no LPDDR/UFS/NVMe timing "
                "or energy parameters are applied"
            ),
        },
        "source_contract": {
            "path": str(replay_contract_path),
            "sha256": sha256_file(replay_contract_path),
        },
        "object_geometry": {
            "ple_row_bytes": training.row_bytes,
            "expert_slot_bytes": training.expert_slot_bytes,
        },
        "policies": policies,
        "case_results": rows,
        "holdout_summary": holdout_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--campaign", type=Path, default=Path("configs/campaign.json")
    )
    parser.add_argument(
        "--replay-contract",
        type=Path,
        default=Path("configs/edge_memory_replay_v1.json"),
    )
    parser.add_argument("--expert-slot-bytes", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        cases = [
            collect_case_counts(
                path,
                args.campaign,
                explicit_expert_slot_bytes=args.expert_slot_bytes,
            )
            for path in args.case_dir
        ]
        result = replay(cases, args.replay_contract)
    except (H2ReplayError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
