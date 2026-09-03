#!/usr/bin/env python3
"""Evaluate the measured FENIX motivation gate."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASELINE = "ple_in_host_dram"
COUNTERFACTUAL = "ple_externalized"
ENDPOINT_FIELDS = (
    "model_revision",
    "runtime_revision",
    "hardware_id",
    "workload_id",
)


@dataclass(frozen=True)
class BudgetPolicy:
    role: str
    motivation_eligible: bool


@dataclass(frozen=True)
class GateConfig:
    minimum_repetitions: int
    max_tpot_ratio: float
    min_storage_reduction: float
    bootstrap_samples: int
    bootstrap_alpha: float
    budget_policies: dict[float, BudgetPolicy]


def bootstrap_difference_interval(
    baseline: list[float],
    counterfactual: list[float],
    samples: int,
    alpha: float,
    seed: int = 20260901,
) -> tuple[float, float]:
    generator = random.Random(seed)
    differences = []

    for _ in range(samples):
        baseline_sample = [generator.choice(baseline) for _ in baseline]
        counterfactual_sample = [
            generator.choice(counterfactual) for _ in counterfactual
        ]
        differences.append(
            statistics.mean(counterfactual_sample)
            - statistics.mean(baseline_sample)
        )

    differences.sort()
    lower_index = int((alpha / 2) * samples)
    upper_index = min(samples - 1, int((1 - alpha / 2) * samples))
    return differences[lower_index], differences[upper_index]


def _load_budget_policies(experiment: dict[str, object]) -> dict[float, BudgetPolicy]:
    raw_budgets = experiment.get("host_memory_budgets_gib")
    raw_roles = experiment.get("budget_roles")
    if not isinstance(raw_budgets, list) or not raw_budgets:
        raise ValueError("capacity_tradeoff host_memory_budgets_gib is missing")
    if not isinstance(raw_roles, dict):
        raise ValueError("capacity_tradeoff budget_roles is missing")

    policies: dict[float, BudgetPolicy] = {}
    for raw_budget in raw_budgets:
        budget = float(raw_budget)
        key = str(int(budget)) if budget.is_integer() else str(budget)
        item = raw_roles.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"capacity_tradeoff budget role is missing for {key} GiB")
        role = item.get("role")
        eligible = item.get("motivation_eligible")
        if not isinstance(role, str) or not role:
            raise ValueError(f"capacity_tradeoff role is invalid for {key} GiB")
        if not isinstance(eligible, bool):
            raise ValueError(
                f"capacity_tradeoff motivation_eligible is invalid for {key} GiB"
            )
        policies[budget] = BudgetPolicy(role=role, motivation_eligible=eligible)
    return policies


def load_gate_config(config_path: Path) -> GateConfig:
    config = json.loads(config_path.read_text())
    experiment = config["experiments"]["capacity_tradeoff"]
    gate = config["motivation_gate"]

    return GateConfig(
        minimum_repetitions=int(experiment["minimum_measured_repetitions"]),
        max_tpot_ratio=float(gate["max_externalized_to_baseline_tpot_ratio"]),
        min_storage_reduction=float(gate["min_expert_storage_bytes_reduction"]),
        bootstrap_samples=int(gate["bootstrap_samples"]),
        bootstrap_alpha=float(gate["bootstrap_alpha"]),
        budget_policies=_load_budget_policies(experiment),
    )


def endpoint_metadata_errors(rows: Iterable[dict[str, object]]) -> list[str]:
    """Return missing or inconsistent endpoint-provenance fields."""

    errors: list[str] = []
    materialized = list(rows)

    for field in ENDPOINT_FIELDS:
        missing_count = sum(row.get(field) in (None, "") for row in materialized)
        if missing_count:
            errors.append(f"{field}:missing={missing_count}")
            continue

        values = {str(row[field]) for row in materialized}
        if len(values) != 1:
            errors.append(f"{field}:inconsistent")

    return errors


def mean_metric(rows: list[dict[str, object]], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows)


def evaluate_budget(
    budget: float,
    baseline_rows: list[dict[str, object]],
    counterfactual_rows: list[dict[str, object]],
    gate: GateConfig,
    policy: BudgetPolicy,
) -> dict[str, object]:
    baseline_tpot = [float(row["tpot_ms"]) for row in baseline_rows]
    counterfactual_tpot = [float(row["tpot_ms"]) for row in counterfactual_rows]

    baseline_storage = mean_metric(
        baseline_rows, "expert_storage_bytes_per_token"
    )
    counterfactual_storage = mean_metric(
        counterfactual_rows, "expert_storage_bytes_per_token"
    )

    tpot_ratio = statistics.mean(counterfactual_tpot) / statistics.mean(
        baseline_tpot
    )
    storage_reduction = (
        1 - counterfactual_storage / baseline_storage
        if baseline_storage > 0
        else 0.0
    )
    ci_low, ci_high = bootstrap_difference_interval(
        baseline_tpot,
        counterfactual_tpot,
        gate.bootstrap_samples,
        gate.bootstrap_alpha,
    )

    passed_materiality = (
        tpot_ratio <= gate.max_tpot_ratio
        and storage_reduction >= gate.min_storage_reduction
        and ci_high < 0
    )

    return {
        "host_budget_gib": budget,
        "budget_role": policy.role,
        "motivation_eligible": policy.motivation_eligible,
        "baseline_repetitions": len(baseline_rows),
        "counterfactual_repetitions": len(counterfactual_rows),
        "externalized_to_baseline_tpot_ratio": tpot_ratio,
        "expert_storage_bytes_reduction": storage_reduction,
        "bootstrap_tpot_difference_ms": {
            "lower": ci_low,
            "upper": ci_high,
        },
        "passed_materiality": passed_materiality,
        "passed": passed_materiality and policy.motivation_eligible,
    }


def evaluate(
    measured_path: Path,
    config_path: Path,
) -> dict[str, object]:
    verdict: dict[str, object] = {
        "verdict": "INCONCLUSIVE",
        "proceed_to_hardware_architecture": False,
        "reasons": [],
        "budgets": [],
    }

    if not measured_path.exists():
        verdict["reasons"].append("measured capacity-tradeoff results are absent")  # type: ignore[union-attr]
        return verdict

    document = json.loads(measured_path.read_text())
    if document.get("evidence_kind") == "trace_projection":
        verdict["reasons"].append(  # type: ignore[union-attr]
            "trace projections cannot establish the motivation gate"
        )
        return verdict

    rows = [row for row in document.get("rows", []) if bool(row.get("measured"))]
    if not rows:
        verdict["reasons"].append("no measured rows are present")  # type: ignore[union-attr]
        return verdict

    metadata_errors = endpoint_metadata_errors(rows)
    if metadata_errors:
        verdict["reasons"].append(  # type: ignore[union-attr]
            "invalid endpoint metadata: " + ", ".join(metadata_errors)
        )
        return verdict

    try:
        gate = load_gate_config(config_path)
    except (KeyError, TypeError, ValueError) as exc:
        verdict["reasons"].append(f"invalid motivation-gate configuration: {exc}")  # type: ignore[union-attr]
        return verdict

    measured_budgets = sorted(
        {
            float(row["host_budget_gib"])
            for row in rows
            if row.get("host_budget_gib") is not None
        }
    )
    unknown = [budget for budget in measured_budgets if budget not in gate.budget_policies]
    if unknown:
        verdict["reasons"].append(  # type: ignore[union-attr]
            f"measured rows contain undeclared host-memory budgets: {unknown}"
        )
        return verdict

    evaluated = []
    for budget in measured_budgets:
        baseline_rows = [
            row
            for row in rows
            if float(row["host_budget_gib"]) == budget
            and row.get("placement") == BASELINE
        ]
        counterfactual_rows = [
            row
            for row in rows
            if float(row["host_budget_gib"]) == budget
            and row.get("placement") == COUNTERFACTUAL
        ]

        if (
            len(baseline_rows) < gate.minimum_repetitions
            or len(counterfactual_rows) < gate.minimum_repetitions
        ):
            continue

        evaluated.append(
            evaluate_budget(
                budget,
                baseline_rows,
                counterfactual_rows,
                gate,
                gate.budget_policies[budget],
            )
        )

    verdict["budgets"] = evaluated
    eligible = [item for item in evaluated if item["motivation_eligible"]]

    if not eligible:
        verdict["reasons"].append(  # type: ignore[union-attr]
            "no motivation-eligible host-memory budget has enough measured "
            "repetitions in both placements"
        )
        return verdict

    if any(item["passed"] for item in eligible):
        verdict.update(
            verdict="SUPPORTED",
            proceed_to_hardware_architecture=True,
            reasons=[
                "at least one predeclared motivation-eligible budget passes "
                "all materiality gates"
            ],
        )
    else:
        verdict.update(
            verdict="FALSIFIED",
            reasons=[
                "no predeclared motivation-eligible capacity-tradeoff point "
                "passes the materiality gates"
            ],
        )

    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measured_results", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/campaign.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate(args.measured_results, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
