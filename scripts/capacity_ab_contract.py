#!/usr/bin/env python3
"""Validate and render the predeclared FENIX capacity A/B execution contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BASELINE = "ple_in_host_dram"
COUNTERFACTUAL = "ple_externalized"
PLACEMENTS = (BASELINE, COUNTERFACTUAL)


class CapacityABContractError(RuntimeError):
    """Raised when the predeclared A/B contract is incomplete or ambiguous."""


@dataclass(frozen=True)
class BudgetContract:
    host_budget_gib: float
    role: str
    motivation_eligible: bool
    placement_order: tuple[str, str]


@dataclass(frozen=True)
class ServerGroup:
    group_id: str
    host_budget_gib: float
    budget_role: str
    motivation_eligible: bool
    placement: str
    placement_order_index: int
    repetition_indices: tuple[int, ...]


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CapacityABContractError(f"{field} must be a positive integer")
    return value


def _budget_key(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def load_contract(config_path: Path) -> tuple[list[BudgetContract], int]:
    if not config_path.is_file():
        raise CapacityABContractError(f"campaign does not exist: {config_path}")
    payload = json.loads(config_path.read_text())
    try:
        raw = payload["experiments"]["capacity_tradeoff"]
    except (KeyError, TypeError) as exc:
        raise CapacityABContractError("capacity_tradeoff experiment is missing") from exc
    if not isinstance(raw, dict):
        raise CapacityABContractError("capacity_tradeoff must be an object")

    budgets_raw = raw.get("host_memory_budgets_gib")
    if not isinstance(budgets_raw, list) or not budgets_raw:
        raise CapacityABContractError("host_memory_budgets_gib must be non-empty")
    budgets = [float(value) for value in budgets_raw]
    if any(value <= 0 for value in budgets) or len(set(budgets)) != len(budgets):
        raise CapacityABContractError("host-memory budgets must be positive and unique")

    if raw.get("measure_only_informative_screened_budgets") is not False:
        raise CapacityABContractError(
            "all predeclared budgets must be measured; "
            "measure_only_informative_screened_budgets must be false"
        )
    if raw.get("measure_all_predeclared_budgets") is not True:
        raise CapacityABContractError("measure_all_predeclared_budgets must be true")

    roles = raw.get("budget_roles")
    orders = raw.get("placement_order")
    if not isinstance(roles, dict) or not isinstance(orders, dict):
        raise CapacityABContractError("budget_roles and placement_order are required")

    contracts: list[BudgetContract] = []
    for budget in budgets:
        key = _budget_key(budget)
        role_raw = roles.get(key)
        if not isinstance(role_raw, dict):
            raise CapacityABContractError(f"budget role is missing for {key} GiB")
        role = role_raw.get("role")
        eligible = role_raw.get("motivation_eligible")
        if not isinstance(role, str) or not role:
            raise CapacityABContractError(f"budget role name is invalid for {key} GiB")
        if not isinstance(eligible, bool):
            raise CapacityABContractError(
                f"motivation_eligible must be boolean for {key} GiB"
            )

        order_raw = orders.get(key)
        if not isinstance(order_raw, list) or len(order_raw) != 2:
            raise CapacityABContractError(
                f"placement order must contain two placements for {key} GiB"
            )
        order = tuple(str(value) for value in order_raw)
        if set(order) != set(PLACEMENTS):
            raise CapacityABContractError(
                f"placement order for {key} GiB must contain exactly {PLACEMENTS}"
            )
        contracts.append(
            BudgetContract(
                host_budget_gib=budget,
                role=role,
                motivation_eligible=eligible,
                placement_order=(order[0], order[1]),
            )
        )

    repetitions = _positive_int(
        raw.get("minimum_measured_repetitions"),
        "minimum_measured_repetitions",
    )
    return contracts, repetitions


def build_server_groups(
    config_path: Path,
) -> list[ServerGroup]:
    budgets, repetitions = load_contract(config_path)
    repetition_indices = tuple(range(1, repetitions + 1))
    groups: list[ServerGroup] = []
    for budget in budgets:
        budget_text = _budget_key(budget.host_budget_gib).replace(".", "p")
        for index, placement in enumerate(budget.placement_order, start=1):
            groups.append(
                ServerGroup(
                    group_id=f"b{budget_text}-{placement}",
                    host_budget_gib=budget.host_budget_gib,
                    budget_role=budget.role,
                    motivation_eligible=budget.motivation_eligible,
                    placement=placement,
                    placement_order_index=index,
                    repetition_indices=repetition_indices,
                )
            )
    return groups


def build_document(config_path: Path) -> dict[str, Any]:
    groups = build_server_groups(config_path)
    measurement_count = sum(len(group.repetition_indices) for group in groups)
    return {
        "schema_version": 1,
        "artifact_kind": "capacity_ab_execution_contract",
        "can_establish_motivation": False,
        "server_groups": [asdict(group) for group in groups],
        "server_start_count": len(groups),
        "measurement_repetition_count": measurement_count,
        "execution_policy": {
            "one_server_start_per_budget_placement": True,
            "warmup_before_measured_repetitions": True,
            "placement_order_is_predeclared": True,
            "all_predeclared_budgets_are_measured": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("configs/campaign.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    try:
        document = build_document(args.campaign)
    except (CapacityABContractError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2) + "\n")
    if args.plan or args.out is None:
        print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
