import json
from pathlib import Path

import pytest

from scripts.capacity_ab_contract import (
    CapacityABContractError,
    build_document,
    load_contract,
)


def write_campaign(path: Path, *, screen_only=False):
    path.write_text(json.dumps({
        "experiments": {
            "capacity_tradeoff": {
                "host_memory_budgets_gib": [64, 96, 112],
                "minimum_measured_repetitions": 3,
                "measure_only_informative_screened_budgets": screen_only,
                "measure_all_predeclared_budgets": not screen_only,
                "budget_roles": {
                    "64": {"role": "strong", "motivation_eligible": True},
                    "96": {"role": "partial", "motivation_eligible": True},
                    "112": {"role": "control", "motivation_eligible": False},
                },
                "placement_order": {
                    "64": ["ple_in_host_dram", "ple_externalized"],
                    "96": ["ple_externalized", "ple_in_host_dram"],
                    "112": ["ple_in_host_dram", "ple_externalized"],
                },
            }
        }
    }))


def test_contract_renders_six_server_groups_and_eighteen_repetitions(tmp_path: Path):
    campaign = tmp_path / "campaign.json"
    write_campaign(campaign)
    doc = build_document(campaign)
    assert doc["server_start_count"] == 6
    assert doc["measurement_repetition_count"] == 18
    assert doc["server_groups"][2]["placement"] == "ple_externalized"
    assert doc["server_groups"][4]["motivation_eligible"] is False


def test_contract_rejects_post_trace_budget_selection(tmp_path: Path):
    campaign = tmp_path / "campaign.json"
    write_campaign(campaign, screen_only=True)
    with pytest.raises(CapacityABContractError, match="all predeclared budgets"):
        load_contract(campaign)
