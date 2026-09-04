import json
from pathlib import Path

import pytest

from analysis import capacity_tradeoff


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_campaign(path: Path):
    path.write_text(json.dumps({
        "model": {
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20_000_000,
            "ple_addressable_rows": 320_001_536,
            "num_hidden_layers": 48,
            "num_experts": 512,
        },
        "experiments": {"capacity_tradeoff": {"host_memory_budgets_gib": [64]}},
    }))


def test_expert_bytes_derive_from_explicit_transfers(tmp_path: Path):
    trace = tmp_path / "moe.jsonl"
    write_jsonl(trace, [
        {"transfer_expert_ids": [1, 2], "transfer_bytes": 8192},
        {"transfer_expert_ids": [3], "transfer_bytes": 4096},
    ])
    assert capacity_tradeoff.derive_expert_bytes(trace) == 4096


def test_expert_bytes_fail_on_inconsistent_slot_size(tmp_path: Path):
    trace = tmp_path / "moe.jsonl"
    write_jsonl(trace, [
        {"transfer_expert_ids": [1], "transfer_bytes": 4096},
        {"transfer_expert_ids": [2], "transfer_bytes": 8192},
    ])
    with pytest.raises(ValueError, match="one expert byte size"):
        capacity_tradeoff.derive_expert_bytes(trace)


def test_ple_bytes_derive_from_row_width_and_versioned_geometry(tmp_path: Path):
    trace = tmp_path / "ple.jsonl"; campaign = tmp_path / "campaign.json"
    write_jsonl(trace, [{"physical_row_id": 1, "bytes": 160}])
    write_campaign(campaign)
    total, row_bytes, rows = capacity_tradeoff.derive_ple_host_bytes(trace, campaign)
    assert row_bytes == 160
    assert rows == 320_001_536
    assert total == 51_200_245_760


def test_manual_overrides_are_explicitly_labeled(tmp_path: Path):
    moe = tmp_path / "moe.jsonl"; campaign = tmp_path / "campaign.json"
    write_jsonl(moe, [{"layer": 0, "selected_expert_ids": [1]}])
    write_campaign(campaign)
    inputs = capacity_tradeoff.resolve_capacity_inputs(
        moe_trace=moe, ple_trace=None, config_path=campaign,
        expert_bytes_override=123, ple_host_bytes_override=456,
    )
    assert inputs.expert_bytes_source == "manual_override"
    assert inputs.ple_host_bytes_source == "manual_override"
    assert inputs.num_hidden_layers == 48
    assert inputs.num_experts == 512


def test_paired_delta_reports_effective_additional_capacity():
    sequence = [(0, 0), (0, 1), (0, 0)]
    rows = [
        capacity_tradeoff.project_budget(
            sequence,
            1,
            128 * 1024**2,
            512 * 1024**2,
            placement,
            num_hidden_layers=1,
            num_experts=8,
        )
        for placement in ("ple_in_host_dram", "ple_externalized")
    ]
    delta = capacity_tradeoff.paired_deltas(rows)[0]
    assert delta["additional_expert_capacity"] == 4


def test_ple_rows_fall_back_to_base_vocab_formula_for_legacy_configs(tmp_path: Path):
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({
        "model": {
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20_000_000,
        }
    }))

    rows, source = capacity_tradeoff.load_ple_addressable_rows(campaign)

    assert rows == 320_000_000
    assert source == "legacy_base_vocab_formula"
