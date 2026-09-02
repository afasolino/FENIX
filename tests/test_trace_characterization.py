import json
from pathlib import Path

import pytest

from analysis import trace_characterization


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_joint_characterization_accounts_per_request(tmp_path: Path):
    ple = tmp_path / "ple.jsonl"
    moe = tmp_path / "moe.jsonl"
    write_jsonl(
        ple,
        [
            {"request_id": "r", "physical_row_id": 10, "bytes": 160, "phase": "prefill"},
            {"request_id": "r", "physical_row_id": 10, "bytes": 160, "phase": "decode"},
            {"request_id": "r", "physical_row_id": 11, "bytes": 160, "phase": "decode"},
        ],
    )
    write_jsonl(
        moe,
        [{
            "request_id": "r", "layer": "model.layers.3.mlp",
            "selected_expert_ids": [1, 2], "cache_hit": [True, False],
            "transfer_expert_ids": [2], "transfer_bytes": 4096, "phase": "decode",
        }],
    )
    result = trace_characterization.analyze(ple, moe, expected_request_ids={"r"})
    item = result["request_metrics"][0]
    assert item["ple_row_accesses"] == 3
    assert item["ple_unique_rows"] == 2
    assert item["ple_unique_working_set_bytes"] == 320
    assert item["expert_selections"] == 2
    assert item["expert_transfer_bytes"] == 4096
    assert item["observed_expert_cache_hit_rate"] == 0.5


def test_joint_characterization_rejects_request_set_drift(tmp_path: Path):
    ple = tmp_path / "ple.jsonl"; moe = tmp_path / "moe.jsonl"
    write_jsonl(ple, [{"request_id": "a", "physical_row_id": 1, "bytes": 160}])
    write_jsonl(moe, [{"request_id": "b", "layer": 1, "selected_expert_ids": [1]}])
    with pytest.raises(ValueError, match="request sets differ"):
        trace_characterization.analyze(ple, moe)
