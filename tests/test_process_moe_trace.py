import json
from pathlib import Path

import pytest

from analysis import process_moe_trace


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_moe_events_correlate_to_unique_c1_request(tmp_path: Path):
    clients = tmp_path / "clients.jsonl"
    runtime = tmp_path / "moe.jsonl"
    write_jsonl(
        clients,
        [{
            "request_id": "r0", "start_ns": 100, "first_token_ns": 200,
            "end_ns": 300, "concurrency": 1,
        }],
    )
    write_jsonl(
        runtime,
        [
            {"timestamp_ns": 150, "layer": "model.layers.2.mlp", "selected_expert_ids": [1, 2]},
            {"timestamp_ns": 250, "layer": "model.layers.2.mlp", "selected_expert_ids": [3, 4]},
        ],
    )
    rows, summary = process_moe_trace.normalize(runtime, clients)
    assert [row["phase"] for row in rows] == ["prefill", "decode"]
    assert {row["request_id"] for row in rows} == {"r0"}
    assert summary["correlated_events"] == 2


def test_moe_correlation_fails_on_ambiguous_clients(tmp_path: Path):
    clients = tmp_path / "clients.jsonl"
    runtime = tmp_path / "moe.jsonl"
    write_jsonl(
        clients,
        [
            {"request_id": "a", "start_ns": 100, "end_ns": 300, "concurrency": 1},
            {"request_id": "b", "start_ns": 150, "end_ns": 350, "concurrency": 1},
        ],
    )
    write_jsonl(runtime, [{"timestamp_ns": 200, "layer": 1, "selected_expert_ids": [2]}])
    with pytest.raises(ValueError, match="matches=2"):
        process_moe_trace.normalize(runtime, clients)
