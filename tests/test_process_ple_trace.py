from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.process_ple_trace import normalize


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_ambiguous_request_correlation_fails_closed(tmp_path: Path):
    runtime = tmp_path / "runtime.jsonl"
    client = tmp_path / "client.jsonl"

    _write_jsonl(
        runtime,
        [
            {
                "kind": "address_batch",
                "address_known_ns": 150,
                "query_start_loc": [0, 1],
                "input_ids": [42],
                "physical_row_ids": [[1] * 16],
                "ngram_context": [[]],
                "row_bytes": 2560,
                "step_id": 0,
            }
        ],
    )
    _write_jsonl(
        client,
        [
            {
                "request_id": "a",
                "start_ns": 100,
                "end_ns": 200,
                "concurrency": 1,
            },
            {
                "request_id": "b",
                "start_ns": 120,
                "end_ns": 180,
                "concurrency": 1,
            },
        ],
    )

    with pytest.raises(ValueError, match="correlated uniquely"):
        normalize(runtime, client)
