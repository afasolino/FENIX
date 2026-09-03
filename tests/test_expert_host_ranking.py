import json
from pathlib import Path

import pytest

from analysis.expert_host_ranking import build_ranking_document, rank_layer


def test_rank_layer_is_frequency_then_id_and_complete():
    from collections import Counter

    ranking = rank_layer(Counter({2: 3, 1: 3, 3: 1}), 5)
    assert ranking == [1, 2, 3, 0, 4]


def test_ranking_document_includes_unobserved_layers_and_hashes(tmp_path: Path):
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"model": {"num_hidden_layers": 2, "num_experts": 4}}))
    trace = tmp_path / "moe.jsonl"
    trace.write_text(json.dumps({"layer": 0, "selected_expert_ids": [2, 2, 1]}) + "\n")

    doc = build_ranking_document([trace], campaign)
    assert doc["layers"]["0"]["ranking"] == [2, 1, 0, 3]
    assert doc["layers"]["1"]["ranking"] == [0, 1, 2, 3]
    assert len(doc["source_traces"][0]["sha256"]) == 64


def test_ranking_fails_on_out_of_range_expert(tmp_path: Path):
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"model": {"num_hidden_layers": 1, "num_experts": 2}}))
    trace = tmp_path / "moe.jsonl"
    trace.write_text(json.dumps({"layer": 0, "selected_expert_ids": [2]}) + "\n")
    with pytest.raises(ValueError, match="outside model geometry"):
        build_ranking_document([trace], campaign)
