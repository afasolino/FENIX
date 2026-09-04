
from __future__ import annotations
import json
from pathlib import Path
import pytest
from analysis.h1_working_set import analyze_case, H1AnalysisError

def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))

def make_case(tmp_path: Path, *, incomplete=False):
    campaign = {
        "model": {
            "num_hidden_layers": 2,
            "num_experts": 4,
            "experts_per_token": 2,
            "ple_addressable_rows": 100,
        }
    }
    campaign_path = tmp_path/"campaign.json"
    campaign_path.write_text(json.dumps(campaign))
    case = tmp_path/"i000002-c01-r01"
    case.mkdir()
    (case/"evidence.json").write_text(json.dumps({
        "trace_valid": True,
        "repository_commit": "abc",
        "launch": {"runtime_image":"x","runtime_image_id":"sha256:y"},
        "case": {
            "case_id": case.name,
            "input_tokens": 2,
            "output_tokens": 2,
            "requests": 1,
            "concurrency": 1,
            "correlation_mode":"exact_request_correlation",
        }
    }))
    write_jsonl(case/"client.jsonl", [{
        "request_id":"r1","prompt_tokens":2,"completion_tokens":2,
    }])
    ple=[]
    for pos in range(3):
        for row in (pos*2, pos*2+1):
            ple.append({
                "request_id":"r1","token_position":pos,
                "physical_row_id":row,"bytes":10,"phase":"prefill" if pos<2 else "decode",
            })
    write_jsonl(case/"ple_normalized.jsonl", ple)
    moe=[]
    layers = [0] if incomplete else [0,1]
    for layer in layers:
        moe.append({
            "request_id":"r1","layer":f"model.layers.{layer}.mlp",
            "selected_expert_ids":[0,1, 1,2, 0,2],
            "phase":"prefill",
            "transfer_expert_ids":[0],
            "transfer_bytes":100,
        })
    write_jsonl(case/"moe_normalized.jsonl", moe)
    return case, campaign_path

def test_h1_complete_coverage_and_bytes(tmp_path):
    case,campaign=make_case(tmp_path)
    result=analyze_case(case,campaign,topk_values=[1,2,4])
    assert result["h1_coverage_complete"] is True
    assert result["measured_working_set"]["model_token_observations"] == 3
    assert result["geometry"]["expert_slot_bytes"] == 100
    assert result["measured_working_set"]["expert_selections"] == 12
    assert result["coverage"]["requests"][0]["moe_matches_ple_all_layers"] is True
    assert result["expert_reuse_gap"]["reused_selections"] > 0

def test_h1_refuses_missing_layer(tmp_path):
    case,campaign=make_case(tmp_path,incomplete=True)
    with pytest.raises(H1AnalysisError, match="coverage is incomplete"):
        analyze_case(case,campaign,topk_values=[1,2,4])
