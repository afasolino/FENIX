
from __future__ import annotations
import json
from scripts.h1_h2_campaign import discover_exact_cases

def test_discovery_selects_exact_required_cases(tmp_path):
    contract={
        "schema_version":1,
        "artifact_kind":"fenix_h1_h2_edge_replay_contract",
        "required_trace_cases":{
            "input_tokens":[128,1024,4096],
            "concurrency":1,
            "correlation_mode":"exact_request_correlation",
        },
        "h2":{
            "volatile_cache_budgets_gib":[4,7],
            "policy":"cross_case_static_hotset_demand_fill",
            "training_input_tokens":1024,
            "policy_objective":"maximize_training_avoided_lower_tier_bytes",
        }
    }
    cp=tmp_path/"contract.json";cp.write_text(json.dumps(contract))
    root=tmp_path/"traces";root.mkdir()
    for n in (128,1024,4096):
        d=root/f"i{n:06d}-c01-r01";d.mkdir()
        (d/"evidence.json").write_text(json.dumps({
            "trace_valid":True,
            "case":{
                "input_tokens":n,"concurrency":1,
                "correlation_mode":"exact_request_correlation",
            }
        }))
    selected=discover_exact_cases(root,cp)
    assert [int(json.loads((p/"evidence.json").read_text())["case"]["input_tokens"]) for p in selected] == [128,1024,4096]
