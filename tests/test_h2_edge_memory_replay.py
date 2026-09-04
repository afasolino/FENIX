
from __future__ import annotations
import collections, json
from analysis.h2_edge_memory_replay import CaseCounts, optimize_static_policy, evaluate_policy, replay

def case(case_id,input_tokens,ple,experts):
    return CaseCounts(
        case_id=case_id,input_tokens=input_tokens,model_tokens=10,
        row_bytes=10,expert_slot_bytes=100,
        ple=collections.Counter(ple),experts=collections.Counter(experts),
    )

def test_demand_fill_selected_object_has_one_compulsory_miss():
    training=case("train",1024,{1:5,2:1},{(0,0):4,(0,1):1})
    policy=optimize_static_policy(training, 200/(1024**3))
    result=evaluate_policy(training,policy,is_training_case=True)
    assert result["conditional_lower_tier_bytes"] < result["conditional_requested_bytes"]
    assert result["cache_used_bytes"] <= result["budget_bytes"]

def test_cross_workload_replay_marks_holdout(tmp_path):
    contract={
        "schema_version":1,
        "artifact_kind":"fenix_h1_h2_edge_replay_contract",
        "scientific_scope":{"budget_scope":"conditional_state_cache_capacity_not_total_edge_device_dram"},
        "h2":{
            "volatile_cache_budgets_gib":[4],
            "policy":"cross_case_static_hotset_demand_fill",
            "training_input_tokens":1024,
            "policy_objective":"maximize_training_avoided_lower_tier_bytes",
        }
    }
    path=tmp_path/"contract.json"; path.write_text(json.dumps(contract))
    train=case("train",1024,{1:5,2:2},{(0,0):4})
    hold=case("hold",4096,{1:4,3:3},{(0,0):3,(0,2):2})
    result=replay([train,hold],path)
    roles={row["case_id"]:row["evaluation_role"] for row in result["case_results"]}
    assert roles["train"]=="training_in_sample"
    assert roles["hold"]=="cross_workload_holdout"
    assert result["can_establish_h3"] is False
