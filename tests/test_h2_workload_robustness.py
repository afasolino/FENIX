import collections

from analysis.h2_workload_robustness import (
    RequestDemand,
    adaptive_request_epoch_lfu,
    evaluate_static_policy,
    optimize_policy,
)


def _request(ordinal, ple, experts):
    return RequestDemand(
        request_id=f"r{ordinal}",
        stratum="chat_en",
        ordinal=ordinal,
        model_tokens=1,
        ple=collections.Counter(ple),
        experts=collections.Counter(experts),
        metadata={},
    )


def test_static_policy_never_exceeds_budget_and_demand_fill_keeps_compulsory_miss():
    gib = 1024**3
    policy = optimize_policy(
        {1: 10, 2: 2},
        {(0, 1): 10, (0, 2): 2},
        budget_gib=1.0,
        row_bytes=160,
        expert_slot_bytes=gib // 2,
    )
    assert policy.used_bytes <= gib
    result = evaluate_static_policy(
        [_request(0, {1: 3}, {(0, 1): 3})],
        policy,
        role="unit",
    )
    assert result["conditional_lower_tier_bytes"] > 0
    assert result["conditional_lower_tier_bytes"] < result["conditional_requested_bytes"]


def test_request_epoch_adaptive_policy_is_cold_on_first_request_then_can_reuse():
    expert_bytes = 1024**2
    sequence = [
        _request(0, {1: 4}, {(0, 1): 4}),
        _request(1, {1: 4}, {(0, 1): 4}),
        _request(2, {1: 4}, {(0, 1): 4}),
    ]
    result = adaptive_request_epoch_lfu(
        sequence,
        budget_gib=0.01,
        row_bytes=160,
        expert_slot_bytes=expert_bytes,
    )
    rows = result["request_rows"]
    assert rows[0]["reduction_fraction"] == 0.0
    assert rows[1]["reduction_fraction"] > 0.0
    assert rows[2]["reduction_fraction"] >= rows[1]["reduction_fraction"]
