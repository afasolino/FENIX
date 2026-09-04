import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/h1_h2_workload_robustness_v1.json"


def test_robustness_contract_has_predeclared_primary_matrix():
    payload = json.loads(CONTRACT.read_text())
    assert payload["artifact_kind"] == "fenix_h1_h2_workload_robustness_contract"
    assert payload["trace"]["concurrency"] == 1
    assert payload["trace"]["server_max_model_len"] == 8192
    assert payload["trace"]["max_input_tokens"] == 7680
    assert payload["trace"]["strata_order"] == [
        "chat_en",
        "knowledge",
        "math",
        "code",
        "multilingual",
        "session",
        "long_context_8k",
    ]
    assert sum(payload["strata"][name]["requests"] for name in payload["trace"]["strata_order"]) == 152


def test_robustness_contract_preserves_edge_capacity_scope_and_h3_boundary():
    payload = json.loads(CONTRACT.read_text())
    assert payload["h2"]["volatile_cache_budgets_gib"] == [4, 7, 8, 12, 16, 19]
    assert payload["h2"]["can_establish_h3"] is False
    assert payload["h2"]["can_establish_edge_latency_or_energy"] is False
    assert payload["scientific_scope"]["h3_excluded"] is True
    assert payload["measured_geometry"]["expert_slot_bytes"] == 2534400
    assert payload["measured_geometry"]["ple_row_bytes_expected"] == 160


def test_long_context_extension_is_predeclared_but_not_mixed_into_primary_runtime():
    payload = json.loads(CONTRACT.read_text())
    extension = payload["extended_context_feasibility"]
    assert extension["primary_campaign_limit"] == 8192
    assert extension["predeclared_follow_on_input_tokens"] == [32768, 65536, 131072]
    assert extension["status"] == "not_primary_until_runtime_capacity_is_requalified"
