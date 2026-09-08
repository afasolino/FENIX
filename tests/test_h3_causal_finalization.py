from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.h3.causal_gate import _load_promotion_contract
from analysis.h3.causal_matrix import required_matrix_keys
from analysis.h3.common import sha256_file
from scripts.h3_causal_campaign import _discover_prerequisites


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/h3/contract_v6.json"
AMENDMENT = ROOT / "configs/h3/prefetch_causality_amendment_v1.json"
PROMOTION = ROOT / "configs/h3/causal_promotion_gate_v1.json"


def test_causal_matrix_is_exactly_84_contract_rows():
    contract = json.loads(CONTRACT.read_text())
    keys = required_matrix_keys(contract)
    assert len(keys) == 84
    assert len(set(keys)) == 84
    assert {key[0] for key in keys} == {
        "chat_en",
        "knowledge",
        "math",
        "code",
        "multilingual",
        "session",
        "long_context_8k",
    }
    assert {key[1] for key in keys} == {
        "useful_object_lru",
        "useful_object_lfu",
        "vm_granularity_lru",
        "vm_granularity_lfu",
    }
    assert {key[2] for key in keys} == {None, "prefill", "decode"}


def test_causal_promotion_contract_is_bound_without_mutating_causality_amendment():
    payload = _load_promotion_contract(PROMOTION, CONTRACT, AMENDMENT)
    assert payload["base_contract_sha256"] == sha256_file(CONTRACT)
    assert payload["causality_amendment_sha256"] == sha256_file(AMENDMENT)
    assert payload["frozen_measurement_head"] == (
        "e08bc8d32b85ef3fe929d350ca122159fcee4073"
    )


def test_causal_promotion_binds_pinned_dynamic_lru_dispatch_ordering():
    payload = json.loads(PROMOTION.read_text())
    audit = payload["pinned_runtime_dispatch_audit"]
    assert audit["revision"] == "7b5f0465db90fc49d6324904f48ad995ebdcb62f"
    assert audit["git_blob_sha"] == "28a67404be1c60252b4ea48cb20547e69c698a1f"
    assert audit["dynamic_lru_enabled_in_fenix_launch"] is True
    assert audit["redesigned_split_or_speculative_scheduler_in_scope"] is False
    assert "gather every missing expert tensor row" in audit["observed_program_order"]
    assert "cache.kernel.apply" in audit["observed_program_order"]


def test_causal_promotion_can_only_support_gap_not_sufficiency():
    payload = json.loads(PROMOTION.read_text())
    substitution = payload["pagecache_substitution"]
    assert substitution["allowed_for_gap_promotion_only"] is True
    assert substitution[
        "measured_linux_pagecache_required_for_conventional_sufficiency"
    ] is True
    assert substitution["conventional_sufficiency_may_not_be_promoted_by_this_gate"] is True


def test_causal_promotion_requires_special_full_belady_control():
    payload = json.loads(PROMOTION.read_text())
    matrix = payload["matrix"]
    assert set(matrix["special_full_strata"]) == {"session", "long_context_8k"}
    assert matrix["required_special_full_baseline"] == (
        "causal_belady_pagecache_plus_measured_storage_bandwidth"
    )


def test_prerequisite_autodiscovery_finds_one_bound_manifest(tmp_path: Path):
    path = tmp_path / "h3-prerequisite-manifest.json"
    path.write_text(json.dumps({"artifact_kind": "fenix_h3_prerequisite_manifest"}))
    assert _discover_prerequisites(tmp_path) == path.resolve()


def test_prerequisite_autodiscovery_fails_closed_on_ambiguity(tmp_path: Path):
    for name in ("a-prerequisite.json", "b-prerequisite.json"):
        (tmp_path / name).write_text(
            json.dumps({"artifact_kind": "fenix_h3_prerequisite_manifest"})
        )
    with pytest.raises(Exception, match="exactly one"):
        _discover_prerequisites(tmp_path)
