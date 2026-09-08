from __future__ import annotations

import json
from pathlib import Path

from analysis.h3.causal_decision import _causalize_candidate_lower
from analysis.h3.common import set_execution_repository_identity
from analysis.h3.expert_prefetch_causality import build_expert_prefetch_causality
from instrumentation.add_h3_prefetch_causality import patch_routed_experts


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/h3/contract_v6.json"
AMENDMENT = ROOT / "configs/h3/prefetch_causality_amendment_v1.json"
FROZEN = "e08bc8d32b85ef3fe929d350ca122159fcee4073"
ANALYSIS_ID = {
    "head": "7" * 40,
    "tree": "8" * 40,
    "branch": "study/h3-causal-test",
    "clean": True,
    "required_base_commit": "8b8694647840e5af73fd4e1fb1275d0b15c19056",
    "root": str(ROOT),
}


def _write_moe(path: Path, slack_ns: int) -> None:
    rows = []
    timestamp = 1_000_000
    for phase in ("prefill", "decode"):
        for layer in range(48):
            ready = timestamp
            dispatch = ready + slack_ns
            rows.append({
                "timestamp_ns": timestamp,
                "trace_scope": "router_all_layers",
                "phase": phase,
                "layer": f"language_model.model.layers.{layer}.mlp.experts",
                "selected_expert_ids": list(range(10)),
                "fenix_h3_host_ids_ready_ns": ready,
                "fenix_h3_dispatch_call_ready_ns": dispatch,
                "fenix_h3_prefetch_semantics": "exact_expert_ids_host_ready_to_native_dispatch_ready_no_prediction",
            })
            timestamp += 1000
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_fio(path: Path, low_ns_per_byte: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_kind": "fenix_h3_fio_calibration",
        "execution_repository": {
            "head": FROZEN,
            "tree": "a1af6ceff0e41e0cde20ee0ee2848e7c700d8a48",
            "branch": "study/h3-conventional-hierarchy-v6",
            "clean": True,
            "required_base_commit": "8b8694647840e5af73fd4e1fb1275d0b15c19056",
            "root": str(ROOT),
        },
        "class_coefficients": {
            "expert:qd32": {
                "ns_per_byte_ci95": [low_ns_per_byte, low_ns_per_byte * 1.2],
            }
        },
        "fio_tool": {"head": "a" * 40, "binary_sha256": "b" * 64},
        "provenance": {
            "storage_binding_sha256": "c" * 64,
            "device_major_minor": "8:0",
        },
    }
    path.write_text(json.dumps(payload))


def _projection_fio(expert_coeff: float = 2.0) -> dict:
    windows = ("w0", "w1", "w2")
    def node(value: float) -> dict:
        return {
            "replicate_groups": {key: [value] for key in windows},
            "replicate_values": [value] * len(windows),
        }
    return {
        "paired_window_design": {"verified": True},
        "bootstrap_draws": 128,
        "class_coefficients": {
            "ple:qd32": node(1.0),
            "expert:qd32": node(expert_coeff),
        },
        "mixed_interaction": {"qd32": node(1.0)},
    }


def test_runtime_patch_places_causal_timestamps_after_host_id_materialization(tmp_path: Path):
    source = tmp_path / "routed_experts.py"
    source.write_text(
        '            _fenix_selected = topk_ids.reshape(-1).detach().cpu().tolist()\n'
        '            emit("moe_runtime", {\n'
        '                "kind": "selection_batch",\n'
        '                "step_id": next_id("moe_router"),\n'
        '                "layer": self.layer_name,\n'
        '            })\n'
    )
    patch_routed_experts(source)
    text = source.read_text()
    assert "fenix_h3_host_ids_ready_ns" in text
    assert "fenix_h3_dispatch_call_ready_ns" in text
    assert text.index("_fenix_selected =") < text.index("_fenix_host_ids_ready_ns")
    assert text.index("_fenix_dispatch_call_ready_ns") < text.index('emit("moe_runtime"')


def test_causal_lower_adds_unavoidable_expert_service_before_dispatch():
    candidate = {
        "baseline": "synthetic",
        "service_ns_bounds": [1000.0, 5000.0],
    }
    _causalize_candidate_lower(
        candidate,
        all_lpddr_ns=1000.0,
        expert_storage_bytes=100.0,
        fio=_projection_fio(expert_coeff=2.0),
        queue_depth=32,
    )
    assert candidate["pre_causal_service_ns_bounds"] == [1000.0, 5000.0]
    assert candidate["causal_expert_storage_service_ns"]["low_ns"] == 200.0
    assert candidate["service_ns_bounds"][0] == 1200.0
    assert candidate["service_ns_bounds"][1] == 5000.0
    assert candidate["expert_storage_serialized_before_native_dispatch"] is True


def test_causality_passes_only_when_native_window_is_shorter_than_fastest_expert_io(tmp_path: Path):
    moe = tmp_path / "moe.jsonl"
    _write_moe(moe, slack_ns=100)
    fio_root = tmp_path / "fio"
    # 1 ns/byte -> ~2.5 ms per expert, much larger than 100 ns native slack.
    _write_fio(fio_root / "one" / "fio-summary.json", 1.0)
    out = tmp_path / "causality.json"
    set_execution_repository_identity(ANALYSIS_ID)
    try:
        result = build_expert_prefetch_causality(
            [moe], fio_root, CONTRACT, AMENDMENT, out
        )
    finally:
        set_execution_repository_identity(None)
    assert result["derived_pass"] is True
    assert result["phase_stats"]["prefill"]["events"] == 48
    assert result["phase_stats"]["decode"]["events"] == 48
    assert result["slack_over_fastest_one_expert_service"] < 1.0
    assert json.loads(out.read_text())["execution_repository"] == ANALYSIS_ID


def test_causality_fails_closed_when_one_expert_can_fit_in_native_window(tmp_path: Path):
    moe = tmp_path / "moe.jsonl"
    _write_moe(moe, slack_ns=3_000_000)
    fio_root = tmp_path / "fio"
    _write_fio(fio_root / "one" / "fio-summary.json", 1.0)
    out = tmp_path / "causality.json"
    set_execution_repository_identity(ANALYSIS_ID)
    try:
        result = build_expert_prefetch_causality(
            [moe], fio_root, CONTRACT, AMENDMENT, out
        )
    finally:
        set_execution_repository_identity(None)
    assert result["derived_pass"] is False
    assert any("one_expert_transfer" in item for item in result["failures"])
