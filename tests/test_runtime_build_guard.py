from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_runtime_wrapper_delegates_policy_inside_main():
    text = (ROOT / "scripts/build_runtime.sh").read_text()

    assert "main() {" in text
    assert 'main "$@"' in text
    assert '"$PY" -m scripts.build_runtime "$@"' in text
    assert "PREPARE_RC" not in text
    assert "fenix-qwen38:locked" not in text
    assert "\nexit " not in text


def test_moe_instrumentation_anchor_matches_pinned_launch_shape():
    text = (ROOT / "instrumentation/prepare_runtime_inplace.py").read_text()

    assert "capacity_block=triton.next_power_of_2(capacity)," in text
    assert "num_warps=4," in text


def test_moe_instrumentation_uses_actual_pinned_cache_map():
    text = (ROOT / "instrumentation/prepare_runtime_inplace.py").read_text()

    assert "global_num_experts=cache.hot_map.numel()," in text
    assert "global_num_experts=layer.expert_map.numel()," not in text
