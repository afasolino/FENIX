from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _moe_instrumentation_function() -> str:
    text = (ROOT / "instrumentation/prepare_runtime_inplace.py").read_text()
    start = text.index("def instrument_moe_trace(path: Path) -> None:")
    end = text.index("\n\ndef main() -> int:", start)
    return text[start:end]


def _replacement_template(function: str) -> str:
    marker = "replacement = "
    assert marker in function
    return function.split(marker, 1)[1]


def test_build_runtime_wrapper_delegates_policy_inside_main():
    text = (ROOT / "scripts/build_runtime.sh").read_text()

    assert "main() {" in text
    assert 'main "$@"' in text
    assert '"$PY" -m scripts.build_runtime "$@"' in text
    assert "PREPARE_RC" not in text
    assert "fenix-qwen38:locked" not in text
    assert "\nexit " not in text


def test_moe_instrumentation_targets_common_routed_experts_file():
    text = (ROOT / "instrumentation/prepare_runtime_inplace.py").read_text()
    function = _moe_instrumentation_function()

    assert (
        "runtime/vllm-overlay/model_executor/layers/fused_moe/"
        "routed_experts.py"
    ) in text
    assert "assert not self.quant_method.is_monolithic" in function
    assert "# Modular kernels use pre-computed routing" in function


def test_moe_instrumentation_emits_final_topk_before_backend_dispatch():
    function = _moe_instrumentation_function()
    replacement = _replacement_template(function)

    assert "topk_ids.reshape(-1)" in replacement
    assert '"trace_scope": "router_all_layers"' in replacement
    assert 'emit("moe_runtime"' in replacement
    assert "return self.quant_method.apply(" in replacement
    assert replacement.index('emit("moe_runtime"') < replacement.index(
        "return self.quant_method.apply("
    )


def test_moe_instrumentation_is_independent_of_dynamic_lru_cache_state():
    function = _moe_instrumentation_function()

    assert "capacity_block=triton.next_power_of_2(capacity)," not in function
    assert "cache.hot_map.numel()" not in function
    assert "cache.miss_slots" not in function
    assert "_static_hot_cache" not in function
