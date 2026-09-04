from pathlib import Path

from instrumentation.prepare_runtime_inplace import instrument_moe_trace


def test_moe_instrumentation_is_above_backend_dispatch(tmp_path: Path) -> None:
    path = tmp_path / "routed_experts.py"
    path.write_text(
        """    def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        assert not self.quant_method.is_monolithic

        # Modular kernels use pre-computed routing
        return self.quant_method.apply(
            layer=self,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
"""
    )
    instrument_moe_trace(path)
    text = path.read_text()
    assert '"trace_scope": "router_all_layers"' in text
    assert 'emit("moe_runtime"' in text
    assert 'topk_ids.reshape(-1)' in text
    assert text.index('emit("moe_runtime"') < text.index('return self.quant_method.apply(')
    assert "_static_hot_cache" not in text
