
from pathlib import Path
from instrumentation.prepare_runtime_inplace import instrument_moe_trace

def test_moe_instrumentation_covers_fallback_and_dynamic_paths(tmp_path):
    path=tmp_path/"wna16.py"
    path.write_text(
"""        cache = self._static_hot_cache
        if cache is None or x.shape[0] > self._static_hot_cache_max_tokens:
            return fallback
            _update_lru_expert_map_kernel[(1,)](
                global_ids,
                cache.cold_map,
                cache.hot_map,
                cache.slot_global_ids,
                cache.slot_ages,
                cache.clock,
                cache.miss_local_ids,
                cache.miss_slots,
                num_ids=num_ids,
                global_num_experts=cache.hot_map.numel(),
                capacity=capacity,
                id_block=triton.next_power_of_2(num_ids),
                capacity_block=triton.next_power_of_2(capacity),
                num_warps=4,
            )
"""
    )
    instrument_moe_trace(path)
    text=path.read_text()
    assert '"trace_scope": "selection_only"' in text
    assert '"trace_scope":"selection_and_runtime_cache"' in text
    assert '"token_count": int(x.shape[0])' in text
    assert text.count('emit("moe_runtime"') == 2
