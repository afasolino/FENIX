#!/usr/bin/env python3
"""Instrument only the ignored, pinned external runtime clone.

Every textual anchor is checked exactly once. A source mismatch aborts before a
runtime image is built.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
PIN="7b5f0465db90fc49d6324904f48ad995ebdcb62f"

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_once(path,old,new):
    t=path.read_text();n=t.count(old)
    if n!=1:raise RuntimeError(f"{path}: anchor count {n}, expected 1")
    path.write_text(t.replace(old,new,1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--runtime",required=True);ap.add_argument("--skip-git-pin-check",action="store_true");ap.add_argument("--source-pin",default=PIN);a=ap.parse_args()
    root=Path(a.runtime).resolve()
    if a.skip_git_pin_check:
        head=a.source_pin
    else:
        head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        if head!=PIN:raise SystemExit(f"runtime HEAD {head} != {PIN}")
    shutil.copy2(Path(__file__).with_name("fenix_trace_runtime.py"),root/"runtime/vllm-overlay/fenix_trace_runtime.py")
    man={"runtime_head":head,"before":{},"after":{}}

    worker=root/"runtime/vllm-overlay/v1/ple_offload/worker.py";man["before"][str(worker.relative_to(root))]=sha(worker)
    old="        self._load_weights()\n"
    new="""        self._load_weights()
        if __import__("os").getenv("FENIX_TRACE","0").lower() in {"1","true","yes"}:
            from vllm.fenix_trace_runtime import emit, next_id
            for _lname, _layer in self._layers.items():
                _target = _layer
                if not hasattr(_target, "compute_ngram_ids"):
                    for _m in _layer.modules():
                        if hasattr(_m, "compute_ngram_ids"):
                            _target = _m
                            break
                if not hasattr(_target, "compute_ngram_ids"):
                    continue
                _orig = _target.compute_ngram_ids
                def _wrapped(input_ids, query_start_loc, ngram_context, _orig=_orig, _lname=_lname, _target=_target):
                    import time as _time
                    _step = next_id("ple_address")
                    _rows = _orig(input_ids, query_start_loc, ngram_context)
                    _row_bytes = None
                    _emb = getattr(_target, "ngram_embedding", None)
                    _weight = getattr(_emb, "weight", None)
                    if _weight is not None and getattr(_weight, "ndim", 0) >= 2:
                        _row_bytes = int(_weight[0].numel() * _weight.element_size())
                    emit("ple_runtime", {
                        "kind":"address_batch","step_id":_step,"layer":_lname,
                        "address_known_ns":_time.monotonic_ns(),
                        "input_ids":input_ids.detach().cpu().reshape(-1).tolist(),
                        "query_start_loc":query_start_loc.detach().cpu().reshape(-1).tolist(),
                        "ngram_context":ngram_context.detach().cpu().tolist(),
                        "physical_row_ids":_rows.detach().cpu().tolist(),
                        "row_bytes":_row_bytes,
                    })
                    return _rows
                _target.compute_ngram_ids = _wrapped
"""
    replace_once(worker,old,new);man["after"][str(worker.relative_to(root))]=sha(worker)

    ple=root/"runtime/vllm-overlay/model_executor/layers/ple_offload_layer.py";man["before"][str(ple.relative_to(root))]=sha(ple)
    replace_once(ple,"import functools\n","import functools\nimport os\n")
    old="""    stream = torch.cuda.current_stream()
    cuda_stream = cuda_driver.CUstream(stream.cuda_stream)
"""
    new="""    stream = torch.cuda.current_stream()
    cuda_stream = cuda_driver.CUstream(stream.cuda_stream)
    _fenix_trace = os.getenv("FENIX_TRACE","0").lower() in {"1","true","yes"}
    if _fenix_trace:
        from vllm.fenix_trace_runtime import next_id
        _fenix_step = next_id("ple_consume")
        torch.cuda.nvtx.range_push(f"fenix.ple.consume.{_fenix_step}")
"""
    replace_once(ple,old,new)
    old="""    _cuda_check(
        cuda_driver.cuStreamWriteValue32(
            cuda_stream,
            cuda_driver.CUdeviceptr(consumed_device_ptr),
            1,
            0,
        ),
        "cuStreamWriteValue32(PLE consumed)",
    )
"""
    replace_once(ple,old,old+"""    if _fenix_trace:
        torch.cuda.nvtx.range_pop()
""")
    man["after"][str(ple.relative_to(root))]=sha(ple)

    moe=root/"runtime/vllm-overlay/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py"
    man["before"][str(moe.relative_to(root))]=sha(moe)
    old="""            _update_lru_expert_map_kernel[(1,)](
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
    extra="""            if os.getenv("FENIX_TRACE","0").lower() in ("1","true","yes"):
                from vllm.fenix_trace_runtime import emit, next_id
                _selected = global_ids.detach().cpu().tolist()
                _miss_slots = cache.miss_slots[:num_ids].detach().cpu().tolist()
                _transfer = [int(_selected[_i]) for _i,_s in enumerate(_miss_slots) if int(_s) >= 0]
                _resident = cache.slot_global_ids.detach().cpu().tolist()
                _cache_tensors = [
                    getattr(cache, _name, None)
                    for _name in (
                        "w13_weight","w2_weight","w13_scale","w2_scale",
                        "w13_zp","w2_zp","w13_g_idx","w2_g_idx",
                        "w13_sort","w2_sort"
                    )
                ]
                _bytes_per_slot = 0
                for _t in _cache_tensors:
                    if _t is not None and _t.shape[0] == capacity:
                        _bytes_per_slot += _t[0].numel() * _t.element_size()
                emit("moe_runtime", {
                    "step_id":next_id("moe"),"layer":self.layer_name,
                    "selected_expert_ids":[int(x) for x in _selected],
                    "cache_hit":[int(s)<0 for s in _miss_slots],
                    "transfer_expert_ids":_transfer,
                    "transfer_bytes":int(len(_transfer)*_bytes_per_slot),
                    "resident_expert_ids":[int(x) for x in _resident if int(x)>=0],
                })
"""
    replace_once(moe,old,old+extra);man["after"][str(moe.relative_to(root))]=sha(moe)
    (root/"fenix-instrumentation-manifest.json").write_text(json.dumps(man,indent=2));print(json.dumps(man,indent=2))
if __name__=="__main__":main()
