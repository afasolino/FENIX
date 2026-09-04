#!/usr/bin/env python3
"""Instrument only the ignored, pinned external Qwen3.8 runtime clone.

Every runtime edit is anchored exactly once. A source mismatch aborts before an
image can be built, and the pinned third-party checkout itself is never edited
by the normal FENIX staging flow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


PIN = "7b5f0465db90fc49d6324904f48ad995ebdcb62f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: anchor count {count}, expected 1")
    path.write_text(text.replace(old, new, 1))


def copy_overlay_helper(root: Path, filename: str) -> None:
    source = Path(__file__).with_name(filename)
    if not source.is_file():
        raise RuntimeError(f"missing FENIX runtime helper: {source}")
    destination = root / "runtime/vllm-overlay" / filename
    shutil.copy2(source, destination)


def instrument_ple_address_trace(path: Path) -> None:
    anchor = "        ngram_ids = torch.cat(id_blocks, dim=-1)\n"
    extra = '''        if is_offload_process() and __import__("os").getenv("FENIX_TRACE","0").lower() in {"1","true","yes"}:
            from vllm.fenix_trace_runtime import emit, next_id
            _address_known_ns = __import__("time").monotonic_ns()
            _weight = getattr(self.ngram_embedding, "weight", None)
            _row_bytes = getattr(self.ngram_embedding, "row_bytes", None)
            if _weight is not None and getattr(_weight, "ndim", 0) >= 2:
                _row_bytes = int(_weight[0].numel() * _weight.element_size())
            emit("ple_runtime", {
                "kind":"address_batch",
                "step_id":next_id("ple_address"),
                "address_known_ns":_address_known_ns,
                "input_ids":input_ids.detach().cpu().reshape(-1).tolist(),
                "query_start_loc":query_start_loc.detach().cpu().reshape(-1).tolist(),
                "ngram_context":ngram_context.detach().cpu().tolist(),
                "physical_row_ids":ngram_ids.detach().cpu().tolist(),
                "row_bytes":_row_bytes,
            })
'''
    replace_once(path, anchor, anchor + extra)


def instrument_ple_bank_runtime(path: Path) -> None:
    """Add an opt-in mmap PLE bank path to the CPU-offload embedding only."""

    replace_once(path, "import math\n", "import math\nimport os\n")

    embedding_anchor = '''        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
            quant_method=_get_ple_embedding_quant_method(
                quant_config,
                f"{prefix}.ngram_embedding",
                getattr(config, "ple_embedding_dtype", None),
            ),
        )
'''
    embedding_replacement = '''        if (
            is_offload_process()
            and os.getenv("FENIX_PLE_STORAGE_MODE", "resident") == "mmap"
        ):
            from vllm.fenix_ple_bank_runtime import FenixPleBankEmbedding

            self.ngram_embedding = FenixPleBankEmbedding.from_environment(
                expected_rows=padded_vocab_size,
                embedding_width=self.head_dim,
            )
        else:
            self.ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                self.head_dim,
                params_dtype=params_dtype,
                padding_size=divisor,
                prefix=f"{prefix}.ngram_embedding",
                quant_method=_get_ple_embedding_quant_method(
                    quant_config,
                    f"{prefix}.ngram_embedding",
                    getattr(config, "ple_embedding_dtype", None),
                ),
            )
'''
    replace_once(path, embedding_anchor, embedding_replacement)

    forward_anchor = '''        ngram_ids = torch.cat(id_blocks, dim=-1)
        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]
            torch.index_select(
                self.ngram_embedding.weight,
                0,
                ngram_ids.reshape(-1),
                out=output.reshape(-1, self.head_dim),
            )
            return output
        return self.ngram_embedding(ngram_ids).flatten(-2)
'''
    forward_replacement = '''        ngram_ids = torch.cat(id_blocks, dim=-1)
        if getattr(self.ngram_embedding, "_fenix_ple_bank", False):
            if output_buffer is None:
                output = torch.empty(
                    (num_tokens, self.embedding_dim),
                    dtype=self.ngram_embedding.torch_dtype,
                    device=input_ids.device,
                )
            else:
                output = output_buffer[:num_tokens, : self.embedding_dim]
                if output.dtype != self.ngram_embedding.torch_dtype:
                    raise RuntimeError(
                        "PLE mmap output-buffer dtype mismatch: "
                        f"buffer={output.dtype}, bank={self.ngram_embedding.torch_dtype}"
                    )
            output_rows = output.reshape(-1, self.head_dim).view(torch.uint8)
            self.ngram_embedding.gather_into(
                ngram_ids.reshape(-1),
                output_rows,
            )
            return output
        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]
            torch.index_select(
                self.ngram_embedding.weight,
                0,
                ngram_ids.reshape(-1),
                out=output.reshape(-1, self.head_dim),
            )
            return output
        return self.ngram_embedding(ngram_ids).flatten(-2)
'''
    replace_once(path, forward_anchor, forward_replacement)

    dtype_anchor = '''        embedding = getattr(self, "ngram_embedding", None)
        weight = getattr(embedding, "weight", None)
'''
    dtype_replacement = '''        embedding = getattr(self, "ngram_embedding", None)
        if getattr(embedding, "_fenix_ple_bank", False):
            return embedding.torch_dtype
        weight = getattr(embedding, "weight", None)
'''
    replace_once(path, dtype_anchor, dtype_replacement)

    load_anchor = '''        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
'''
    load_replacement = '''        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"
        externalized_bank = getattr(
            self.ngram_embedding, "_fenix_ple_bank", False
        )

        for name, loaded_weight in weights:
'''
    replace_once(path, load_anchor, load_replacement)

    scale_anchor = '''            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
'''
    scale_replacement = '''            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if externalized_bank and name == "ngram_embedding.weight_scale":
                # The GPU worker retains the global FP8 scale. The CPU bank
                # worker returns byte-exact FP8 rows and therefore needs no
                # scale parameter of its own.
                loaded.add(name)
                continue
            if name in persistent_buffers:
'''
    replace_once(path, scale_anchor, scale_replacement)

    shard_anchor = '''            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
'''
    shard_replacement = '''            if name.startswith(shard_prefix) and name.endswith(".weight"):
                if externalized_bank:
                    # The checkpoint iterator may expose a lazy tensor view,
                    # but no PLE bytes are copied into anonymous DRAM. Runtime
                    # lookup is served from the separately validated bank.
                    loaded.add("ngram_embedding.weight")
                    continue
                shard_text = name[len(shard_prefix) : -len(".weight")]
'''
    replace_once(path, shard_anchor, shard_replacement)


def instrument_ple_offload_trace(path: Path) -> None:
    replace_once(path, "import functools\n", "import functools\nimport os\n")
    anchor = '''    stream = torch.cuda.current_stream()
    cuda_stream = cuda_driver.CUstream(stream.cuda_stream)
'''
    replacement = '''    stream = torch.cuda.current_stream()
    cuda_stream = cuda_driver.CUstream(stream.cuda_stream)
    _fenix_trace = os.getenv("FENIX_TRACE","0").lower() in {"1","true","yes"}
    if _fenix_trace:
        from vllm.fenix_trace_runtime import next_id
        _fenix_step = next_id("ple_consume")
        torch.cuda.nvtx.range_push(f"fenix.ple.consume.{_fenix_step}")
'''
    replace_once(path, anchor, replacement)

    release_anchor = '''    _cuda_check(
        cuda_driver.cuStreamWriteValue32(
            cuda_stream,
            cuda_driver.CUdeviceptr(consumed_device_ptr),
            1,
            0,
        ),
        "cuStreamWriteValue32(PLE consumed)",
    )
'''
    release_extra = '''    if _fenix_trace:
        torch.cuda.nvtx.range_pop()
'''
    replace_once(path, release_anchor, release_anchor + release_extra)


def instrument_moe_trace(path: Path) -> None:
    anchor = '''            _update_lru_expert_map_kernel[(1,)](
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
'''
    extra = '''            if os.getenv("FENIX_TRACE","0").lower() in ("1","true","yes"):
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
'''
    replace_once(path, anchor, anchor + extra)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--skip-git-pin-check", action="store_true")
    parser.add_argument("--source-pin", default=PIN)
    args = parser.parse_args()

    root = Path(args.runtime).resolve()
    if args.skip_git_pin_check:
        head = args.source_pin
    else:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != PIN:
            raise SystemExit(f"runtime HEAD {head} != {PIN}")

    copy_overlay_helper(root, "fenix_trace_runtime.py")
    copy_overlay_helper(root, "fenix_ple_bank_runtime.py")
    manifest: dict[str, object] = {
        "runtime_head": head,
        "before": {},
        "after": {},
        "helpers": {},
    }
    manifest["helpers"] = {
        name: sha256(root / "runtime/vllm-overlay" / name)
        for name in ("fenix_trace_runtime.py", "fenix_ple_bank_runtime.py")
    }

    ple_impl = (
        root
        / "runtime/vllm-overlay/models/qwen3_8_flash_next/nvidia/ple_layer.py"
    )
    manifest["before"][str(ple_impl.relative_to(root))] = sha256(ple_impl)  # type: ignore[index]
    instrument_ple_bank_runtime(ple_impl)
    instrument_ple_address_trace(ple_impl)
    manifest["after"][str(ple_impl.relative_to(root))] = sha256(ple_impl)  # type: ignore[index]

    ple_offload = root / "runtime/vllm-overlay/model_executor/layers/ple_offload_layer.py"
    manifest["before"][str(ple_offload.relative_to(root))] = sha256(ple_offload)  # type: ignore[index]
    instrument_ple_offload_trace(ple_offload)
    manifest["after"][str(ple_offload.relative_to(root))] = sha256(ple_offload)  # type: ignore[index]

    moe = root / (
        "runtime/vllm-overlay/model_executor/layers/quantization/"
        "compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py"
    )
    manifest["before"][str(moe.relative_to(root))] = sha256(moe)  # type: ignore[index]
    instrument_moe_trace(moe)
    manifest["after"][str(moe.relative_to(root))] = sha256(moe)  # type: ignore[index]

    output = root / "fenix-instrumentation-manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
