#!/usr/bin/env python3
"""Add H3 causal expert-prefetch timestamps to the staged pinned runtime.

This script is intentionally applied *after* prepare_runtime_inplace.py has
inserted the normal H1 MoE trace hook. It edits only the ignored external
runtime overlay, never the authoritative FENIX measurement artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MARKER = "fenix_h3_host_ids_ready_ns"


def patch_routed_experts(path: Path) -> dict[str, object]:
    text = path.read_text()
    if MARKER in text:
        raise RuntimeError(f"{path}: H3 prefetch-causality instrumentation already present")

    anchor = '''            _fenix_selected = topk_ids.reshape(-1).detach().cpu().tolist()\n            emit("moe_runtime", {\n'''
    replacement = '''            _fenix_selected = topk_ids.reshape(-1).detach().cpu().tolist()\n            # The expert IDs are now materialized on the host. A conventional\n            # host-driven storage scheduler cannot issue an exact expert read\n            # before this point without prediction/speculation. Capture the\n            # native dispatch-ready timestamp *before* trace-file emission so\n            # the measurement does not manufacture artificial prefetch slack.\n            _fenix_host_ids_ready_ns = __import__("time").monotonic_ns()\n            _fenix_dispatch_call_ready_ns = __import__("time").monotonic_ns()\n            emit("moe_runtime", {\n'''
    if text.count(anchor) != 1:
        raise RuntimeError(f"{path}: expected exactly one staged H1 MoE hook anchor")
    text = text.replace(anchor, replacement, 1)

    dict_anchor = '''                "step_id": next_id("moe_router"),\n                "layer": self.layer_name,\n'''
    dict_replacement = '''                "step_id": next_id("moe_router"),\n                "fenix_h3_host_ids_ready_ns": _fenix_host_ids_ready_ns,\n                "fenix_h3_dispatch_call_ready_ns": _fenix_dispatch_call_ready_ns,\n                "fenix_h3_prefetch_semantics": "exact_expert_ids_host_ready_to_native_dispatch_ready_no_prediction",\n                "layer": self.layer_name,\n'''
    if text.count(dict_anchor) != 1:
        raise RuntimeError(f"{path}: expected exactly one H1 MoE trace dictionary anchor")
    text = text.replace(dict_anchor, dict_replacement, 1)
    path.write_text(text)
    return {
        "path": str(path),
        "marker": MARKER,
        "semantics": (
            "host IDs are ready after topk_ids CPU materialization; dispatch-ready is "
            "captured immediately afterward and before trace emission"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    root = args.runtime.resolve()
    target = root / "runtime/vllm-overlay/model_executor/layers/fused_moe/routed_experts.py"
    if not target.is_file():
        raise SystemExit(f"staged routed_experts.py not found: {target}")
    result = patch_routed_experts(target)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
