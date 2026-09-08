#!/usr/bin/env python3
"""Build a disposable H3 causal-prefetch runtime image transactionally.

The pinned source checkout is read-only.  Standard FENIX instrumentation is
first staged by prepare_runtime.py, then the H3 causal timestamps are added to
the disposable staging copy.  The overlay checksum and instrumentation
provenance are regenerated from the final bytes before the image build.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from instrumentation.add_h3_prefetch_causality import patch_routed_experts
from instrumentation.prepare_runtime import regenerate_overlay_manifest, sha256


DEFAULT_IMAGE = "fenix-qwen38:h3-causal-prefetch-v1"
DEFAULT_OUTPUT = Path(".runtime/instrumented/qwen38-h3-causal")
PIN = "7b5f0465db90fc49d6324904f48ad995ebdcb62f"


def _git(runtime: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(runtime), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout.strip())
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    runtime = args.runtime.resolve()
    output = args.output if args.output.is_absolute() else (root / args.output)
    output = output.resolve()

    try:
        output.relative_to(root)
    except ValueError:
        print("causal runtime output must remain inside the FENIX worktree", file=sys.stderr)
        return 2

    if not (runtime / ".git").exists():
        print(f"pinned runtime checkout is missing: {runtime}", file=sys.stderr)
        return 2
    if _git(runtime, "rev-parse", "HEAD") != PIN:
        print("runtime checkout is not at the pinned revision", file=sys.stderr)
        return 2
    if _git(runtime, "status", "--porcelain=v1"):
        print("pinned runtime checkout is dirty", file=sys.stderr)
        return 2

    prepare = [
        sys.executable,
        "instrumentation/prepare_runtime.py",
        "--runtime",
        str(runtime),
        "--output",
        str(output),
    ]
    print("+", " ".join(prepare), flush=True)
    completed = subprocess.run(prepare, cwd=root, check=False)
    if completed.returncode:
        return completed.returncode

    target = output / "runtime/vllm-overlay/model_executor/layers/fused_moe/routed_experts.py"
    causal = patch_routed_experts(target)

    overlay_manifest = regenerate_overlay_manifest(output)
    instrumentation_manifest = output / "fenix-instrumentation-manifest.json"
    provenance = json.loads(instrumentation_manifest.read_text())
    relative_target = str(target.relative_to(output))
    if isinstance(provenance.get("after"), dict):
        provenance["after"][relative_target] = sha256(target)
    provenance["h3_prefetch_causality"] = {
        **causal,
        "target_relative_path": relative_target,
        "target_sha256": sha256(target),
        "builder_repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    provenance["overlay_manifest"] = overlay_manifest
    instrumentation_manifest.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    result = {
        "runtime_head": PIN,
        "source_checkout": str(runtime),
        "source_checkout_modified": False,
        "output": str(output),
        "image": args.image,
        "causal_target_sha256": sha256(target),
        "overlay_manifest": overlay_manifest,
        "instrumentation_manifest_sha256": sha256(instrumentation_manifest),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.prepare_only:
        return 0

    build = [
        sys.executable,
        "-m",
        "scripts.fenix_podman",
        "build",
        "-t",
        args.image,
        "-f",
        str((output / "docker/Dockerfile").relative_to(root)),
        str(output.relative_to(root)),
    ]
    print("+", " ".join(build), flush=True)
    return subprocess.run(build, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
