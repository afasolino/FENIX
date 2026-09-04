#!/usr/bin/env python3
"""Fail-closed verification of the primary H1 trace-server execution policy."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from scripts import trace_capture

DEFAULT_POLICY = Path("configs/h1_h2_trace_execution_v1.json")


class TracePolicyError(RuntimeError):
    pass


def _load_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise TracePolicyError("unsupported trace execution policy schema")
    if payload.get("artifact_kind") != "fenix_h1_h2_trace_execution_policy":
        raise TracePolicyError("unexpected trace execution policy kind")
    if payload.get("prefix_caching") != "disabled":
        raise TracePolicyError("primary H1 policy must disable prefix caching")
    if payload.get("trace_required") is not True:
        raise TracePolicyError("primary H1 policy must require tracing")
    return payload


def verify(server_log: Path, policy_path: Path) -> dict[str, Any]:
    policy = _load_policy(policy_path)
    launch = trace_capture.require_trace_server(server_log)
    text = server_log.read_text(errors="replace")
    command_lines = [
        line for line in text.splitlines()
        if "scripts.fenix_podman" in line and " serve " in f" {line} "
    ]
    if len(command_lines) != 1:
        raise TracePolicyError(
            f"expected exactly one podman/vLLM launch line; observed={len(command_lines)}"
        )
    tokens = shlex.split(command_lines[0])
    if "--no-enable-prefix-caching" not in tokens:
        raise TracePolicyError("server was not launched with --no-enable-prefix-caching")
    if "--enable-prefix-caching" in tokens:
        raise TracePolicyError("server launch contains contradictory --enable-prefix-caching")
    if "FENIX_PREFIX_CACHING=0" not in tokens:
        raise TracePolicyError("server launch is missing FENIX_PREFIX_CACHING=0 provenance")
    try:
        max_len_index = tokens.index("--max-model-len")
        observed_max_len = int(tokens[max_len_index + 1])
    except (ValueError, IndexError) as exc:
        raise TracePolicyError("cannot establish server --max-model-len") from exc
    expected_max_len = int(policy["server_max_model_len"])
    if observed_max_len != expected_max_len:
        raise TracePolicyError(
            f"server max_model_len differs: {observed_max_len}!={expected_max_len}"
        )
    return {
        "schema_version": 1,
        "artifact_kind": "fenix_h1_h2_trace_execution_verification",
        "prefix_caching": "disabled",
        "trace_values": list(launch.trace_values),
        "runtime_images": list(launch.runtime_images),
        "server_max_model_len": observed_max_len,
        "policy_path": str(policy_path),
        "policy_sha256": trace_capture.sha256_file(policy_path),
        "server_log_sha256_at_verification": trace_capture.sha256_file(server_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.server_log, args.policy)
    except (
        TracePolicyError,
        trace_capture.TraceCaptureError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3
    if args.out is not None:
        trace_capture.write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
