#!/usr/bin/env python3
"""Compare exact Qwen3.8 router traces across expert-residency configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from analysis.expert_locality import parse_layer_id
from analysis.process_ple_trace import load_jsonl


class RouterInvarianceError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RouterInvarianceError(f"{path}: expected object")
    return value


def _hot_experts(server_log: Path) -> int:
    lines = [line for line in server_log.read_text(errors="replace").splitlines()
             if "scripts.fenix_podman" in line and " serve " in f" {line} "]
    if len(lines) != 1:
        raise RouterInvarianceError(f"{server_log}: expected one launch line, got {len(lines)}")
    tokens = shlex.split(lines[0])
    if "--no-enable-prefix-caching" not in tokens:
        raise RouterInvarianceError(f"{server_log}: prefix caching not explicitly disabled")
    values = [token.split("=", 1)[1] for token in tokens
              if token.startswith("VLLM_WNA16_STATIC_HOT_CACHE_SIZE=")]
    if len(values) != 1:
        raise RouterInvarianceError(f"{server_log}: cannot establish hot-expert setting")
    return int(values[0])


def canonical_case(case_dir: Path) -> dict[str, Any]:
    evidence = _load(case_dir / "evidence.json")
    clients = [r for r in load_jsonl(case_dir / "client.jsonl") if "error" not in r]
    rid_to_ord = {str(r["request_id"]): int(r["ordinal"]) for r in clients}
    client_shape = {int(r["ordinal"]): (int(r["prompt_tokens"]), int(r["completion_tokens"])) for r in clients}

    grouped = {ordinal: {} for ordinal in client_shape}
    for event in load_jsonl(case_dir / "moe_normalized.jsonl"):
        ordinal = rid_to_ord[str(event["request_id"])]
        key = (str(event.get("phase", "unknown")), parse_layer_id(event["layer"]))
        selected = [int(v) for v in event["selected_expert_ids"]]
        token_count = int(event.get("token_count", 0))
        grouped[ordinal].setdefault(key, []).append([token_count, *selected])

    fingerprints = {}
    for ordinal in sorted(grouped):
        rows = [[phase, layer, batches] for (phase, layer), batches in sorted(grouped[ordinal].items())]
        fingerprints[ordinal] = hashlib.sha256(
            json.dumps(rows, separators=(",", ":"), sort_keys=False).encode()
        ).hexdigest()

    return {
        "repository_commit": evidence.get("repository_commit"),
        "runtime_image_id": evidence.get("launch", {}).get("runtime_image_id"),
        "prompt_set_sha256": evidence.get("case", {}).get("prompt_set_sha256"),
        "stratum": evidence.get("case", {}).get("stratum"),
        "client_shape": client_shape,
        "fingerprints": fingerprints,
    }


def compare(runs: list[tuple[str, Path, Path]]) -> dict[str, Any]:
    if len(runs) < 2:
        raise RouterInvarianceError("at least two runs are required")
    observed = []
    for label, case_dir, server_log in runs:
        payload = canonical_case(case_dir)
        payload["label"] = label
        payload["hot_experts"] = _hot_experts(server_log)
        payload["case_dir"] = str(case_dir)
        payload["server_log"] = str(server_log)
        observed.append(payload)

    first = observed[0]
    failures = []
    for other in observed[1:]:
        for field in ("runtime_image_id", "prompt_set_sha256", "stratum", "client_shape"):
            if other[field] != first[field]:
                failures.append(f"{other['label']}:{field}_mismatch")
        if other["fingerprints"] != first["fingerprints"]:
            mismatched = sorted(
                ordinal for ordinal in set(first["fingerprints"]) | set(other["fingerprints"])
                if first["fingerprints"].get(ordinal) != other["fingerprints"].get(ordinal)
            )
            failures.append(f"{other['label']}:router_fingerprint_mismatch_ordinals={mismatched}")

    if len({int(item["hot_experts"]) for item in observed}) != len(observed):
        failures.append("hot_expert_settings_not_distinct")

    return {
        "schema_version": 1,
        "artifact_kind": "fenix_moe_router_placement_invariance",
        "exact_invariance": not failures,
        "runs": [
            {
                "label": item["label"], "hot_experts": item["hot_experts"],
                "case_dir": item["case_dir"], "server_log": item["server_log"],
                "repository_commit": item["repository_commit"],
                "runtime_image_id": item["runtime_image_id"],
                "prompt_set_sha256": item["prompt_set_sha256"],
                "stratum": item["stratum"],
                "request_fingerprints": item["fingerprints"],
            }
            for item in observed
        ],
        "failures": failures,
        "interpretation": (
            "Exact equality supports that router-level tracing is independent of "
            "expert-residency capacity; it does not establish checkpoint-precision invariance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=CASE_DIR")
    parser.add_argument("--server-log", action="append", required=True, help="LABEL=SERVER_LOG")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    def parse(values: list[str]) -> dict[str, Path]:
        out = {}
        for value in values:
            if "=" not in value:
                raise RouterInvarianceError(f"expected LABEL=PATH, got {value!r}")
            label, raw = value.split("=", 1)
            out[label] = Path(raw)
        return out

    try:
        cases, logs = parse(args.run), parse(args.server_log)
        if set(cases) != set(logs):
            raise RouterInvarianceError("run and server-log labels differ")
        result = compare([(label, cases[label], logs[label]) for label in sorted(cases)])
    except (RouterInvarianceError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exact_invariance": result["exact_invariance"], "failures": result["failures"], "out": str(args.out)}, indent=2))
    return 0 if result["exact_invariance"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
