#!/usr/bin/env python3
"""Plan, execute, and verify the predeclared FENIX trace campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts import trace_capture, trace_case, trace_contract


DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3.8-flash-next"
DEFAULT_RUNTIME_LANE = Path("configs/runtime_lane.json")
DEFAULT_PLE_TRACE = Path("traces/raw/ple_runtime.jsonl")
DEFAULT_MOE_TRACE = Path("traces/raw/moe_runtime.jsonl")


class TraceCampaignError(RuntimeError):
    """Raised when campaign-level execution or verification fails."""


def verify_complete(
    out_root: Path,
    contract: trace_contract.TraceContract,
) -> dict[str, Any]:
    expected = {case.case_id: case for case in trace_contract.planned_cases(contract)}
    failures: list[str] = []
    manifests: dict[str, dict[str, Any]] = {}

    for case_id, case in expected.items():
        path = out_root / case_id / "evidence.json"
        if not path.is_file():
            failures.append(f"missing_case:{case_id}")
            continue
        payload = json.loads(path.read_text())
        manifests[case_id] = payload
        if payload.get("trace_valid") is not True:
            failures.append(f"invalid_case:{case_id}")
        observed = payload.get("case", {})
        for field, value in (
            ("input_tokens", case.input_tokens),
            ("concurrency", case.concurrency),
            ("repetition_index", case.repetition_index),
            ("correlation_mode", case.correlation_mode),
        ):
            if observed.get(field) != value:
                failures.append(f"case_contract_mismatch:{case_id}:{field}")

        case_dir = out_root / case_id
        for relative, expected_hash in payload.get("artifacts_sha256", {}).items():
            artifact = case_dir / relative
            if not artifact.is_file():
                failures.append(f"missing_artifact:{case_id}:{relative}")
            elif trace_capture.sha256_file(artifact) != expected_hash:
                failures.append(f"artifact_hash_mismatch:{case_id}:{relative}")

    fingerprints = set()
    prompt_sets: dict[int, set[str]] = {}
    for payload in manifests.values():
        launch = payload.get("launch", {})
        fingerprints.add(
            json.dumps(
                {
                    "repository_commit": payload.get("repository_commit"),
                    "campaign_sha256": payload.get("campaign_sha256"),
                    "runtime_lane": payload.get("runtime_lane"),
                    "runtime_image": launch.get("runtime_image"),
                    "runtime_image_id": launch.get("runtime_image_id"),
                },
                sort_keys=True,
            )
        )
        case = payload.get("case", {})
        input_tokens = case.get("input_tokens")
        prompt_hash = case.get("prompt_set_sha256")
        if isinstance(input_tokens, int) and isinstance(prompt_hash, str):
            prompt_sets.setdefault(input_tokens, set()).add(prompt_hash)

    if len(fingerprints) > 1:
        failures.append("cross_case_provenance_mismatch")
    for input_tokens, hashes in prompt_sets.items():
        if len(hashes) != 1:
            failures.append(f"prompt_set_mismatch:input_tokens={input_tokens}")

    failures = sorted(set(failures))
    return {
        "schema_version": 1,
        "experiment": contract.experiment,
        "expected_cases": len(expected),
        "observed_cases": len(manifests),
        "complete": not failures and len(manifests) == len(expected),
        "failures": failures,
        "case_ids": sorted(manifests),
    }


def plan(contract: trace_contract.TraceContract) -> dict[str, Any]:
    cases = trace_contract.planned_cases(contract)
    return {
        "experiment": contract.experiment,
        "workload_profile": contract.workload_profile,
        "requests_per_input_length": contract.requests_per_input_length,
        "output_tokens": contract.output_tokens,
        "case_count": len(cases),
        "cases": [
            {
                "case_id": case.case_id,
                "input_tokens": case.input_tokens,
                "concurrency": case.concurrency,
                "repetition_index": case.repetition_index,
                "correlation_mode": case.correlation_mode,
            }
            for case in cases
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=trace_contract.DEFAULT_CAMPAIGN)
    parser.add_argument("--runtime-lane", type=Path, default=DEFAULT_RUNTIME_LANE)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--ple-trace", type=Path, default=DEFAULT_PLE_TRACE)
    parser.add_argument("--moe-trace", type=Path, default=DEFAULT_MOE_TRACE)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--tokenize-url")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--settle-ms", type=int, default=500)
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--repetition-index", type=int)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--verify-complete", action="store_true")
    parser.add_argument("--summary-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        contract = trace_contract.load_trace_contract(args.campaign)
    except trace_contract.TraceContractError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2

    if args.plan:
        print(json.dumps(plan(contract), indent=2))
        return 0
    if args.out_root is None:
        print(json.dumps({"error": "--out-root is required"}, indent=2))
        return 2
    if args.verify_complete:
        result = verify_complete(args.out_root, contract)
        if args.summary_out is not None:
            trace_capture.write_json(args.summary_out, result)
        print(json.dumps(result, indent=2))
        return 0 if result["complete"] else 3
    if args.server_log is None:
        print(json.dumps({"error": "--server-log is required for collection"}, indent=2))
        return 2
    if args.settle_ms < 0:
        print(json.dumps({"error": "--settle-ms must be >= 0"}, indent=2))
        return 2

    try:
        cases = trace_contract.select_cases(
            trace_contract.planned_cases(contract),
            input_tokens=args.input_tokens,
            concurrency=args.concurrency,
            repetition_index=args.repetition_index,
        )
        root = Path.cwd().resolve()
        if not (root / ".git").is_dir():
            raise TraceCampaignError("run from the FENIX repository root")

        with trace_capture.campaign_lock(args.out_root / ".trace-campaign.lock"):
            completed = []
            for case in cases:
                destination = trace_case.run_case(
                    root=root,
                    contract=contract,
                    case=case,
                    campaign_path=args.campaign,
                    runtime_lane_path=args.runtime_lane,
                    server_log=args.server_log,
                    out_root=args.out_root,
                    ple_source=args.ple_trace,
                    moe_source=args.moe_trace,
                    url=args.url,
                    model=args.model,
                    tokenize_url=args.tokenize_url,
                    settle_ms=args.settle_ms,
                )
                completed.append(str(destination))
                print(json.dumps({"completed": case.case_id, "path": str(destination)}))
        print(json.dumps({"completed_cases": completed}, indent=2))
        return 0
    except (
        TraceCampaignError,
        trace_case.TraceCaseError,
        trace_capture.TraceCaptureError,
        trace_contract.TraceContractError,
        ValueError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
