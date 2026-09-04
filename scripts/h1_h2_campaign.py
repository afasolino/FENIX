#!/usr/bin/env python3
"""Run the FENIX H1/H2 analysis pipeline over exact C=1 trace cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analysis.h1_working_set import H1AnalysisError, analyze_case
from analysis.h2_edge_memory_replay import (
    H2ReplayError,
    collect_case_counts,
    load_replay_contract,
    replay,
)


class H1H2CampaignError(ValueError):
    """Raised when the H1/H2 campaign contract is not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise H1H2CampaignError(f"{path}: expected JSON object")
    return value


def discover_exact_cases(
    trace_root: Path,
    replay_contract_path: Path,
) -> list[Path]:
    if not trace_root.is_dir():
        raise H1H2CampaignError(f"trace root does not exist: {trace_root}")
    contract = load_replay_contract(replay_contract_path)
    required = contract.get("required_trace_cases")
    if not isinstance(required, dict):
        raise H1H2CampaignError("required_trace_cases contract is missing")
    required_inputs = [int(value) for value in required["input_tokens"]]
    required_concurrency = int(required["concurrency"])
    required_correlation = str(required["correlation_mode"])

    found: dict[int, list[Path]] = {value: [] for value in required_inputs}
    for candidate in sorted(trace_root.iterdir()):
        if not candidate.is_dir():
            continue
        evidence_path = candidate / "evidence.json"
        if not evidence_path.is_file():
            continue
        evidence = _load_json(evidence_path)
        if evidence.get("trace_valid") is not True:
            continue
        case = evidence.get("case")
        if not isinstance(case, dict):
            continue
        try:
            input_tokens = int(case["input_tokens"])
            concurrency = int(case["concurrency"])
            correlation = str(case["correlation_mode"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            input_tokens in found
            and concurrency == required_concurrency
            and correlation == required_correlation
        ):
            found[input_tokens].append(candidate)

    failures = []
    selected = []
    for input_tokens in required_inputs:
        cases = found[input_tokens]
        if len(cases) != 1:
            failures.append(
                f"input_tokens={input_tokens}:expected_one_exact_case:found={len(cases)}"
            )
        else:
            selected.append(cases[0])
    if failures:
        raise H1H2CampaignError("; ".join(failures))
    return selected


def _plan(replay_contract_path: Path) -> dict[str, Any]:
    contract = load_replay_contract(replay_contract_path)
    return {
        "artifact_kind": "fenix_h1_h2_plan",
        "required_trace_cases": contract["required_trace_cases"],
        "h1": contract["h1"],
        "h2": contract["h2"],
        "measured_geometry": contract.get("measured_geometry"),
        "scientific_scope": contract["scientific_scope"],
    }


def run_pipeline(
    trace_root: Path,
    out_root: Path,
    campaign_path: Path,
    replay_contract_path: Path,
    *,
    expert_slot_bytes: int | None = None,
) -> dict[str, Any]:
    case_dirs = discover_exact_cases(trace_root, replay_contract_path)
    contract = load_replay_contract(replay_contract_path)
    topk = [int(value) for value in contract["h1"]["expert_concentration_topk"]]
    measured_geometry = contract.get("measured_geometry")
    if not isinstance(measured_geometry, dict):
        raise H1H2CampaignError("measured_geometry is missing from H1/H2 contract")
    if expert_slot_bytes is None:
        try:
            expert_slot_bytes = int(measured_geometry["expert_slot_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise H1H2CampaignError(
                "measured_geometry.expert_slot_bytes is invalid"
            ) from exc
    if expert_slot_bytes <= 0:
        raise H1H2CampaignError("expert slot bytes must be positive")

    h1_root = out_root / "h1"
    h2_root = out_root / "h2"
    h1_root.mkdir(parents=True, exist_ok=True)
    h2_root.mkdir(parents=True, exist_ok=True)

    h1_results = []
    case_counts = []
    source_commits: set[str] = set()
    source_images: set[str] = set()
    for case_dir in case_dirs:
        result = analyze_case(
            case_dir,
            campaign_path,
            topk_values=topk,
            explicit_expert_slot_bytes=expert_slot_bytes,
        )
        h1_results.append(result)
        destination = h1_root / f"{result['case_id']}.json"
        destination.write_text(json.dumps(result, indent=2) + "\n")
        commit = result.get("source", {}).get("repository_commit")
        image_id = result.get("source", {}).get("runtime_image_id")
        if isinstance(commit, str):
            source_commits.add(commit)
        if isinstance(image_id, str):
            source_images.add(image_id)
        case_counts.append(
            collect_case_counts(
                case_dir,
                campaign_path,
                explicit_expert_slot_bytes=expert_slot_bytes,
            )
        )

    if len(source_commits) != 1:
        raise H1H2CampaignError(
            f"cross-case repository provenance differs: {sorted(source_commits)}"
        )
    if len(source_images) != 1:
        raise H1H2CampaignError(
            f"cross-case runtime image provenance differs: {sorted(source_images)}"
        )

    h2_result = replay(case_counts, replay_contract_path)
    h2_path = h2_root / "edge_memory_replay.json"
    h2_path.write_text(json.dumps(h2_result, indent=2) + "\n")

    summary = {
        "schema_version": 1,
        "artifact_kind": "fenix_h1_h2_campaign_summary",
        "trace_root": str(trace_root),
        "trace_root_name": trace_root.name,
        "source_repository_commit": next(iter(source_commits)),
        "source_runtime_image_id": next(iter(source_images)),
        "campaign": {
            "path": str(campaign_path),
            "sha256": sha256_file(campaign_path),
        },
        "replay_contract": {
            "path": str(replay_contract_path),
            "sha256": sha256_file(replay_contract_path),
        },
        "measured_geometry": contract.get("measured_geometry"),
        "h1": {
            "evidence_kind": "local_measured_trace_analysis",
            "case_count": len(h1_results),
            "coverage_complete": all(
                result.get("h1_coverage_complete") is True
                for result in h1_results
            ),
            "cases": [
                {
                    "case_id": result["case_id"],
                    "input_tokens": result["case"]["input_tokens"],
                    "model_token_observations": result["measured_working_set"][
                        "model_token_observations"
                    ],
                    "conditional_unique_bytes": result["measured_working_set"][
                        "conditional_unique_bytes"
                    ],
                    "conditional_requested_bytes_per_model_token": result[
                        "measured_working_set"
                    ]["conditional_requested_bytes_per_model_token"],
                }
                for result in sorted(
                    h1_results,
                    key=lambda value: int(value["case"]["input_tokens"]),
                )
            ],
        },
        "h2": {
            "evidence_kind": "trace_projection",
            "budget_scope": h2_result["budget_scope"],
            "policy": h2_result["policy"],
            "holdout_summary": h2_result["holdout_summary"],
            "can_establish_h3": False,
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument(
        "--campaign", type=Path, default=Path("configs/campaign.json")
    )
    parser.add_argument(
        "--replay-contract",
        type=Path,
        default=Path("configs/edge_memory_replay_v1.json"),
    )
    parser.add_argument("--expert-slot-bytes", type=int)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    if args.plan:
        try:
            print(json.dumps(_plan(args.replay_contract), indent=2))
            return 0
        except (H2ReplayError, json.JSONDecodeError) as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 2

    if args.trace_root is None or args.out_root is None:
        print(
            json.dumps(
                {"error": "--trace-root and --out-root are required unless --plan"},
                indent=2,
            )
        )
        return 2

    try:
        summary = run_pipeline(
            args.trace_root,
            args.out_root,
            args.campaign,
            args.replay_contract,
            expert_slot_bytes=args.expert_slot_bytes,
        )
    except (
        H1AnalysisError,
        H2ReplayError,
        H1H2CampaignError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
