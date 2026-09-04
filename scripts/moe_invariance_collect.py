#!/usr/bin/env python3
"""Collect the predeclared small router-placement invariance trace case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis import moe_router_invariance
from scripts import check_h1_trace_server_policy, robust_trace_campaign


class InvarianceCollectError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-hot-experts", type=int, required=True)
    parser.add_argument(
        "--hotness-contract",
        type=Path,
        default=Path("configs/moe_hotness_validation_v1.json"),
    )
    parser.add_argument(
        "--robustness-contract",
        type=Path,
        default=Path("configs/h1_h2_workload_robustness_v1.json"),
    )
    parser.add_argument(
        "--model-campaign", type=Path, default=Path("configs/campaign.json")
    )
    parser.add_argument(
        "--runtime-lane", type=Path, default=Path("configs/runtime_lane.json")
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("external/workloads/h1_h2_workload_robustness_v1/corpus.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "external/workloads/h1_h2_workload_robustness_v1/source_manifest.json"
        ),
    )
    parser.add_argument("--settle-ms", type=int, default=500)
    args = parser.parse_args()

    try:
        policy = check_h1_trace_server_policy.verify(
            args.server_log, Path("configs/h1_h2_trace_execution_v1.json")
        )
        observed_hot = moe_router_invariance._hot_experts(args.server_log)
        if observed_hot != args.expected_hot_experts:
            raise InvarianceCollectError(
                f"server hot-expert setting differs: {observed_hot}!={args.expected_hot_experts}"
            )
        hotness = json.loads(args.hotness_contract.read_text())
        stratum = str(hotness["placement_invariance"]["stratum"])
        contract = robust_trace_campaign.load_contract(args.robustness_contract)
        if stratum not in contract["strata"]:
            raise InvarianceCollectError(f"undeclared robustness stratum: {stratum}")

        destination = robust_trace_campaign.run_stratum(
            root=Path.cwd().resolve(),
            stratum=stratum,
            contract_path=args.robustness_contract,
            model_campaign_path=args.model_campaign,
            runtime_lane_path=args.runtime_lane,
            corpus_path=args.corpus,
            source_manifest_path=args.source_manifest,
            server_log=args.server_log,
            out_root=args.out_root,
            ple_source=Path("traces/raw/ple_runtime.jsonl"),
            moe_source=Path("traces/raw/moe_runtime.jsonl"),
            url=robust_trace_campaign.DEFAULT_URL,
            model=robust_trace_campaign.DEFAULT_MODEL,
            tokenize_url=None,
            settle_ms=args.settle_ms,
        )
        evidence = json.loads((destination / "evidence.json").read_text())
        if evidence.get("trace_valid") is not True:
            raise InvarianceCollectError("collected invariance case is not trace_valid")
    except (
        InvarianceCollectError,
        moe_router_invariance.RouterInvarianceError,
        robust_trace_campaign.RobustTraceError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3

    print(json.dumps({
        "artifact_kind": "fenix_moe_router_invariance_collection",
        "stratum": stratum,
        "expected_hot_experts": args.expected_hot_experts,
        "prefix_caching": policy["prefix_caching"],
        "case_dir": str(destination),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
