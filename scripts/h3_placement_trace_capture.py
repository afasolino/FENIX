#!/usr/bin/env python3
"""Capture exactly the three H3 placement-invariance trace strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import robust_trace_campaign

REQUIRED_STRATA = ("chat_en", "session", "long_context_8k")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=robust_trace_campaign.DEFAULT_CONTRACT)
    parser.add_argument("--model-campaign", type=Path, default=robust_trace_campaign.DEFAULT_MODEL_CAMPAIGN)
    parser.add_argument("--corpus", type=Path, default=robust_trace_campaign.DEFAULT_CORPUS)
    parser.add_argument("--source-manifest", type=Path, default=robust_trace_campaign.DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--runtime-lane", type=Path, default=robust_trace_campaign.DEFAULT_RUNTIME_LANE)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--ple-trace", type=Path, default=robust_trace_campaign.DEFAULT_PLE_TRACE)
    parser.add_argument("--moe-trace", type=Path, default=robust_trace_campaign.DEFAULT_MOE_TRACE)
    parser.add_argument("--url", default=robust_trace_campaign.DEFAULT_URL)
    parser.add_argument("--tokenize-url")
    parser.add_argument("--model", default=robust_trace_campaign.DEFAULT_MODEL)
    parser.add_argument("--settle-ms", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        contract = robust_trace_campaign.load_contract(args.contract)
        corpus, _ = robust_trace_campaign.load_corpus(args.corpus, args.source_manifest, contract)
        root = Path.cwd().resolve()
        if args.settle_ms < 0:
            raise robust_trace_campaign.RobustTraceError("--settle-ms must be >= 0")

        completed: list[str] = []
        skipped: list[str] = []
        with robust_trace_campaign.trace_capture.campaign_lock(args.out_root / ".trace-campaign.lock"):
            for stratum in REQUIRED_STRATA:
                case_id = f"s-{stratum}-r01"
                evidence_path = args.out_root / case_id / "evidence.json"
                if evidence_path.is_file() and args.resume:
                    evidence = robust_trace_campaign._load_json(evidence_path)
                    if (
                        evidence.get("trace_valid") is True
                        and evidence.get("case", {}).get("stratum") == stratum
                        and evidence.get("case", {}).get("concurrency") == 1
                    ):
                        skipped.append(case_id)
                        continue
                    raise robust_trace_campaign.RobustTraceError(
                        f"{case_id}: --resume found incompatible placement evidence"
                    )

                destination = robust_trace_campaign.run_stratum(
                    root=root,
                    stratum=stratum,
                    contract_path=args.contract,
                    model_campaign_path=args.model_campaign,
                    runtime_lane_path=args.runtime_lane,
                    corpus_path=args.corpus,
                    source_manifest_path=args.source_manifest,
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

        failures: list[str] = []
        cases: dict[str, str] = {}
        for stratum in REQUIRED_STRATA:
            case_dir = args.out_root / f"s-{stratum}-r01"
            evidence_path = case_dir / "evidence.json"
            if not evidence_path.is_file():
                failures.append(f"missing:{stratum}")
                continue
            evidence = robust_trace_campaign._load_json(evidence_path)
            if evidence.get("trace_valid") is not True:
                failures.append(f"invalid:{stratum}")
            if evidence.get("case", {}).get("stratum") != stratum:
                failures.append(f"stratum:{stratum}")
            cases[stratum] = str(case_dir.resolve())

        result = {
            "artifact_kind": "fenix_h3_placement_trace_capture_summary",
            "required_strata": list(REQUIRED_STRATA),
            "complete": not failures and len(cases) == len(REQUIRED_STRATA),
            "completed": completed,
            "skipped": skipped,
            "cases": cases,
            "failures": failures,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["complete"] else 3
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
