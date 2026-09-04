#!/usr/bin/env python3
"""Run the FENIX H1/H2 workload-robustness analysis over frozen C=1 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analysis.h1_workload_robustness import (
    H1RobustnessError,
    analyze_stratum_internal,
    summarize_cross_stratum,
)
from analysis.h2_workload_robustness import H2RobustnessError, replay_robustness
from scripts.robust_trace_campaign import RobustTraceError, load_contract

DEFAULT_CONTRACT = Path("configs/h1_h2_workload_robustness_v1.json")
DEFAULT_MODEL_CAMPAIGN = Path("configs/campaign.json")


class RobustnessCampaignError(ValueError):
    """Raised when the robustness campaign cannot be promoted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RobustnessCampaignError(f"{path}: expected JSON object")
    return value


def discover_cases(trace_root: Path, contract_path: Path) -> list[Path]:
    if not trace_root.is_dir():
        raise RobustnessCampaignError(f"trace root does not exist: {trace_root}")
    contract = load_contract(contract_path)
    selected: list[Path] = []
    failures: list[str] = []
    for stratum in contract["trace"]["strata_order"]:
        case_dir = trace_root / f"s-{stratum}-r01"
        evidence_path = case_dir / "evidence.json"
        if not evidence_path.is_file():
            failures.append(f"missing_case:{stratum}")
            continue
        evidence = _load_json(evidence_path)
        if evidence.get("trace_valid") is not True:
            failures.append(f"invalid_case:{stratum}")
            continue
        case = evidence.get("case")
        if not isinstance(case, dict) or case.get("stratum") != stratum:
            failures.append(f"stratum_mismatch:{stratum}")
            continue
        if case.get("correlation_mode") != "exact_request_correlation":
            failures.append(f"correlation_mismatch:{stratum}")
            continue
        if int(case.get("concurrency", 0)) != 1:
            failures.append(f"concurrency_mismatch:{stratum}")
            continue
        selected.append(case_dir)
    if failures:
        raise RobustnessCampaignError("; ".join(failures))
    return selected


def _provenance(case_dirs: list[Path], contract_path: Path) -> dict[str, str]:
    commits: set[str] = set()
    image_ids: set[str] = set()
    corpus_hashes: set[str] = set()
    source_manifest_hashes: set[str] = set()
    contract_hash = sha256_file(contract_path)
    for case_dir in case_dirs:
        evidence = _load_json(case_dir / "evidence.json")
        if evidence.get("contract_sha256") != contract_hash:
            raise RobustnessCampaignError(
                f"{case_dir.name}: robustness contract hash differs"
            )
        commit = evidence.get("repository_commit")
        image_id = evidence.get("launch", {}).get("runtime_image_id")
        corpus_hash = evidence.get("frozen_corpus_sha256")
        manifest_hash = evidence.get("source_manifest_sha256")
        if not all(isinstance(value, str) and value for value in (commit, image_id, corpus_hash, manifest_hash)):
            raise RobustnessCampaignError(f"{case_dir.name}: provenance is incomplete")
        commits.add(commit)
        image_ids.add(image_id)
        corpus_hashes.add(corpus_hash)
        source_manifest_hashes.add(manifest_hash)
    for name, values in (
        ("repository_commit", commits),
        ("runtime_image_id", image_ids),
        ("frozen_corpus_sha256", corpus_hashes),
        ("source_manifest_sha256", source_manifest_hashes),
    ):
        if len(values) != 1:
            raise RobustnessCampaignError(
                f"cross-stratum {name} differs: {sorted(values)}"
            )
    return {
        "repository_commit": next(iter(commits)),
        "runtime_image_id": next(iter(image_ids)),
        "frozen_corpus_sha256": next(iter(corpus_hashes)),
        "source_manifest_sha256": next(iter(source_manifest_hashes)),
    }


def plan(contract_path: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    strata = contract["trace"]["strata_order"]
    return {
        "schema_version": 1,
        "artifact_kind": "fenix_h1_h2_workload_robustness_plan",
        "scientific_scope": contract["scientific_scope"],
        "primary_trace": {
            "strata": strata,
            "stratum_count": len(strata),
            "total_requests": sum(int(contract["strata"][name]["requests"]) for name in strata),
            "concurrency": contract["trace"]["concurrency"],
            "server_max_model_len": contract["trace"]["server_max_model_len"],
            "natural_prompt_policy": "no_synthetic_exact_length_padding",
        },
        "h1": contract["h1"],
        "h2": contract["h2"],
        "measured_geometry": contract["measured_geometry"],
        "extended_context_feasibility": contract["extended_context_feasibility"],
        "homogeneous_reference": {
            "role": contract["scientific_scope"]["homogeneous_v1_role"],
            "not_replaced": True,
        },
    }


def run_pipeline(
    trace_root: Path,
    out_root: Path,
    contract_path: Path,
    model_campaign_path: Path,
) -> dict[str, Any]:
    case_dirs = discover_cases(trace_root, contract_path)
    contract = load_contract(contract_path)
    provenance = _provenance(case_dirs, contract_path)

    h1_root = out_root / "h1"
    h2_root = out_root / "h2"
    h1_root.mkdir(parents=True, exist_ok=True)
    h2_root.mkdir(parents=True, exist_ok=True)

    analyses = {}
    for case_dir in case_dirs:
        internal = analyze_stratum_internal(case_dir, contract_path, model_campaign_path)
        stratum = str(internal.public["stratum"])
        analyses[stratum] = internal
        (h1_root / f"{stratum}.json").write_text(
            json.dumps(internal.public, indent=2, ensure_ascii=False) + "\n"
        )

    expected = list(contract["trace"]["strata_order"])
    if set(analyses) != set(expected):
        raise RobustnessCampaignError(
            f"H1 strata differ: observed={sorted(analyses)} expected={sorted(expected)}"
        )
    cross = summarize_cross_stratum(analyses)
    (h1_root / "cross_stratum.json").write_text(
        json.dumps(cross, indent=2, ensure_ascii=False) + "\n"
    )

    h2 = replay_robustness(case_dirs, contract_path, model_campaign_path)
    (h2_root / "robustness_replay.json").write_text(
        json.dumps(h2, indent=2, ensure_ascii=False) + "\n"
    )

    h1_compact = []
    for name in expected:
        row = analyses[name].public
        top128 = next(
            item for item in row["experts"]["concentration"]
            if int(item["topk_experts_per_layer"]) == 128
        )
        h1_compact.append(
            {
                "stratum": name,
                "request_count": row["request_count"],
                "model_token_observations": row["model_token_observations"],
                "ple_unique_rows": row["ple"]["unique_rows"],
                "ple_unique_row_fraction_of_table": row["ple"]["unique_row_fraction_of_table"],
                "expert_unique_fraction_of_all_layer_experts": row["experts"]["unique_fraction_of_all_layer_experts"],
                "expert_top128_mean_selection_fraction": top128["mean_selection_fraction"],
                "prefill_model_tokens": row["phase"]["prefill"]["model_tokens"],
                "decode_model_tokens": row["phase"]["decode"]["model_tokens"],
            }
        )

    summary = {
        "schema_version": 1,
        "artifact_kind": "fenix_h1_h2_workload_robustness_summary",
        "trace_root": str(trace_root),
        "trace_root_name": trace_root.name,
        "provenance": provenance,
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "model_campaign": {"path": str(model_campaign_path), "sha256": sha256_file(model_campaign_path)},
        "scientific_scope": contract["scientific_scope"],
        "h1": {
            "evidence_kind": "local_measured_trace_analysis",
            "coverage_complete": all(
                analyses[name].public.get("coverage_complete") is True for name in expected
            ),
            "stratum_count": len(expected),
            "cases": h1_compact,
            "cross_stratum_artifact": "h1/cross_stratum.json",
        },
        "h2": {
            "evidence_kind": "trace_projection",
            "budgets_gib": contract["h2"]["volatile_cache_budgets_gib"],
            "leave_one_domain_out_summary": h2["leave_one_domain_out_summary"],
            "mixed_online_summary": h2["mixed_online_summary"],
            "structural_holdout_count": len(h2["structural_holdouts"]),
            "can_establish_h3": False,
        },
        "extended_context_feasibility": contract["extended_context_feasibility"],
        "promotion_boundary": {
            "h1_h2_representative_workload_claim_requires_interpretation_of_this_robustness_suite": True,
            "h3_remains_unestablished": True,
            "proceed_to_feram_architecture": False,
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--model-campaign", type=Path, default=DEFAULT_MODEL_CAMPAIGN)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    try:
        if args.plan:
            print(json.dumps(plan(args.contract), indent=2))
            return 0
        if args.trace_root is None or args.out_root is None:
            raise RobustnessCampaignError(
                "--trace-root and --out-root are required unless --plan"
            )
        summary = run_pipeline(
            args.trace_root,
            args.out_root,
            args.contract,
            args.model_campaign,
        )
    except (
        RobustnessCampaignError,
        RobustTraceError,
        H1RobustnessError,
        H2RobustnessError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
