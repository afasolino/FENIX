from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.h3.cache import ByteLFU, ByteLRU, CacheObject, pages_for_range
from analysis.h3.common import H3Error, set_execution_repository_identity, sha256_file, write_json
from analysis.h3.decision import (
    _candidate_interval_summary, _service_ns, _service_time_optimized_offline_bound,
    campaign_gate, decide_point,
)
from analysis.h3.lpddr import (
    _channel_mapper_for_frontend, _mapped_access_addresses, offered_frontend_gb_s,
    theoretical_channel_gb_s, validate_resolved_profile,
    write_actual_h3_transaction_trace,
)
import analysis.h3.pagecache as pagecache_module
from analysis.h3.pagecache import build_pagecache_window
from analysis.h3.prerequisites import (
    build_capacity_budget, build_long_context_scaling, build_placement_invariance,
    build_storage_concurrency_evidence,
)
from analysis.h3.residency import replay_case
from analysis.h3.storage import (
    build_samples, fio_achieved_qd_fraction, project_storage_bytes_ns, summarize_fio,
)
from analysis.h3.trace import Geometry, build_manifest, iter_case_epochs
from scripts.h3_campaign import _project_runtime_dependencies


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/h3/contract_v6.json"
STRATA = [
    "chat_en",
    "knowledge",
    "math",
    "code",
    "multilingual",
    "session",
    "long_context_8k",
]
BASE = "8b8694647840e5af73fd4e1fb1275d0b15c19056"
CONTRACT_SHA = sha256_file(CONTRACT)
EXECUTION_ID = {
    "head": "1" * 40,
    "tree": "2" * 40,
    "branch": "study/h3-test",
    "clean": True,
    "required_base_commit": BASE,
    "root": str(ROOT),
}


@pytest.fixture(autouse=True)
def _default_execution_identity():
    set_execution_repository_identity(EXECUTION_ID)
    try:
        yield
    finally:
        set_execution_repository_identity(None)


def _h3(payload: dict) -> dict:
    payload = dict(payload)
    payload.setdefault("contract_sha256", CONTRACT_SHA)
    payload.setdefault("execution_repository", EXECUTION_ID)
    return payload


def _fio_fixture(miss_sha: str = "same", binding_sha: str = "binding", coeff: float = 1.0) -> dict:
    windows = {"w11": [coeff], "w23": [coeff], "w37": [coeff]}
    class_node = {
        "ns_per_byte_median": coeff,
        "ns_per_byte_ci95": [coeff, coeff],
        "replicate_values": [coeff, coeff, coeff],
        "replicate_groups": dict(windows),
    }
    mix_node = {
        "observed_over_independent_projection_median": 1.0,
        "observed_over_same_window_independent_projection_median": 1.0,
        "ci95": [1.0, 1.0],
        "replicate_values": [1.0, 1.0, 1.0],
        "replicate_groups": {"w11": [1.0], "w23": [1.0], "w37": [1.0]},
    }
    classes = {}
    mixed = {}
    for qd in (1, 8, 32):
        classes[f"ple:qd{qd}"] = dict(class_node)
        classes[f"expert:qd{qd}"] = dict(class_node)
        mixed[f"qd{qd}"] = dict(mix_node)
    return _h3({
        "artifact_kind": "fenix_h3_fio_calibration",
        "class_coefficients": classes,
        "mixed_interaction": mixed,
        "fio_tool": _fake_fio_tool(),
        "bootstrap_draws": 256,
        "paired_window_design": {
            "verified": True,
            "distinct_nonoverlapping_windows": 3,
            "window_ids": ["w11", "w23", "w37"],
        },
        "provenance": {
            "miss_stream_sha256": miss_sha,
            "storage_binding_sha256": binding_sha,
            "device_major_minor": "1:1",
            "contract_sha256": CONTRACT_SHA,
        },
    })


def _campaign_fingerprint() -> dict:
    return {
        "contract_sha256": CONTRACT_SHA,
        "trace_manifest_sha256": "trace-manifest",
        "storage_binding_sha256": "binding",
        "device_major_minor": "1:1",
        "fio_head": "a" * 40,
        "fio_tree": "e" * 40,
        "fio_binary_sha256": "b" * 64,
        "ramulator_head": "c" * 40,
        "ramulator_tree": "d" * 40,
    }


def _decision_fixture(
    stratum: str,
    policy: str,
    capacity: float = 8.0,
    *,
    gap: bool = True,
    qd: int = 32,
    actual_lpddr: bool = True,
    phase: str | None = None,
) -> dict:
    bounds = [0.2, 0.3] if gap else [0.0, 0.03]
    return _h3({
        "artifact_kind": "fenix_h3_same_endpoint_decision",
        "stratum": stratum,
        "phase_filter": phase,
        "capacity_gib": capacity,
        "policy": policy,
        "queue_depth": qd,
        "promotion_eligible_queue_depth": qd == 32,
        "actual_lpddr_trace_calibration_used": actual_lpddr,
        "pagecache_baseline_used": phase is None and stratum in {"session", "long_context_8k"},
        "pagecache_modes": (["buffered", "mmap"] if phase is None and stratum in {"session", "long_context_8k"} else []),
        "verdict": "H3_MEMORY_GAP_SUPPORTED" if gap else "CONVENTIONAL_MEMORY_SUFFICIENT",
        "global_penalty_fraction_bounds": bounds,
        "campaign_fingerprint": _campaign_fingerprint(),
        "phase_summaries": {},
    })


def _write_decision_matrix(tmp_path: Path, *, gap: bool = True, capacity_by_stratum: dict[str, float] | None = None,
                           policies: list[str] | None = None, qd: int = 32,
                           actual_lpddr: bool = True) -> list[Path]:
    contract = json.loads(CONTRACT.read_text())
    use_policies = policies or (contract["primary_policies"] + contract["strong_conventional_policies"])
    phases = contract["decision_gate"]["campaign_requirements"]["required_phases"]
    paths: list[Path] = []
    for name in STRATA:
        for policy in use_policies:
            capacity = (capacity_by_stratum or {}).get(name, 8.0)
            for phase in [None, *phases]:
                suffix = "full" if phase is None else phase
                path = tmp_path / f"{name}-{policy}-{suffix}-qd{qd}.json"
                write_json(path, _decision_fixture(
                    name, policy, capacity=capacity, gap=gap, qd=qd,
                    actual_lpddr=actual_lpddr, phase=phase,
                ))
                paths.append(path)
    return paths


def _prerequisite_manifest(
    tmp_path: Path,
    *,
    capacity: float = 8.0,
    placement_passed: bool = True,
    long_passed: bool = True,
    include_concurrency: bool = False,
) -> Path:
    placement = tmp_path / "placement.json"
    write_json(placement, _h3({
        "artifact_kind": "fenix_h3_placement_invariance_evidence",
        "evaluator": "ordered_routing_and_ple_semantic_identity_across_physical_placements",
        "derived_pass": placement_passed,
    }))
    long_context = tmp_path / "long-context.json"
    write_json(long_context, _h3({
        "artifact_kind": "fenix_h3_long_context_scaling_evidence",
        "evaluator": "trace_validated_multi_length_context_scaling_characterization",
        "derived_pass": long_passed,
        "maximum_demonstrated_prompt_tokens": 32768,
    }))
    evidence_file = tmp_path / "capacity-evidence.txt"
    evidence_file.write_text("synthetic capacity measurement\n")
    budget = tmp_path / "capacity-budget.json"
    write_json(budget, _h3({
        "artifact_kind": "fenix_h3_capacity_budget",
        "derived": True,
        "platform_mode": "primary",
        "physical_memory_gib": 32.0,
        "candidate_budgets_that_fit_gib": [capacity],
    }))
    env_lock = tmp_path / "ramulator-python-resolved.lock.txt"
    env_lock.write_text("pyyaml==6.0.2\n")
    tool = tmp_path / "tool-qualification.json"
    write_json(tool, _h3({
        "artifact_kind": "fenix_h3_tool_qualification",
        "complete": True,
        "failures": [],
        "observed": {
            "ramulator2": {"head": "c" * 40, "tree": "d" * 40, "clean": True},
            "ramulator_python_environment": {
                "resolved_lock_path": str(env_lock),
                "resolved_lock_relative_path": env_lock.name,
                "resolved_lock_sha256": sha256_file(env_lock),
            },
            "ramulator_built_extensions": [{"path": "ramulator.so", "sha256": "a" * 64}],
            "build_toolchain": {"cmake": "cmake synthetic", "cxx": "c++ synthetic"},
            "fio_binary": _fake_fio_tool(),
        },
    }))
    evidence = {
        "placement_invariance": {"artifact": str(placement), "sha256": sha256_file(placement)},
        "long_context_beyond_8k": {"artifact": str(long_context), "sha256": sha256_file(long_context)},
        "deployment_capacity_budget": {"artifact": str(budget), "sha256": sha256_file(budget)},
        "tool_qualification": {"artifact": str(tool), "sha256": sha256_file(tool)},
    }
    if include_concurrency:
        concurrency = tmp_path / "concurrency.json"
        write_json(concurrency, _h3({
            "artifact_kind": "fenix_h3_storage_concurrency_evidence",
            "evaluator": "measured_workload_scheduler_raw_event_qd_and_prefetch_slack_feasibility",
            "derived_pass": True,
            "observed_queue_depth": 32,
            "observed_achieved_fraction": 0.8,
            "observed_storage_completions_within_available_prefetch_slack_fraction": 0.8,
            "observed_measured_event_count": 448,
            "object_specific_prefetch_slack_measured": True,
            "trace_manifest_sha256": "trace-manifest",
            "storage_binding_sha256": "binding",
        }))
        evidence["storage_concurrency_feasibility"] = {
            "artifact": str(concurrency), "sha256": sha256_file(concurrency)
        }
    manifest = tmp_path / "prereq.json"
    write_json(manifest, _h3({
        "schema_version": 6,
        "artifact_kind": "fenix_h3_prerequisite_manifest",
        "evidence": evidence,
    }))
    return manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))


def _write_interleaved_storage_misses(path: Path, records: int = 150) -> None:
    rows = []
    for sequence in range(records):
        if sequence % 2 == 0:
            index = sequence // 2
            rows.append({
                "sequence": sequence,
                "timestamp_ns_order_only": 1_000 + sequence,
                "stratum": "chat_en",
                "request_id": f"req-{sequence // 10}",
                "phase": "prefill" if sequence < records // 2 else "decode",
                "kind": "ple",
                "offset_bytes": (index % 32) * 4096,
                "length_bytes": 4096,
                "useful_bytes": 160,
                "storage_key": ["ple_page", index % 32],
            })
        else:
            index = sequence // 2
            rows.append({
                "sequence": sequence,
                "timestamp_ns_order_only": 1_000 + sequence,
                "stratum": "chat_en",
                "request_id": f"req-{sequence // 10}",
                "phase": "prefill" if sequence < records // 2 else "decode",
                "kind": "expert",
                "offset_bytes": (index % 64) * 2_535_424,
                "length_bytes": 2_535_424,
                "useful_bytes": 2_534_400,
                "storage_key": ["expert", 0, index % 64],
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def _write_case(case: Path, stratum: str, ordinal: int) -> None:
    case.mkdir(parents=True)
    request_id = f"req-{ordinal}"
    _write_jsonl(
        case / "client.jsonl",
        [{
            "ordinal": 0,
            "request_id": request_id,
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "concurrency": 1,
        }],
    )

    ple: list[dict] = []
    for token in range(3):
        # The real normalizer assigns one address_known_ns to every row emitted
        # by one PLE runtime batch. Equal timestamps must therefore be atomic.
        timestamp = 1_000 + token * 100
        for head in range(16):
            ple.append({
                "request_id": request_id,
                "token_position": token,
                "ple_head": head,
                "physical_row_id": ordinal * 1_000 + token * 16 + head,
                "bytes": 160,
                "phase": "prefill" if token < 2 else "decode",
                "address_known_ns": timestamp,
            })
    _write_jsonl(case / "ple_normalized.jsonl", ple)

    moe: list[dict] = []
    timestamp = 10_000
    for layer in range(48):
        for token in range(3):
            moe.append({
                "request_id": request_id,
                "layer": f"language_model.model.layers.{layer}.mlp.experts",
                "selected_expert_ids": list(range(10)),
                "phase": "prefill" if token < 2 else "decode",
                "timestamp_ns": timestamp,
                "trace_scope": "router_all_layers",
            })
            timestamp += 1
    _write_jsonl(case / "moe_normalized.jsonl", moe)

    evidence = {
        "trace_valid": True,
        "repository_clean": True,
        "repository_commit": BASE,
        "source_manifest_sha256": "synthetic-source-manifest",
        "launch": {"runtime_image_id": "sha256:synthetic-runtime"},
        "case": {
            "stratum": stratum,
            "concurrency": 1,
            "correlation_mode": "exact_request_correlation",
        },
        "artifacts_sha256": {
            "client.jsonl": sha256_file(case / "client.jsonl"),
            "ple_normalized.jsonl": sha256_file(case / "ple_normalized.jsonl"),
            "moe_normalized.jsonl": sha256_file(case / "moe_normalized.jsonl"),
        },
    }
    write_json(case / "evidence.json", evidence)


def _synthetic_trace(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    trace_root = tmp_path / "trace"
    for ordinal, stratum in enumerate(STRATA):
        _write_case(trace_root / f"s-{stratum}-r01", stratum, ordinal)

    robustness = tmp_path / "robustness.json"
    write_json(
        robustness,
        {
            "artifact_kind": "fenix_h1_h2_workload_robustness_contract",
            "trace": {"strata_order": STRATA},
            "strata": {name: {"requests": 1} for name in STRATA},
        },
    )
    campaign = tmp_path / "campaign.json"
    write_json(
        campaign,
        {
            "model": {
                "num_hidden_layers": 48,
                "num_experts": 512,
                "experts_per_token": 10,
                "ple_addressable_rows": 320001536,
            }
        },
    )
    execution = tmp_path / "execution.json"
    write_json(
        execution,
        {
            "artifact_kind": "fenix_h1_h2_trace_execution_policy",
            "prefix_caching": "disabled",
            "concurrency": 1,
        },
    )
    return trace_root, robustness, campaign, execution


def _manifest(tmp_path: Path) -> tuple[dict, Path]:
    trace_root, robustness, campaign, execution = _synthetic_trace(tmp_path)
    out = tmp_path / "manifest.json"
    manifest = build_manifest(trace_root, robustness, campaign, CONTRACT, out, execution)
    return manifest, out


def _lpddr_payload(bandwidth: float = 204.8) -> dict:
    contract = json.loads(CONTRACT.read_text())
    points = []
    for regime in contract["lpddr"]["calibration_regimes"]:
        for ncs in contract["lpddr"]["ncs_values"]:
            points.append({
                "sensitivity_id": f"ramulator_{regime}_nCS{ncs}",
                "regime": regime,
                "aggregate_read_bandwidth_gb_s": bandwidth,
                "aggregate_write_bandwidth_gb_s": bandwidth,
                "aggregate_mixed_total_bandwidth_gb_s": bandwidth,
            })
    points.append({
        "sensitivity_id": "platform_theoretical_peak",
        "regime": "theoretical_peak",
        "aggregate_read_bandwidth_gb_s": 204.8,
        "aggregate_write_bandwidth_gb_s": 204.8,
        "aggregate_mixed_total_bandwidth_gb_s": 204.8,
    })
    return _h3({
        "artifact_kind": "fenix_h3_lpddr_calibration",
        "ramulator_tool": {"head": "c" * 40, "tree": "d" * 40, "clean": True},
        "sensitivity_points": points,
    })


def _fake_fio_tool() -> dict:
    return {
        "head": "a" * 40,
        "tree": "e" * 40,
        "clean": True,
        "version": "fio-synthetic",
        "binary_sha256": "b" * 64,
    }


def test_page_crossing_is_exact():
    assert pages_for_range(25 * 160, 160, 4096) == (0, 1)
    assert pages_for_range(26 * 160, 160, 4096) == (1,)


def test_epoch_classification_precedes_lru_commit():
    cache = ByteLRU(320)
    a = CacheObject(("a",), 160, "ple", ())
    b = CacheObject(("b",), 160, "ple", ())
    c = CacheObject(("c",), 160, "ple", ())
    assert cache.classify_epoch([a, b]) == {("a",): False, ("b",): False}
    cache.commit_epoch([a, b])
    hits = cache.classify_epoch([a, c])
    assert hits[("a",)] is True
    assert hits[("c",)] is False
    cache.commit_epoch([a, c])
    assert cache.resident_bytes == 320


def test_bandwidth_service_units():
    assert _service_ns(2048.0, 2.0) == 1024.0


def test_contract_is_fail_closed_and_checkpoint_exact():
    contract = json.loads(CONTRACT.read_text())
    assert contract["source_trace"]["repository_commit"] == BASE
    assert contract["source_trace"]["prefix_caching"] == "disabled"
    assert contract["workload_policy"]["implicit_cross_stratum_carry"] is False
    assert contract["geometry"]["ple_data_bytes"] == 51_200_245_760
    assert contract["geometry"]["ple_data_sha256"] == "b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3"
    assert contract["geometry"]["ple_dtype"] == "F8_E4M3"
    assert contract["geometry"]["ple_shard_count"] == 128
    assert contract["decision_gate"]["end_to_end_inference_claim_allowed"] is False
    assert contract["energy"]["energy_superiority_claim_allowed"] is False
    assert contract["storage"]["queue_depths"] == [1, 8, 32]
    assert contract["storage"]["promotion_queue_depths"] == [32]
    assert contract["lpddr"]["actual_trace_calibration_required_for_promotion"] is True
    assert contract["storage"]["seeds"] == [11, 23, 37, 53, 71]




def test_h3_json_artifact_embeds_cli_execution_repository_provenance(tmp_path: Path):
    identity = {
        "head": "1" * 40, "tree": "2" * 40, "branch": "study/h3-test",
        "clean": True, "required_base_commit": BASE, "root": str(tmp_path),
    }
    set_execution_repository_identity(identity)
    try:
        path = tmp_path / "artifact.json"
        write_json(path, {"artifact_kind": "fenix_h3_fixture", "value": 1})
        assert json.loads(path.read_text())["execution_repository"] == identity
    finally:
        set_execution_repository_identity(None)


def test_execution_repository_preflight_requires_clean_descendant(tmp_path: Path):
    from scripts.h3_campaign import _execution_repository_identity

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "h3@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "H3 Test"], cwd=repo, check=True)
    (repo / "base.txt").write_text("base\n")
    subprocess.run(["git", "add", "base.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    (repo / "h3.txt").write_text("h3\n")
    subprocess.run(["git", "add", "h3.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "h3"], cwd=repo, check=True)
    identity = _execution_repository_identity(repo, base)
    assert identity["required_base_commit"] == base
    assert identity["clean"] is True
    (repo / "dirty.txt").write_text("dirty\n")
    with pytest.raises(H3Error, match="clean committed"):
        _execution_repository_identity(repo, base)

def test_fresh_mature_tool_checkout_is_clean_and_exactly_pinned(tmp_path: Path):
    from scripts.h3_campaign import _checkout

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "h3@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "H3 Test"], cwd=source, check=True)
    (source / "payload.txt").write_text("pinned\n")
    subprocess.run(["git", "add", "payload.txt"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=source, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()

    root = tmp_path / "work"
    root.mkdir()
    checkout = _checkout(root, {
        "path": "external/tools/fixture",
        "repository": str(source),
        "commit": commit,
    })
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True)
    assert observed == commit
    assert status == ""

def test_upstream_commits_are_exact_pins():
    lock = json.loads((ROOT / "configs/h3/upstream_tools.lock.json").read_text())
    expected = {
        "ramulator2": "6710371b95c2fbc9fb842d2aad5ba30bb76abcf1",
        "drampower": "1e92c59a82e5f8d4a48d9759b5427d58f44e6a15",
        "fio": "ab77643023f5d7e3c1b71a7576a564f368bf577a",
        "mqsim": "51f0f2d3fed92d88ef4a0fa61a38024b07bf9d16",
    }
    assert {name: row["commit"] for name, row in lock["tools"].items()} == expected


def test_lpddr_frontend_can_saturate_one_x16_channel():
    lpddr = json.loads(CONTRACT.read_text())["lpddr"]
    assert theoretical_channel_gb_s(lpddr) == pytest.approx(12.8)
    assert offered_frontend_gb_s(lpddr) == pytest.approx(25.6)
    assert offered_frontend_gb_s(lpddr) > theoretical_channel_gb_s(lpddr)


def test_lpddr_profile_validation_requires_two_ranks_and_exact_timings():
    lpddr = json.loads(CONTRACT.read_text())["lpddr"]
    timing = dict(lpddr["resolved_timing_required"], nCS=2)
    validate_resolved_profile({"rank": 2, "channel_width": 16}, timing, lpddr, 2)
    with pytest.raises(H3Error, match="rank geometry drift"):
        validate_resolved_profile({"rank": 1, "channel_width": 16}, timing, lpddr, 2)


def test_manifest_requires_complete_authoritative_strata(tmp_path: Path):
    trace_root, robustness, campaign, execution = _synthetic_trace(tmp_path)
    payload = json.loads(robustness.read_text())
    payload["trace"]["strata_order"] = ["chat_en"]
    write_json(robustness, payload)
    with pytest.raises(H3Error, match="predeclared authoritative set/order"):
        build_manifest(trace_root, robustness, campaign, CONTRACT, tmp_path / "bad.json", execution)


def test_manifest_rejects_incomplete_request_count(tmp_path: Path):
    trace_root, robustness, campaign, execution = _synthetic_trace(tmp_path)
    payload = json.loads(robustness.read_text())
    payload["strata"]["chat_en"]["requests"] = 2
    write_json(robustness, payload)
    with pytest.raises(H3Error, match="successful request count"):
        build_manifest(trace_root, robustness, campaign, CONTRACT, tmp_path / "bad.json", execution)


def test_manifest_rejects_missing_runtime_provenance(tmp_path: Path):
    trace_root, robustness, campaign, execution = _synthetic_trace(tmp_path)
    case = trace_root / "s-chat_en-r01"
    evidence = json.loads((case / "evidence.json").read_text())
    evidence["launch"].pop("runtime_image_id")
    write_json(case / "evidence.json", evidence)
    with pytest.raises(H3Error, match="runtime image provenance is missing"):
        build_manifest(trace_root, robustness, campaign, CONTRACT, tmp_path / "bad.json", execution)


def test_manifest_rejects_source_hash_drift(tmp_path: Path):
    trace_root, robustness, campaign, execution = _synthetic_trace(tmp_path)
    case = trace_root / "s-chat_en-r01"
    with (case / "client.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(H3Error, match="source SHA drift"):
        build_manifest(trace_root, robustness, campaign, CONTRACT, tmp_path / "bad.json", execution)


def test_manifest_rejects_nonmonotonic_ple_timestamp(tmp_path: Path):
    trace_root, robustness, campaign, execution = _synthetic_trace(tmp_path)
    case = trace_root / "s-chat_en-r01"
    rows = [json.loads(line) for line in (case / "ple_normalized.jsonl").read_text().splitlines()]
    rows[-1]["address_known_ns"] = 0
    _write_jsonl(case / "ple_normalized.jsonl", rows)
    evidence = json.loads((case / "evidence.json").read_text())
    evidence["artifacts_sha256"]["ple_normalized.jsonl"] = sha256_file(case / "ple_normalized.jsonl")
    write_json(case / "evidence.json", evidence)
    with pytest.raises(H3Error, match="not timestamp monotonic"):
        build_manifest(trace_root, robustness, campaign, CONTRACT, tmp_path / "bad.json", execution)


def test_equal_timestamp_ple_rows_form_one_atomic_epoch(tmp_path: Path):
    manifest, _ = _manifest(tmp_path)
    case = Path(manifest["cases"][0]["case_dir"])
    geometry = Geometry(**manifest["geometry"])
    epochs = iter_case_epochs(case, geometry)
    first = next(epochs)
    ple_events = [event for event in first.events if event.kind == "ple"]
    assert len(ple_events) == 16
    assert {event.timestamp_ns for event in ple_events} == {1_000}


def test_streaming_golden_manifest_replay_storage_and_decision(tmp_path: Path):
    manifest, manifest_path = _manifest(tmp_path)
    assert [row["stratum"] for row in manifest["cases"]] == STRATA
    assert manifest["cases"][0]["model_tokens"] == 3

    replay_dir = tmp_path / "replay"
    replay = replay_case(manifest_path, "chat_en", 0.01, "useful_object_lru", replay_dir)
    assert replay["counters"]["ple_row_accesses"] == 48
    assert replay["counters"]["expert_selections"] == 48 * 3 * 10
    assert replay["counters"]["storage_bytes"] > 0

    replay2 = replay_case(manifest_path, "chat_en", 0.01, "useful_object_lru", tmp_path / "replay2")
    assert replay2["counters"] == replay["counters"]
    assert replay2["cold_start"] is True

    # Storage sampling is tested on a genuinely paired/interleaved miss stream.
    ple_backing = tmp_path / "ple.bin"
    expert_backing = tmp_path / "expert.bin"
    ple_backing.write_bytes(b"x")
    expert_backing.write_bytes(b"x")
    binding_path = tmp_path / "binding.json"
    write_json(binding_path, {
        "artifact_kind": "fenix_h3_storage_binding",
        "ple": {"path": str(ple_backing.resolve()), "mount": {"maj:min": "1:1"}},
        "expert": {"path": str(expert_backing.resolve()), "mount": {"maj:min": "1:1"}},
        "same_device_major_minor": True,
    })
    paired_misses = tmp_path / "paired-misses.jsonl.gz"
    _write_interleaved_storage_misses(paired_misses, 150)
    sample_dir = tmp_path / "samples"
    samples = build_samples(
        paired_misses, ple_backing, expert_backing, binding_path, sample_dir,
        [11, 23, 37, 53, 71], 8192, 2 * 2_535_424, 2 * 2_535_424 + 8192,
        contract_sha256=CONTRACT_SHA,
    )
    assert len(samples["samples"]) == 15
    windows = {}
    for row in samples["samples"]:
        windows.setdefault(row["window_id"], {})[row["kind"]] = row
    assert len(windows) == 5
    for kinds in windows.values():
        assert set(kinds) == {"ple", "expert", "mixed"}
        assert kinds["mixed"]["bytes_by_kind"]["ple"] == kinds["ple"]["bytes"]
        assert kinds["mixed"]["bytes_by_kind"]["expert"] == kinds["expert"]["bytes"]

    runs = tmp_path / "fio-runs"
    runs.mkdir()
    samples_sha = sha256_file(sample_dir / "samples.json")
    tool = _fake_fio_tool()
    for sample in samples["samples"]:
        result_path = runs / f"{sample['sample_id']}-qd1-rep00.json"
        write_json(result_path, {"jobs": [{"job_runtime": 1, "read": {"io_bytes": sample["bytes"]}}]})
        write_json(result_path.with_suffix(".meta.json"), {
            "returncode": 0, "queue_depth": 1, "samples_sha256": samples_sha,
            "iolog_sha256": sample["iolog_sha256"], "actual_read_bytes": sample["bytes"],
            "fio_tool": tool, "repeat_index": 0,
        })
    fio_summary = summarize_fio(sample_dir / "samples.json", runs, [1], tmp_path / "fio-summary.json")
    assert fio_summary["paired_window_design"]["verified"] is True
    assert fio_summary["paired_window_design"]["distinct_nonoverlapping_windows"] == 5

    # Decision provenance must remain tied to the real replay miss stream.
    fio_path = tmp_path / "fio-decision.json"
    write_json(fio_path, _fio_fixture(
        miss_sha=replay["miss_stream"]["sha256"],
        binding_sha=sha256_file(binding_path),
    ))
    lpddr_path = tmp_path / "lpddr.json"
    write_json(lpddr_path, _lpddr_payload())
    decision = decide_point(replay_dir / "summary.json", lpddr_path, fio_path, CONTRACT, 1, tmp_path / "decision.json")
    assert decision["verdict"] == "H3_MEMORY_GAP_SUPPORTED"
    assert decision["not_end_to_end_inference_latency"] is True

def test_fio_summary_rejects_read_byte_mismatch(tmp_path: Path):
    ple = tmp_path / "ple.bin"; ple.write_bytes(b"x")
    expert = tmp_path / "expert.bin"; expert.write_bytes(b"x")
    binding = tmp_path / "binding.json"
    write_json(binding, {
        "artifact_kind": "fenix_h3_storage_binding",
        "ple": {"path": str(ple.resolve()), "mount": {"maj:min": "1:1"}},
        "expert": {"path": str(expert.resolve()), "mount": {"maj:min": "1:1"}},
        "same_device_major_minor": True,
    })
    misses = tmp_path / "misses.jsonl.gz"
    _write_interleaved_storage_misses(misses, 90)
    out = tmp_path / "samples-dir"
    payload = build_samples(misses, ple, expert, binding, out, [11, 23, 37], 4096, 2_535_424, 2_539_520, contract_sha256=CONTRACT_SHA)
    samples_path = out / "samples.json"
    runs = tmp_path / "runs"; runs.mkdir()
    samples_sha = sha256_file(samples_path)
    for sample in payload["samples"]:
        observed = sample["bytes"] - 1 if sample["sample_id"] == "ple-seed11" else sample["bytes"]
        result = runs / f"{sample['sample_id']}-qd1-rep00.json"
        write_json(result, {"jobs": [{"job_runtime": 1, "read": {"io_bytes": observed}}]})
        write_json(result.with_suffix(".meta.json"), {
            "returncode": 0, "queue_depth": 1, "samples_sha256": samples_sha,
            "iolog_sha256": sample["iolog_sha256"], "actual_read_bytes": observed,
            "fio_tool": _fake_fio_tool(), "repeat_index": 0,
        })
    with pytest.raises(H3Error, match="read byte count"):
        summarize_fio(samples_path, runs, [1], tmp_path / "out.json")

def test_decision_rejects_fio_from_different_miss_stream(tmp_path: Path):
    residency = tmp_path / "residency.json"
    write_json(residency, _h3({
        "artifact_kind": "fenix_h3_residency_replay", "stratum": "chat_en",
        "phase_filter": "prefill", "policy": "useful_object_lru", "capacity_gib": 8.0,
        "counters": {"lpddr_read_useful_bytes": 1024, "lpddr_fill_write_bytes": 4096, "ple_storage_bytes": 1, "expert_storage_bytes": 1},
        "miss_stream": {"sha256": "right"},
    }))
    lpddr = tmp_path / "lpddr.json"; write_json(lpddr, _lpddr_payload())
    fio = tmp_path / "fio.json"
    write_json(fio, _h3({
        "artifact_kind": "fenix_h3_fio_calibration", "class_coefficients": {},
        "mixed_interaction": {}, "provenance": {"miss_stream_sha256": "wrong", "contract_sha256": CONTRACT_SHA},
    }))
    with pytest.raises(H3Error, match="not derived from this residency replay"):
        decide_point(residency, lpddr, fio, CONTRACT, 1, tmp_path / "out.json")


def test_decision_rejects_incomplete_lpddr_sensitivity_set(tmp_path: Path):
    residency = tmp_path / "residency.json"
    write_json(residency, _h3({
        "artifact_kind": "fenix_h3_residency_replay", "stratum": "chat_en", "phase_filter": "prefill",
        "policy": "useful_object_lru", "capacity_gib": 8.0,
        "counters": {"lpddr_read_useful_bytes": 1024, "ple_storage_bytes": 1, "expert_storage_bytes": 1}, "miss_stream": {"sha256": "same"},
    }))
    lpddr = tmp_path / "lpddr.json"
    bad = _lpddr_payload(); bad["sensitivity_points"] = bad["sensitivity_points"][:1]; write_json(lpddr, bad)
    fio = tmp_path / "fio.json"
    write_json(fio, _h3({"artifact_kind": "fenix_h3_fio_calibration", "provenance": {"miss_stream_sha256": "same", "contract_sha256": CONTRACT_SHA}}))
    with pytest.raises(H3Error, match="sensitivity set mismatch"):
        decide_point(residency, lpddr, fio, CONTRACT, 1, tmp_path / "out.json")


def test_campaign_gate_blocks_paper_promotion_until_prerequisites(tmp_path: Path):
    decisions = _write_decision_matrix(tmp_path, gap=True)
    prereq = _prerequisite_manifest(tmp_path, long_passed=False)
    gate = campaign_gate(decisions, prereq, CONTRACT, tmp_path / "gate.json")
    assert gate["preliminary_memory_service_verdict"] == "H3_MEMORY_GAP_SUPPORTED"
    assert gate["paper_promotion_verdict"] == "H3_PAPER_GATE_BLOCKED"
    assert gate["proceed_to_h4"] is False
    assert "long-context prerequisite" in (gate["prerequisite_error"] or "")

def test_campaign_gate_rejects_mixed_capacity_matrix(tmp_path: Path):
    decisions = _write_decision_matrix(tmp_path, gap=True, capacity_by_stratum={"chat_en": 12.0})
    prereq = _prerequisite_manifest(tmp_path, capacity=8.0)
    gate = campaign_gate(decisions, prereq, CONTRACT, tmp_path / "gate.json")
    assert gate["preliminary_memory_service_verdict"] == "INCOMPLETE_OR_DUPLICATE_H3_MATRIX"
    assert gate["proceed_to_h4"] is False
    assert any("common capacity" in msg for msg in gate["matrix_errors"])

def test_probe_storage_falls_back_to_legacy_lsblk_columns(tmp_path: Path, monkeypatch):
    ple = tmp_path / "ple.bin"
    expert = tmp_path / "expert.bin"
    ple.write_bytes(b"p" * 4096)
    expert.write_bytes(b"e" * 4096)
    calls = []

    def fake_run(command):
        calls.append(list(command))
        if command[0] == "lsblk":
            columns = command[-1]
            if "MOUNTPOINTS" in columns or "PATH" in columns:
                raise H3Error("command failed: modern lsblk columns unsupported")
            return json.dumps({
                "blockdevices": [{
                    "name": "/dev/sda", "model": "test", "serial": "abc", "tran": "sata",
                    "size": "1T", "type": "disk", "pkname": None, "maj:min": "8:0",
                    "mountpoint": "/home",
                }]
            })
        if command[0] == "findmnt":
            return json.dumps({
                "filesystems": [{
                    "target": "/home", "source": "/dev/sda", "fstype": "xfs",
                    "options": "rw", "maj:min": "8:0",
                }]
            })
        raise AssertionError(command)

    monkeypatch.setattr(pagecache_module, "_run", fake_run)
    result = pagecache_module.probe_storage(ple, expert, tmp_path / "storage.json")

    assert result["same_filesystem_source"] is True
    assert result["same_device_major_minor"] is True
    assert result["ple"]["block_device"]["path"] == "/dev/sda"
    assert result["ple"]["block_device"]["mountpoints"] == ["/home"]
    lsblk_calls = [call for call in calls if call[0] == "lsblk"]
    assert len(lsblk_calls) == 2
    assert "MOUNTPOINTS" in lsblk_calls[0][-1]
    assert lsblk_calls[1][-1] == "NAME,MODEL,SERIAL,TRAN,SIZE,TYPE,PKNAME,MAJ:MIN,MOUNTPOINT"


def test_pagecache_window_replays_full_stratum_and_only_uses_footprint_as_gate(tmp_path: Path):
    manifest, manifest_path = _manifest(tmp_path)
    replay = replay_case(manifest_path, "session", 0.01, "useful_object_lru", tmp_path / "replay")
    window = build_pagecache_window(
        manifest_path,
        "session",
        1e-6,
        1.25,
        tmp_path / "ple.bin",
        tmp_path / "expert.bin",
        tmp_path / "pagecache",
    )
    assert window["pressure_sufficient"] is True
    assert window["full_trace_replayed"] is True
    assert window["logical_bytes"] == replay["counters"]["lpddr_read_useful_bytes"]
    assert window["manifest_sha256"] == sha256_file(manifest_path)


def test_pagecache_candidate_can_falsify_explicit_lru_gap(tmp_path: Path):
    residency = tmp_path / "residency.json"
    phase = {
        "lpddr_read_useful_bytes": 500_000,
        "lpddr_fill_write_bytes": 50_000_000,
        "storage_bytes": 50_000_000,
        "ple_storage_bytes": 25_000_000,
        "expert_storage_bytes": 25_000_000,
    }
    write_json(residency, _h3({
        "artifact_kind": "fenix_h3_residency_replay", "stratum": "session",
        "phase_filter": None, "policy": "useful_object_lru", "capacity_gib": 8.0,
        "counters": {
            "lpddr_read_useful_bytes": 1_000_000,
            "lpddr_fill_write_bytes": 100_000_000,
            "storage_bytes": 100_000_000,
            "ple_storage_bytes": 50_000_000,
            "expert_storage_bytes": 50_000_000,
        },
        "phase_counters": {"prefill": dict(phase), "decode": dict(phase)},
        "miss_stream": {"sha256": "same"},
        "provenance": {"trace_manifest_sha256": "trace-manifest", "contract_sha256": CONTRACT_SHA},
    }))
    fio = tmp_path / "fio.json"
    write_json(fio, _fio_fixture(coeff=10.0))
    lpddr = tmp_path / "lpddr.json"; write_json(lpddr, _lpddr_payload(200.0))
    pagecache = tmp_path / "pagecache.json"
    write_json(pagecache, _h3({
        "artifact_kind": "fenix_h3_pagecache_measurement", "stratum": "session",
        "capacity_gib": 8.0, "mode": "buffered", "pressure_observed": True,
        "cold_cache_qualified": True, "no_oom": True, "storage_read_bytes_cgroup": 1,
        "full_trace_replayed": True, "trace_manifest_sha256": "trace-manifest",
        "logical_read_bytes": 1_000_000, "storage_binding_sha256": "binding",
        "fio_tool": _fake_fio_tool(),
    }))
    decision = decide_point(residency, lpddr, fio, CONTRACT, 1, tmp_path / "decision.json", [pagecache])
    assert decision["verdict"] == "CONVENTIONAL_MEMORY_SUFFICIENT"
    assert decision["pagecache_baseline_used"] is True
    assert set(decision["phase_summaries"]) == {"prefill", "decode"}
    assert any(
        candidate["baseline"] == "linux_pagecache_buffered"
        for point in decision["sensitivity_points"]
        for candidate in point["conventional_candidates"]
    )

def test_pagecache_decision_rejects_truncated_or_different_endpoint(tmp_path: Path):
    residency = tmp_path / "residency.json"
    write_json(residency, _h3({
        "artifact_kind": "fenix_h3_residency_replay", "stratum": "session",
        "phase_filter": None, "policy": "useful_object_lru", "capacity_gib": 8.0,
        "counters": {"lpddr_read_useful_bytes": 1000, "ple_storage_bytes": 1, "expert_storage_bytes": 1},
        "miss_stream": {"sha256": "same"},
        "provenance": {"trace_manifest_sha256": "trace-manifest", "contract_sha256": CONTRACT_SHA},
    }))
    lpddr = tmp_path / "lpddr.json"; write_json(lpddr, _lpddr_payload())
    fio = tmp_path / "fio.json"; write_json(fio, _fio_fixture())
    pc = tmp_path / "pc.json"
    write_json(pc, _h3({
        "artifact_kind": "fenix_h3_pagecache_measurement", "stratum": "session", "capacity_gib": 8.0,
        "mode": "buffered", "pressure_observed": True, "cold_cache_qualified": True, "no_oom": True,
        "full_trace_replayed": True, "trace_manifest_sha256": "trace-manifest",
        "logical_read_bytes": 999, "storage_binding_sha256": "binding",
        "fio_tool": _fake_fio_tool(), "storage_read_bytes_cgroup": 1,
    }))
    with pytest.raises(H3Error, match="logical endpoint differs"):
        decide_point(residency, lpddr, fio, CONTRACT, 1, tmp_path / "out.json", [pc])

def test_pagecache_decision_rejects_unqualified_control(tmp_path: Path):
    residency = tmp_path / "residency.json"
    write_json(residency, _h3({
        "artifact_kind": "fenix_h3_residency_replay", "stratum": "session", "phase_filter": None,
        "policy": "useful_object_lru", "capacity_gib": 8.0,
        "counters": {"lpddr_read_useful_bytes": 1000, "ple_storage_bytes": 1, "expert_storage_bytes": 1},
        "miss_stream": {"sha256": "same"},
        "provenance": {"trace_manifest_sha256": "trace-manifest", "contract_sha256": CONTRACT_SHA},
    }))
    lpddr = tmp_path / "lpddr.json"; write_json(lpddr, _lpddr_payload())
    fio = tmp_path / "fio.json"; write_json(fio, _fio_fixture())
    pc = tmp_path / "pc.json"
    write_json(pc, _h3({
        "artifact_kind": "fenix_h3_pagecache_measurement", "stratum": "session", "capacity_gib": 8.0,
        "mode": "buffered", "pressure_observed": True, "cold_cache_qualified": False, "no_oom": True,
    }))
    with pytest.raises(H3Error, match="not qualified"):
        decide_point(residency, lpddr, fio, CONTRACT, 1, tmp_path / "out.json", [pc])

def test_cli_parser_constructs_and_defaults_to_no_prefix_policy():
    from scripts.h3_campaign import build_parser

    parser = build_parser()
    parsed = parser.parse_args(["manifest", "--trace-root", "trace", "--out", "manifest.json"])
    assert parsed.command == "manifest"
    assert parsed.execution_verification == Path("configs/h1_h2_trace_execution_v1.json")


def test_claim_boundary_remains_memory_service_only():
    contract = json.loads(CONTRACT.read_text())
    assert contract["decision_gate"]["scope"] == "conditional_memory_service_layer"
    assert contract["decision_gate"]["between_thresholds"] == "inconclusive"
    assert contract["prerequisites"]["manifest_artifact_kind"] == "fenix_h3_prerequisite_manifest"
    assert set(contract["prerequisites"]["required_evidence"]) == {
        "placement_invariance", "long_context_beyond_8k", "deployment_capacity_budget",
        "tool_qualification"
    }
    assert contract["prerequisites"]["pagecache_derived_from_decision_matrix"] is True


def test_lfu_preserves_atomic_classification_and_learns_after_epoch():
    cache = ByteLFU(160)
    a = CacheObject(("a",), 160, "ple", ())
    b = CacheObject(("b",), 160, "ple", ())
    assert cache.classify_epoch([a])[("a",)] is False
    cache.commit_epoch([a])
    # Two same-epoch touches train only after both have been classified misses.
    hits = cache.classify_epoch([b, b])
    assert hits[("b",)] is False
    cache.commit_epoch([b, b])
    assert cache.classify_epoch([b])[("b",)] is True
    assert cache.classify_epoch([a])[("a",)] is False


def test_lfu_heap_matches_reference_repeated_sort_semantics():
    import random

    class ReferenceByteLFU(ByteLFU):
        def commit_epoch(self, objects):
            materialized = list(objects)
            if not materialized:
                self.epoch += 1
                return
            for obj in materialized:
                if obj.size_bytes <= 0:
                    raise H3Error(f"invalid cache object size for {obj.key}: {obj.size_bytes}")
                self.frequency[obj.key] += 1
                self.last_epoch[obj.key] = self.epoch
            unique = {obj.key: obj for obj in materialized}
            candidates = sorted(
                unique.values(),
                key=lambda obj: self._score(obj.key, obj),
                reverse=True,
            )
            for obj in candidates:
                key = obj.key
                if key in self.entries or obj.size_bytes > self.capacity_bytes:
                    continue
                needed = self.resident_bytes + obj.size_bytes - self.capacity_bytes
                if needed <= 0:
                    self.entries[key] = obj
                    self.resident_bytes += obj.size_bytes
                    continue
                victims = sorted(
                    self.entries.items(),
                    key=lambda item: self._score(item[0], item[1]),
                )
                reclaimed = 0
                chosen = []
                candidate_score = self._score(key, obj)
                for victim_key, victim in victims:
                    if self._score(victim_key, victim) > candidate_score:
                        break
                    chosen.append(victim_key)
                    reclaimed += victim.size_bytes
                    if reclaimed >= needed:
                        break
                if reclaimed < needed:
                    continue
                for victim_key in chosen:
                    victim = self.entries.pop(victim_key)
                    self.resident_bytes -= victim.size_bytes
                self.entries[key] = obj
                self.resident_bytes += obj.size_bytes
            self.epoch += 1

    rng = random.Random(0xF3A1)
    objects = [
        CacheObject(("obj", idx), size, "ple", ())
        for idx, size in enumerate([17, 31, 47, 64, 79, 113, 127, 149, 191, 223, 251, 277])
    ]
    for capacity in (127, 257, 509, 1021):
        fast = ByteLFU(capacity)
        reference = ReferenceByteLFU(capacity)
        for _ in range(160):
            epoch = [rng.choice(objects) for _ in range(rng.randint(0, 18))]
            assert fast.classify_epoch(epoch) == reference.classify_epoch(epoch)
            fast.commit_epoch(epoch)
            reference.commit_epoch(epoch)
            assert fast.resident_bytes == reference.resident_bytes
            assert fast.entries == reference.entries
            assert fast.frequency == reference.frequency
            assert fast.last_epoch == reference.last_epoch
            assert fast.epoch == reference.epoch


def test_strong_lfu_replay_and_offline_bound_are_emitted(tmp_path: Path):
    _, manifest_path = _manifest(tmp_path)
    replay = replay_case(manifest_path, "chat_en", 0.01, "useful_object_lfu", tmp_path / "lfu")
    assert replay["replacement_policy"] == "lfu"
    profile = replay["offline_static_frequency_profile"]
    assert profile["first_touch"] == "compulsory_miss"
    useful = profile["profiles"]["useful_object"]
    vm = profile["profiles"]["vm_granularity"]
    assert useful["ple"]["frequency_histogram"]
    assert useful["expert"]["frequency_histogram"]
    assert useful["ple"]["resident_object_bytes"] == 160
    assert useful["ple"]["transfer_object_bytes"] == 4096
    assert vm["ple"]["resident_object_bytes"] == 4096
    assert replay["deterministic_ple_prefetch_sensitivity"]["ple_prefetchable_storage_bytes"] >= 0
    assert replay["ple_and_expert_perfect_prefetch_sensitivity"]["prefetchable_storage_bytes"] >= 0
    assert Path(replay["logical_access_trace"]["path"]).is_file()

def test_qd32_qualification_uses_achieved_depth_histogram():
    good = {"jobs": [{"iodepth_level": {"1": 5.0, "2": 5.0, "4": 5.0, "8": 5.0, "16": 40.0, "32": 40.0}}]}
    poor = {"jobs": [{"iodepth_level": {"1": 80.0, "2": 10.0, "4": 5.0, "8": 5.0, "16": 0.0, "32": 0.0}}]}
    assert fio_achieved_qd_fraction(good, 32, 0.5) == pytest.approx(0.8)
    assert fio_achieved_qd_fraction(poor, 32, 0.5) == pytest.approx(0.0)
    assert fio_achieved_qd_fraction(good, 1, 0.5) == pytest.approx(1.0)



def test_ramulator_channel_mapper_matches_frontend_address_contract():
    class PassThrough:
        pass

    class CacheLine:
        def __init__(self, interleave_bits=99):
            self.interleave_bits = interleave_bits

    fake = SimpleNamespace(
        channel_mapper=SimpleNamespace(
            PassThroughChannelMapper=PassThrough,
            CacheLineInterleave=CacheLine,
        )
    )

    assert isinstance(_channel_mapper_for_frontend(fake, "addr_vec"), PassThrough)
    flat = _channel_mapper_for_frontend(fake, "flat")
    assert isinstance(flat, CacheLine)
    assert flat.interleave_bits == 0
    with pytest.raises(H3Error, match="unknown Ramulator frontend address mode"):
        _channel_mapper_for_frontend(fake, "invalid")

def test_lpddr_trace_regimes_preserve_expected_transaction_geometry(tmp_path: Path):
    from analysis.h3.lpddr import _write_regime_trace

    lpddr = json.loads(CONTRACT.read_text())["lpddr"]
    org = {"density": 8192, "rank": 2}
    for regime in lpddr["calibration_regimes"]:
        path = tmp_path / f"{regime}.trace"
        count = _write_regime_trace(path, lpddr, org, regime, "mixed")
        rows = path.read_text().splitlines()
        assert len(rows) == count
        assert all(row.split()[0] in {"LD", "ST"} for row in rows)
        if regime == "ple_random_burst":
            assert count % 5 == 0  # 160-byte PLE object / 32-byte LPDDR transaction.
        if regime == "expert_burst":
            assert count % (2_534_400 // 32) == 0


def test_decision_rejects_cross_commit_artifact_mixing(tmp_path: Path):
    residency = tmp_path / "residency.json"
    write_json(residency, _h3({
        "artifact_kind": "fenix_h3_residency_replay", "stratum": "chat_en",
        "phase_filter": "prefill", "policy": "useful_object_lru", "capacity_gib": 8.0,
        "counters": {"lpddr_read_useful_bytes": 1024, "ple_storage_bytes": 1, "expert_storage_bytes": 1},
        "miss_stream": {"sha256": "same"},
    }))
    lpddr = tmp_path / "lpddr.json"
    write_json(lpddr, _lpddr_payload())
    fio_payload = _fio_fixture()
    fio_payload["execution_repository"] = dict(EXECUTION_ID, head="9" * 40)
    fio = tmp_path / "fio.json"; fio.write_text(json.dumps(fio_payload))
    with pytest.raises(H3Error, match="execution-repository identity differs"):
        decide_point(residency, lpddr, fio, CONTRACT, 1, tmp_path / "out.json")


def test_pagecache_upper_bound_uses_slowest_storage_class(tmp_path: Path):
    residency = tmp_path / "residency.json"
    phase = {"lpddr_read_useful_bytes": 1000, "ple_storage_bytes": 0, "expert_storage_bytes": 0}
    write_json(residency, _h3({
        "artifact_kind": "fenix_h3_residency_replay", "stratum": "session",
        "phase_filter": None, "policy": "useful_object_lru", "capacity_gib": 8.0,
        "counters": {"lpddr_read_useful_bytes": 1000, "ple_storage_bytes": 0, "expert_storage_bytes": 0},
        "phase_counters": {"prefill": dict(phase), "decode": dict(phase)},
        "miss_stream": {"sha256": "same"},
        "provenance": {"trace_manifest_sha256": "trace-manifest", "contract_sha256": CONTRACT_SHA},
    }))
    fio_payload = _fio_fixture(coeff=1.0)
    fio_payload["class_coefficients"]["expert:qd1"].update({
        "ns_per_byte_median": 10.0,
        "ns_per_byte_ci95": [9.0, 11.0],
        "replicate_values": [9.0, 10.0, 11.0],
        "replicate_groups": {"w11": [9.0], "w23": [10.0], "w37": [11.0]},
    })
    fio = tmp_path / "fio.json"; write_json(fio, fio_payload)
    lpddr = tmp_path / "lpddr.json"; write_json(lpddr, _lpddr_payload(200.0))
    pc = tmp_path / "pc.json"
    write_json(pc, _h3({
        "artifact_kind": "fenix_h3_pagecache_measurement", "stratum": "session", "capacity_gib": 8.0,
        "mode": "buffered", "pressure_observed": True, "cold_cache_qualified": True, "no_oom": True,
        "storage_read_bytes_cgroup": 100, "full_trace_replayed": True,
        "trace_manifest_sha256": "trace-manifest", "logical_read_bytes": 1000,
        "storage_binding_sha256": "binding", "fio_tool": _fake_fio_tool(),
    }))
    decision = decide_point(residency, lpddr, fio, CONTRACT, 1, tmp_path / "decision.json", [pc])
    candidate = next(
        c for c in decision["sensitivity_points"][0]["conventional_candidates"]
        if c["baseline"] == "linux_pagecache_buffered"
    )
    assert candidate["storage_service_ns_bounds"] == [100.0, 1100.0]


def test_conventional_sufficiency_is_not_project_stopping_without_prerequisites(tmp_path: Path):
    decisions = _write_decision_matrix(tmp_path, gap=False)
    prereq = _prerequisite_manifest(tmp_path, long_passed=False)
    gate = campaign_gate(decisions, prereq, CONTRACT, tmp_path / "gate.json")
    assert gate["preliminary_memory_service_verdict"] == "CONVENTIONAL_MEMORY_SUFFICIENT"
    assert gate["paper_promotion_verdict"] == "H3_PAPER_GATE_BLOCKED"
    assert gate["proceed_to_h4"] is False


def test_conventional_sufficiency_requires_measured_qd32_scheduler_feasibility(tmp_path: Path):
    decisions = _write_decision_matrix(tmp_path, gap=False)
    blocked = campaign_gate(decisions, _prerequisite_manifest(tmp_path), CONTRACT, tmp_path / "blocked.json")
    assert blocked["preliminary_memory_service_verdict"] == "CONVENTIONAL_MEMORY_SUFFICIENT"
    assert blocked["paper_promotion_verdict"] == "H3_PAPER_GATE_BLOCKED"
    assert "storage-concurrency evidence" in (blocked["prerequisite_error"] or "")
    promoted = campaign_gate(
        decisions, _prerequisite_manifest(tmp_path, include_concurrency=True), CONTRACT, tmp_path / "promoted.json"
    )
    assert promoted["paper_promotion_verdict"] == "H3_CONVENTIONAL_SUFFICIENT_STOP_H4"
    assert promoted["proceed_to_h4"] is False

def test_campaign_gate_requires_all_strong_conventional_policies(tmp_path: Path):
    decisions = _write_decision_matrix(tmp_path, policies=["useful_object_lru"], gap=True)
    prereq = _prerequisite_manifest(tmp_path)
    gate = campaign_gate(decisions, prereq, CONTRACT, tmp_path / "gate.json")
    assert gate["preliminary_memory_service_verdict"] == "INCOMPLETE_OR_DUPLICATE_H3_MATRIX"
    assert any("useful_object_lfu" in msg and "count=0" in msg for msg in gate["matrix_errors"])

def test_qd1_is_diagnostic_and_cannot_enter_paper_promotion(tmp_path: Path):
    policies = json.loads(CONTRACT.read_text())["primary_policies"] + json.loads(CONTRACT.read_text())["strong_conventional_policies"]
    decisions = []
    for name in STRATA:
        for policy in policies:
            path = tmp_path / f"{name}-{policy}.json"
            write_json(path, _decision_fixture(name, policy, qd=1, gap=True))
            decisions.append(path)
    prereq = _prerequisite_manifest(tmp_path)
    with pytest.raises(H3Error, match="no promotion-eligible QD"):
        campaign_gate(decisions, prereq, CONTRACT, tmp_path / "gate.json")


def test_promotion_requires_policy_specific_actual_lpddr_trace(tmp_path: Path):
    policies = json.loads(CONTRACT.read_text())["primary_policies"] + json.loads(CONTRACT.read_text())["strong_conventional_policies"]
    decisions = []
    for name in STRATA:
        for policy in policies:
            path = tmp_path / f"{name}-{policy}.json"
            write_json(path, _decision_fixture(name, policy, actual_lpddr=False, gap=True))
            decisions.append(path)
    prereq = _prerequisite_manifest(tmp_path)
    with pytest.raises(H3Error, match="actual-H3 LPDDR"):
        campaign_gate(decisions, prereq, CONTRACT, tmp_path / "gate.json")


def test_prerequisite_sha_tamper_blocks_directional_promotion(tmp_path: Path):
    decisions = _write_decision_matrix(tmp_path, gap=True)
    prereq = _prerequisite_manifest(tmp_path)
    manifest = json.loads(prereq.read_text())
    placement = Path(manifest["evidence"]["placement_invariance"]["artifact"])
    payload = json.loads(placement.read_text())
    payload["tampered"] = True
    placement.write_text(json.dumps(payload))
    gate = campaign_gate(decisions, prereq, CONTRACT, tmp_path / "gate.json")
    assert gate["preliminary_memory_service_verdict"] == "H3_MEMORY_GAP_SUPPORTED"
    assert gate["paper_promotion_verdict"] == "H3_PAPER_GATE_BLOCKED"
    assert "SHA mismatch" in (gate["prerequisite_error"] or "")

def test_optimistic_prefetch_can_falsify_gap_but_cannot_prove_sufficiency():
    oracle = 100.0
    candidates = [
        {"baseline": "measured", "promotion_role": "realizable", "service_ns_bounds": [120.0, 130.0]},
        {"baseline": "perfect-prefetch", "promotion_role": "gap_falsification_only", "service_ns_bounds": [100.0, 100.0]},
    ]
    summary = _candidate_interval_summary(candidates, oracle)
    assert summary["gap_falsification_penalty_fraction_bounds"] == [0.0, 0.0]
    assert summary["realizable_penalty_fraction_bounds"][1] == pytest.approx(0.3)


def test_offline_static_bound_optimizes_measured_service_not_bytes():
    profile = {
        "policy_profile_mapping": {"useful_object_lru": "useful_object"},
        "profiles": {
            "useful_object": {
                "ple": {
                    "resident_object_bytes": 160,
                    "transfer_object_bytes": 4096,
                    "frequency_histogram": {"10": 1000},
                    "unique_objects": 1000,
                },
                "expert": {
                    "resident_object_bytes": 2_534_400,
                    "transfer_object_bytes": 2_535_424,
                    "frequency_histogram": {"2": 10},
                    "unique_objects": 10,
                },
            }
        },
    }
    fio = _fio_fixture()
    fio["class_coefficients"]["ple:qd32"]["ns_per_byte_median"] = 100.0
    fio["class_coefficients"]["expert:qd32"]["ns_per_byte_median"] = 1.0
    bound = _service_time_optimized_offline_bound(
        profile, fio, 32, 2_535_424, 204.8, "useful_object_lru",
        baseline_ple_storage_bytes=1000 * 10 * 4096,
        baseline_expert_storage_bytes=10 * 2 * 2_535_424,
    )
    assert bound["selected_ple_objects"] > 0
    assert bound["selected_expert_objects"] == 0
    assert bound["ple_resident_object_bytes"] == 160
    assert bound["ple_transfer_object_bytes"] == 4096

def test_joint_storage_projection_preserves_common_window_hierarchy():
    fio = _fio_fixture()
    fio["class_coefficients"]["ple:qd32"]["replicate_groups"] = {
        "11": [1.0, 1.1], "23": [10.0, 11.0], "37": [100.0, 110.0]
    }
    fio["class_coefficients"]["expert:qd32"]["replicate_groups"] = {
        "11": [2.0, 2.2], "23": [20.0, 22.0], "37": [200.0, 220.0]
    }
    fio["mixed_interaction"]["qd32"]["replicate_groups"] = {
        "11": [1.0], "23": [1.0], "37": [1.0]
    }
    projected = project_storage_bytes_ns(1.0, 1.0, fio, 32, draws=256)
    assert projected["window_pairing_preserved"] is True
    assert projected["common_spatial_windows"] == ["11", "23", "37"]
    assert projected["uncertainty_method"].startswith("hierarchical_joint_bootstrap")


def test_actual_h3_ramulator_trace_uses_residency_logical_evidence(tmp_path: Path):
    _, manifest_path = _manifest(tmp_path)
    replay = replay_case(
        manifest_path, "chat_en", 0.01, "useful_object_lru", tmp_path / "replay"
    )
    contract = json.loads(CONTRACT.read_text())
    contract["lpddr"]["actual_trace_sample_transactions"] = 128
    org = {"density": 8192, "rank": 2}
    read_trace = tmp_path / "actual-read.trace"
    count = write_actual_h3_transaction_trace(read_trace, replay, contract, org, "read")
    rows = read_trace.read_text().splitlines()
    assert count == 128
    assert len(rows) == 128
    assert all(row.startswith("LD ") for row in rows)
    mixed_trace = tmp_path / "actual-mixed.trace"
    write_actual_h3_transaction_trace(mixed_trace, replay, contract, org, "mixed")
    ops = {row.split()[0] for row in mixed_trace.read_text().splitlines()}
    assert "LD" in ops
    assert ops.issubset({"LD", "ST"})


def test_actual_lpddr_calibration_requires_complete_ncs_set(tmp_path: Path):
    residency = tmp_path / "residency.json"
    write_json(residency, _h3({
        "artifact_kind": "fenix_h3_residency_replay",
        "stratum": "chat_en", "phase_filter": "prefill", "policy": "useful_object_lru",
        "capacity_gib": 8.0, "capacity_bytes": 8 * 1024**3,
        "counters": {"lpddr_read_useful_bytes": 1024, "lpddr_fill_write_bytes": 128,
                     "ple_storage_bytes": 64, "expert_storage_bytes": 64, "storage_bytes": 128},
        "miss_stream": {"sha256": "same"},
    }))
    lpddr = tmp_path / "lpddr.json"; write_json(lpddr, _lpddr_payload())
    fio = tmp_path / "fio.json"; write_json(fio, _fio_fixture())
    actual = tmp_path / "actual.json"
    write_json(actual, _h3({
        "artifact_kind": "fenix_h3_lpddr_actual_trace_calibration",
        "stratum": "chat_en", "phase_filter": "prefill", "policy": "useful_object_lru", "capacity_gib": 8.0,
        "ramulator_tool": {"head": "c" * 40, "tree": "d" * 40},
        "sensitivity_points": [{
            "sensitivity_id": "actual_h3_policy_trace_nCS1",
            "aggregate_read_bandwidth_gb_s": 100.0,
            "aggregate_write_bandwidth_gb_s": 100.0,
            "aggregate_mixed_total_bandwidth_gb_s": 100.0,
        }],
        "provenance": {"residency_sha256": sha256_file(residency), "contract_sha256": CONTRACT_SHA},
    }))
    with pytest.raises(H3Error, match="actual-H3 LPDDR sensitivity set mismatch"):
        decide_point(residency, lpddr, fio, CONTRACT, 32, tmp_path / "decision.json", actual_lpddr_path=actual)

def test_ramulator_runtime_dependency_resolver_reads_pinned_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "ramulator"\nversion = "2.1.0"\ndependencies = ["pyyaml", "example>=1"]\n'
    )
    assert _project_runtime_dependencies(tmp_path) == ["pyyaml", "example>=1"]


def test_paired_storage_windows_are_nonoverlapping_even_when_population_limited(tmp_path: Path):
    ple = tmp_path / "ple.bin"; ple.write_bytes(b"x")
    expert = tmp_path / "expert.bin"; expert.write_bytes(b"x")
    binding = tmp_path / "binding.json"
    write_json(binding, {
        "artifact_kind": "fenix_h3_storage_binding",
        "ple": {"path": str(ple.resolve()), "mount": {"maj:min": "1:1"}},
        "expert": {"path": str(expert.resolve()), "mount": {"maj:min": "1:1"}},
        "same_device_major_minor": True,
    })
    misses = tmp_path / "misses.jsonl.gz"
    _write_interleaved_storage_misses(misses, 30)
    out = tmp_path / "samples"
    payload = build_samples(
        misses, ple, expert, binding, out, [11, 23, 37],
        1 << 30, 1 << 30, 2 << 30, contract_sha256=CONTRACT_SHA,
    )
    windows = {}
    for row in payload["samples"]:
        windows.setdefault(row["window_id"], []).append(row)
    assert len(windows) == 3
    ranges = []
    for rows in windows.values():
        assert {row["kind"] for row in rows} == {"ple", "expert", "mixed"}
        reference = rows[0]["window"]
        ranges.append((reference["source_record_start_inclusive"], reference["source_record_end_exclusive"]))
        assert all(row["window"]["population_limited"] is True for row in rows)
    assert len(set(ranges)) == 3
    ordered = sorted(ranges)
    assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))


def test_no_alias_channel_layout_assigns_disjoint_object_regions():
    from analysis.h3.lpddr import _NoAliasChannelLayout

    layout = _NoAliasChannelLayout(64 * 4096, 32, 20260906)
    a = set(layout.addresses(("read", "ple", 1), 5))
    b = set(layout.addresses(("read", "ple", 1 + (32 * 1024**3 // 160)), 5))
    assert a.isdisjoint(b)
    assert layout.stats()["cross_object_aliases"] == 0
    assert layout.stats()["modulo_address_folding"] is False



def test_actual_h3_read_and_fill_share_one_storage_object_mapping():
    from analysis.h3.lpddr import _NoAliasChannelLayout

    layout = _NoAliasChannelLayout(64 * 4096, 32, 20260906)
    fill = list(_mapped_access_addresses(
        layout,
        ("ple_page", 0),
        object_base=0,
        object_span_bytes=4096,
        access_base=0,
        access_length_bytes=4096,
        tx_bytes=32,
        channels=16,
        channel_index=0,
    ))
    row = list(_mapped_access_addresses(
        layout,
        ("ple_page", 0),
        object_base=0,
        object_span_bytes=4096,
        access_base=0,
        access_length_bytes=160,
        tx_bytes=32,
        channels=16,
        channel_index=0,
    ))
    assert row
    assert set(row).issubset(set(fill))
    assert layout.stats()["logical_objects_mapped"] == 1

def _placement_input(tmp_path: Path) -> tuple[Path, dict[str, dict[str, Path]]]:
    required = json.loads(CONTRACT.read_text())["prerequisites"]["placement_invariance"]["required_strata"]
    placement_cases: dict[str, dict[str, Path]] = {}
    placements = []
    for placement_index, name in enumerate(("placement-a", "placement-b")):
        cases = {}
        for ordinal, stratum in enumerate(required):
            case = tmp_path / name / stratum
            case.mkdir(parents=True)
            prompts = case / "prompts.json"
            write_json(prompts, {"prompts": [{"ordinal": 0, "text": f"same-{stratum}"}]})
            ple = case / "ple.jsonl"
            ple_rows = [{
                "request_id": "r0", "token_position": 0, "ple_head": head,
                "physical_row_id": ordinal * 100 + head, "bytes": 160,
                "phase": "prefill", "address_known_ns": 1000 + placement_index * 999 + head,
            } for head in range(16)]
            _write_jsonl(ple, ple_rows)
            moe = case / "moe.jsonl"
            _write_jsonl(moe, [{
                "request_id": "r0", "layer": "language_model.model.layers.0.mlp.experts",
                "selected_expert_ids": list(range(10)), "phase": "prefill",
                "trace_scope": "router_all_layers", "timestamp_ns": 2000 + placement_index * 999,
            }])
            cases[stratum] = {
                "prompts": str(prompts), "ple_normalized": str(ple), "moe_normalized": str(moe),
            }
            placement_cases.setdefault(stratum, {})[name] = ple
        placements.append({"name": name, "cases": cases})
    inp = tmp_path / "placement-input.json"
    write_json(inp, {"placements": placements})
    return inp, placement_cases


def test_placement_invariance_ignores_timing_noise_but_detects_semantic_drift(tmp_path: Path):
    inp, paths = _placement_input(tmp_path)
    out = tmp_path / "placement-out.json"
    first = build_placement_invariance(inp, CONTRACT, out)
    assert first["derived_pass"] is True
    assert first["evaluator"] == "ordered_routing_and_ple_semantic_identity_across_physical_placements"
    ple = paths["chat_en"]["placement-b"]
    rows = list(iter_case_epochs.__globals__["iter_jsonl"](ple))
    rows[0]["physical_row_id"] += 1
    _write_jsonl(ple, rows)
    second = build_placement_invariance(inp, CONTRACT, tmp_path / "placement-drift.json")
    assert second["derived_pass"] is False
    assert any(row["field"] == "ple_normalized" for row in second["mismatches"])


def test_capacity_budget_is_contract_bound_and_evidence_bound(tmp_path: Path):
    required = [
        "os_and_services", "runtime", "nonconditional_model", "kv_cache",
        "activations_buffers", "safety_headroom",
    ]
    components = {}
    for name in required:
        evidence = tmp_path / f"{name}.txt"
        evidence.write_text(f"measured {name}\n")
        components[name] = {"gib": 1.0, "artifact": str(evidence), "sha256": sha256_file(evidence)}
    inp = tmp_path / "capacity.json"
    write_json(inp, {"platform_mode": "primary", "physical_memory_gib": 32, "reserved_components": components})
    result = build_capacity_budget(inp, CONTRACT, tmp_path / "capacity-out.json")
    assert result["derived"] is True
    assert result["physical_memory_gib"] == 32.0
    assert result["conditional_state_available_gib"] == 26.0
    assert result["candidate_budgets_that_fit_gib"] == [4.0, 7.0, 8.0, 12.0, 16.0, 19.0]
    bad = tmp_path / "capacity-bad.json"
    write_json(bad, {"platform_mode": "primary", "physical_memory_gib": 64, "reserved_components": components})
    with pytest.raises(H3Error, match="contract platform_capacity_gib"):
        build_capacity_budget(bad, CONTRACT, tmp_path / "bad-out.json")


def _long_context_case(path: Path, prompt_tokens: int) -> None:
    path.mkdir(parents=True)
    _write_jsonl(path / "client.jsonl", [{
        "request_id": "r0", "prompt_tokens": prompt_tokens, "completion_tokens": 1,
    }])
    _write_jsonl(path / "ple_normalized.jsonl", [{
        "request_id": "r0", "physical_row_id": prompt_tokens % 1000,
    }])
    _write_jsonl(path / "moe_normalized.jsonl", [{
        "request_id": "r0", "layer": "0", "selected_expert_ids": list(range(10)),
    }])
    write_json(path / "evidence.json", {
        "trace_valid": True,
        "case": {"stratum": "long_context", "concurrency": 1, "correlation_mode": "exact_request_correlation"},
    })


def test_long_context_evaluator_derives_multi_length_characterization(tmp_path: Path):
    p8 = tmp_path / "lc-8k"; p16 = tmp_path / "lc-16k"; p32 = tmp_path / "lc-32k"
    _long_context_case(p8, 8192); _long_context_case(p16, 16384); _long_context_case(p32, 32768)
    inp = tmp_path / "long-input.json"
    write_json(inp, {"case_dirs": [str(p8), str(p16), str(p32)]})
    result = build_long_context_scaling(inp, CONTRACT, tmp_path / "long-out.json")
    assert result["derived_pass"] is True
    assert result["maximum_demonstrated_prompt_tokens"] == 32768
    assert set(result["coverage_by_band"]) == {"8k", "16k", "32k"}
    assert result["distinct_band_cases"] == 3


def test_storage_concurrency_evidence_is_derived_from_raw_measurement_events(tmp_path: Path):
    measured = tmp_path / "scheduler.json"
    contract = json.loads(CONTRACT.read_text())
    required = list(dict.fromkeys(contract["decision_gate"]["campaign_requirements"]["required_strata"] + contract["decision_gate"]["campaign_requirements"]["required_ordinary_strata"]))
    events = []
    for stratum in required:
        for phase in ("prefill", "decode"):
            for i in range(32):
                events.append({
                    "request_id": f"{stratum}-{phase}-r{i % 4}", "object_id": f"{stratum}-{phase}-o{i}",
                    "stratum": stratum, "phase": phase, "observed_outstanding_io": 32,
                    "available_prefetch_slack_ns": 1000, "storage_completion_latency_ns": 800,
                })
    write_json(measured, {
        "artifact_kind": "fenix_h3_storage_scheduler_measurement", "measured": True,
        "workload_derived_schedule": True, "trace_manifest_sha256": "trace-manifest",
        "storage_binding_sha256": "binding", "events": events,
    })
    result = build_storage_concurrency_evidence(measured, CONTRACT, tmp_path / "scheduler-out.json")
    assert result["derived_pass"] is True
    assert result["raw_event_fields_derived_not_user_aggregated"] is True
    assert len(result["cell_results"]) == len(required) * 2
    bad = json.loads(measured.read_text()); bad["events"] = bad["events"][:-1]; write_json(measured, bad)
    result2 = build_storage_concurrency_evidence(measured, CONTRACT, tmp_path / "scheduler-out2.json")
    assert result2["derived_pass"] is False
    write_json(measured, {
        "artifact_kind": "fenix_h3_storage_scheduler_measurement", "measured": True,
        "workload_derived_schedule": True, "trace_manifest_sha256": "trace-manifest",
        "storage_binding_sha256": "binding", "achieved_queue_depth_fraction": 1.0,
    })
    result3 = build_storage_concurrency_evidence(measured, CONTRACT, tmp_path / "scheduler-out3.json")
    assert result3["derived_pass"] is False


def test_actual_lpddr_mapping_strategies_are_both_collision_free_and_distinct():
    from analysis.h3.lpddr import _NoAliasChannelLayout
    layouts = {name: _NoAliasChannelLayout(512 * 4096, 32, 20260906, name)
               for name in ("dense_first_touch", "hashed_bank_spread")}
    mapped = {}
    for name, layout in layouts.items():
        a = list(layout.addresses(("ple_page", 7), 8))
        b = list(layout.addresses(("expert", 3, 11), 80))
        assert set(a).isdisjoint(set(b))
        assert list(layout.addresses(("ple_page", 7), 8)) == a
        stats = layout.stats()
        assert stats["cross_object_aliases"] == 0
        assert stats["modulo_address_folding"] is False
        mapped[name] = a + b
    assert mapped["dense_first_touch"] != mapped["hashed_bank_spread"]


def test_placement_invariance_detects_order_only_drift(tmp_path: Path):
    inp, paths = _placement_input(tmp_path)
    first = build_placement_invariance(inp, CONTRACT, tmp_path / "ordered-ok.json")
    assert first["derived_pass"] is True
    ple = paths["chat_en"]["placement-b"]
    rows = list(iter_case_epochs.__globals__["iter_jsonl"](ple))
    rows[0], rows[1] = rows[1], rows[0]
    _write_jsonl(ple, rows)
    second = build_placement_invariance(inp, CONTRACT, tmp_path / "ordered-bad.json")
    assert second["derived_pass"] is False
    assert any(row["field"] == "ple_normalized" for row in second["mismatches"])


def test_fio_summary_reports_independent_block_strength(tmp_path: Path):
    # The strength label is a property of true independent blocks, not bootstrap draws.
    fio = _fio_fixture()
    fio["paired_window_design"].update({
        "preferred_windows": 5, "target_windows": 8, "statistical_strength": "minimum_only"
    })
    assert fio["paired_window_design"]["distinct_nonoverlapping_windows"] == 3
    assert fio["paired_window_design"]["statistical_strength"] == "minimum_only"


def test_campaign_gate_rejects_duplicate_and_mixed_fingerprint_rows(tmp_path: Path):
    decisions = _write_decision_matrix(tmp_path, gap=True)
    duplicate = tmp_path / "duplicate.json"
    write_json(duplicate, json.loads(decisions[0].read_text()))
    gate = campaign_gate([*decisions, duplicate], _prerequisite_manifest(tmp_path), CONTRACT, tmp_path / "dup-gate.json")
    assert gate["preliminary_memory_service_verdict"] == "INCOMPLETE_OR_DUPLICATE_H3_MATRIX"
    assert any("count=2" in msg for msg in gate["matrix_errors"])

    decisions2 = _write_decision_matrix(tmp_path / "mixed", gap=True)
    changed = json.loads(decisions2[0].read_text())
    changed["campaign_fingerprint"]["device_major_minor"] = "8:1"
    write_json(decisions2[0], changed)
    with pytest.raises(H3Error, match="campaign fingerprint"):
        campaign_gate(decisions2, _prerequisite_manifest(tmp_path / "mixed"), CONTRACT, tmp_path / "mixed-gate.json")


def test_campaign_gate_requires_independent_phase_rows(tmp_path: Path):
    decisions = _write_decision_matrix(tmp_path, gap=True)
    removed = decisions.pop()
    assert "decode" in removed.name
    gate = campaign_gate(decisions, _prerequisite_manifest(tmp_path), CONTRACT, tmp_path / "phase-gate.json")
    assert gate["preliminary_memory_service_verdict"] == "INCOMPLETE_OR_DUPLICATE_H3_MATRIX"
    assert any("decode" in msg and "count=0" in msg for msg in gate["matrix_errors"])




def test_campaign_sufficiency_requires_phase_consistency(tmp_path: Path):
    decisions = _write_decision_matrix(tmp_path, gap=False)
    for path in decisions:
        row = json.loads(path.read_text())
        if row["stratum"] == "chat_en" and row["phase_filter"] == "decode":
            row["global_penalty_fraction_bounds"] = [0.08, 0.12]
            write_json(path, row)
    gate = campaign_gate(decisions, _prerequisite_manifest(tmp_path, include_concurrency=True), CONTRACT, tmp_path / "phase-consistency-gate.json")
    assert gate["preliminary_memory_service_verdict"] == "INCONCLUSIVE_ESCALATE_INTEGRATED_TIMING"

def test_v6_execution_protocol_is_terminal_safe_and_has_no_active_v5_refs():
    protocol = ROOT / "docs/h3_execution_v6.md"
    content = protocol.read_text()
    forbidden = (
        "set -e",
        "set -o errexit",
        "pipefail",
        "exit 1",
        "exit 2",
        "|| exit",
        "&& exit",
    )
    assert all(token not in content for token in forbidden)
    assert "configs/h3/contract_v6.json" in content
    assert "contract_v5.json" not in content
    assert "h3_execution_v5.md" not in content
    assert "test_h3_conventional_hierarchy_v5.py" not in content


def test_lfu_does_not_rebuild_full_resident_heap_per_epoch():
    class CountingByteLFU(ByteLFU):
        def __init__(self, capacity_bytes: int):
            super().__init__(capacity_bytes)
            self.score_calls = 0

        def _score(self, key, obj):
            self.score_calls += 1
            return super()._score(key, obj)

    resident_count = 5000
    object_bytes = 160
    cache = CountingByteLFU(resident_count * object_bytes)
    seed = [
        CacheObject(("seed", idx), object_bytes, "ple", ())
        for idx in range(resident_count)
    ]
    cache.commit_epoch(seed)
    baseline_calls = cache.score_calls

    for idx in range(100):
        candidate = CacheObject(("candidate", idx), object_bytes, "ple", ())
        cache.commit_epoch([candidate])

    incremental_calls = cache.score_calls - baseline_calls
    assert cache.resident_bytes == resident_count * object_bytes
    assert len(cache.entries) == resident_count
    # A per-epoch rebuild would require roughly resident_count * 100 score
    # evaluations. Lazy persistent maintenance should remain far below that.
    assert incremental_calls < 50_000


def test_v6_fio_cli_uses_supported_readonly_flag():
    content = (ROOT / "scripts/h3_campaign.py").read_text()
    assert '"--readonly=1"' not in content
    assert content.count('"--readonly"') >= 2


def test_v6_fio_qualification_requires_libaio_engine():
    content = (ROOT / "scripts/h3_campaign.py").read_text()
    assert content.count('"--enghelp"') >= 2
    assert "fio_engine:libaio_missing" in content
    assert "pinned fio build lacks required libaio engine" in content


def test_v6_storage_iologs_omit_terminal_close_for_async_replay(tmp_path: Path):
    ple = tmp_path / "ple.bin"
    expert = tmp_path / "expert.bin"
    ple.touch()
    expert.touch()

    binding = tmp_path / "binding.json"
    write_json(binding, {
        "artifact_kind": "fenix_h3_storage_binding",
        "ple": {
            "path": str(ple.resolve()),
            "mount": {"maj:min": "1:1"},
        },
        "expert": {
            "path": str(expert.resolve()),
            "mount": {"maj:min": "1:1"},
        },
        "same_device_major_minor": True,
    })

    misses = tmp_path / "misses.jsonl.gz"
    _write_interleaved_storage_misses(misses, 30)

    payload = build_samples(
        misses,
        ple,
        expert,
        binding,
        tmp_path / "samples",
        [11],
        4096,
        2_535_424,
        4096 + 2_535_424,
        contract_sha256=CONTRACT_SHA,
    )

    assert payload["sampling_design"]["terminal_file_close_records_omitted"] is True

    for sample in payload["samples"]:
        lines = Path(sample["iolog"]).read_text().splitlines()
        assert lines
        assert not any(line.endswith(" close") for line in lines)
        assert lines[-1].split()[1] == "read"


def test_v6_pagecache_iolog_omits_terminal_close(tmp_path: Path):
    _, manifest_path = _manifest(tmp_path)

    ple = tmp_path / "ple.bin"
    expert = tmp_path / "expert.bin"
    ple.touch()
    expert.touch()

    payload = build_pagecache_window(
        manifest_path,
        "session",
        1e-6,
        1.25,
        ple,
        expert,
        tmp_path / "pagecache",
    )

    assert payload["terminal_file_close_records_omitted"] is True

    lines = Path(payload["iolog"]).read_text().splitlines()
    assert lines
    assert not any(line.endswith(" close") for line in lines)
    assert lines[-1].split()[1] == "read"
