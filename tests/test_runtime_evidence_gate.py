import json
from pathlib import Path

import qualification.runtime_lane as runtime_lane
from qualification.runtime_lane import inspect_runtime_evidence


def _config() -> dict:
    return {
        "lane_id": "lane",
        "campaign_base_commit": "base",
        "runtime": {
            "revision": "runtime-rev",
            "checkout": "external/runtime/qwen38",
        },
        "model": {"revision": "model-rev"},
        "execution": {
            "runtime_image": "fenix-qwen38:locked",
            "runtime_evidence_report": (
                "results/qualification/runtime_first_boot.json"
            ),
            "runtime_evidence_tested_commit": "tested-commit",
            "runtime_evidence_profile": {
                "tensor_parallel_size": 1,
                "distributed_executor_backend": "mp",
                "cpu_offload_gb": 40.0,
                "hot_experts": 0,
                "max_model_len": 8192,
                "max_num_seqs": 1,
                "max_num_batched_tokens": 2048,
                "kv_cache_memory_bytes": 1073741824,
                "ple_cpu_offload": True,
                "moe_backend": "humming",
                "offload_backend": "uva",
                "mtp_enabled": False,
                "fenix_trace_enabled": False,
            },
            "tp1_distributed_executor_backend": "mp",
            "tp1_reason": "test",
        },
        "known_upstream_state": {},
    }


def _certificate() -> dict:
    return {
        "schema_version": 1,
        "qualification_kind": "real_model_boot_and_generation",
        "lane_id": "lane",
        "tested_repository_commit": "tested-commit",
        "runtime_revision": "runtime-rev",
        "model_revision": "model-rev",
        "runtime_image": "fenix-qwen38:locked",
        "target": {
            "gpu": "NVIDIA RTX A6000",
            "tensor_parallel_size": 1,
            "distributed_executor_backend": "mp",
        },
        "boot_profile": {
            "cpu_offload_gb": 40.0,
            "hot_experts": 0,
            "max_model_len": 8192,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 2048,
            "kv_cache_memory_bytes": 1073741824,
            "ple_cpu_offload": True,
            "moe_backend": "humming",
            "offload_backend": "uva",
            "mtp_enabled": False,
            "fenix_trace_enabled": False,
        },
        "checks": {
            "server_ready": True,
            "generation_response_valid": True,
        },
        "runtime_qualified": True,
        "performance_qualified": False,
        "semantic_smoke_qualified": False,
        "trace_qualified": False,
    }


def _write(root: Path, certificate: dict) -> None:
    path = root / "results/qualification/runtime_first_boot.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(certificate))


def test_runtime_evidence_accepts_matching_certificate(tmp_path: Path):
    _write(tmp_path, _certificate())

    result = inspect_runtime_evidence(tmp_path, _config())

    assert result["passed"] is True
    assert result["failures"] == []


def test_runtime_evidence_rejects_model_revision_drift(tmp_path: Path):
    certificate = _certificate()
    certificate["model_revision"] = "wrong"
    _write(tmp_path, certificate)

    result = inspect_runtime_evidence(tmp_path, _config())

    assert result["passed"] is False
    assert "runtime_evidence_model_revision_mismatch" in result["failures"]


def test_runtime_evidence_rejects_failed_measured_check(tmp_path: Path):
    certificate = _certificate()
    certificate["checks"]["generation_response_valid"] = False
    _write(tmp_path, certificate)

    result = inspect_runtime_evidence(tmp_path, _config())

    assert result["passed"] is False
    assert "runtime_evidence_checks_not_all_passed" in result["failures"]


def test_qualify_promotes_only_with_measured_evidence(
    tmp_path: Path,
    monkeypatch,
):
    config = _config()

    monkeypatch.setattr(
        runtime_lane,
        "inspect_runtime_source",
        lambda *args, **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        runtime_lane,
        "inspect_host_environment",
        lambda *args, **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        runtime_lane,
        "inspect_image_smoke_report",
        lambda *args, **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        runtime_lane,
        "inspect_runtime_evidence",
        lambda *args, **kwargs: {
            "passed": True,
            "failures": [],
            "report": {
                "semantic_smoke_qualified": False,
            },
        },
    )

    result = runtime_lane.qualify(tmp_path, config)

    assert result["status"] == runtime_lane.RUNTIME_QUALIFIED
    assert result["runtime_qualified"] is True
    assert result["performance_qualified"] is False
    assert result["semantic_smoke_qualified"] is False
    assert result["trace_qualified"] is False
