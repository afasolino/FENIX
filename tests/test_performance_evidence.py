import argparse
import json
from pathlib import Path

import pytest

from scripts import performance_evidence, workload_contract


def _record(ordinal: int, phase: str) -> dict:
    return {
        "request_id": f"r{ordinal}",
        "ordinal": ordinal,
        "concurrency": 1,
        "phase": phase,
        "prompt_tokens": 32,
        "completion_tokens": 4,
        "ttft_ms": 10.0,
        "tpot_ms": 2.0,
        "e2e_ms": 16.0,
        "decode_tokens_s": 500.0,
    }


def _runtime_lane(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "lane_id": "lane",
                "runtime": {
                    "repository": "runtime-repo",
                    "revision": "runtime-rev",
                    "container_image": "base@sha256:deadbeef",
                },
                "model": {
                    "repository": "model-repo",
                    "revision": "model-rev",
                },
            }
        )
    )


def _campaign(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiments": {
                    "runtime_qualification": {
                        "workload_profile": "runtime_qualification_v1",
                        "warmup_requests": 3,
                        "measured_requests": 2,
                        "input_tokens": [32],
                        "output_tokens": 4,
                        "concurrency": [1],
                        "temperature": 0.0,
                        "repetitions": 3,
                    }
                }
            }
        )
    )


def _server_log(path: Path, trace: str = "0") -> None:
    path.write_text(
        json.dumps({"FENIX_TRACE": trace})
        + "\n"
        + "python -m scripts.fenix_podman run "
        + "fenix-qwen38:candidate-commit serve /model\n"
        + "Application startup complete.\n"
    )


def _config(tmp_path: Path) -> performance_evidence.MeasurementConfig:
    server_log = tmp_path / "server.log"
    _server_log(server_log)
    runtime_lane = tmp_path / "runtime_lane.json"
    _runtime_lane(runtime_lane)
    campaign = tmp_path / "campaign.json"
    _campaign(campaign)
    return performance_evidence.MeasurementConfig(
        server_log=server_log,
        output=tmp_path / "measured.jsonl",
        url="http://example.invalid/v1/chat/completions",
        model="model",
        log_settle_ms=0,
        runtime_lane=runtime_lane,
        campaign=campaign,
        experiment="runtime_qualification",
        repetition_index=1,
        tokenize_url="http://example.invalid/tokenize",
    )


def _prepared() -> workload_contract.PreparedWorkload:
    return workload_contract.PreparedWorkload(
        prompt="prompt",
        prompt_tokens=32,
        max_model_len=8192,
        tokenize_url="http://example.invalid/tokenize",
        workload_profile="runtime_qualification_v1",
    )


def test_launch_log_requires_started_non_trace_server(tmp_path: Path):
    path = tmp_path / "server.log"
    _server_log(path, trace="0")

    metadata = performance_evidence.require_performance_server(path)

    assert metadata.startup_complete is True
    assert metadata.trace_enabled is False
    assert metadata.trace_values == ("0",)
    assert metadata.runtime_images == (
        "fenix-qwen38:candidate-commit",
    )


@pytest.mark.parametrize("trace", ("1", None))
def test_trace_or_unknown_launch_is_rejected(tmp_path: Path, trace):
    path = tmp_path / "server.log"
    if trace is None:
        path.write_text(
            "fenix-qwen38:candidate-commit\n"
            "Application startup complete.\n"
        )
    else:
        _server_log(path, trace=trace)

    with pytest.raises(
        performance_evidence.PerformanceEvidenceError,
        match="FENIX_TRACE=0",
    ):
        performance_evidence.require_performance_server(path)


def test_ambiguous_runtime_image_is_rejected(tmp_path: Path):
    path = tmp_path / "server.log"
    path.write_text(
        '{"FENIX_TRACE": "0"}\n'
        "fenix-qwen38:candidate-a fenix-qwen38:candidate-b\n"
        "Application startup complete.\n"
    )

    with pytest.raises(
        performance_evidence.PerformanceEvidenceError,
        match="exactly one",
    ):
        performance_evidence.require_performance_server(path)


def test_contamination_markers_are_counted():
    window = (
        "Triton kernel JIT compilation during inference: x\n"
        "Triton kernel JIT compilation during inference: y\n"
        "RuntimeError: failure\n"
    )

    hits = performance_evidence.contamination_hits(window)

    assert hits == [
        {
            "id": "inference_jit",
            "needle": "Triton kernel JIT compilation during inference",
            "count": 2,
        },
        {
            "id": "runtime_error",
            "needle": "RuntimeError:",
            "count": 1,
        },
    ]


def test_eligibility_is_fail_closed():
    repository = performance_evidence.RepositoryState(
        commit="abc",
        clean=False,
        status=(" M file.py",),
    )
    launch = performance_evidence.LaunchMetadata(
        startup_complete=True,
        trace_values=("0",),
        runtime_images=("fenix-qwen38:candidate-commit",),
    )

    eligible, reasons = performance_evidence.evaluate_eligibility(
        repository=repository,
        launch=launch,
        warmup_records=[_record(0, "warmup")],
        measured_records=[_record(0, "measured")],
        workload_mismatches=[
            {
                "field": "completion_tokens",
                "expected": 4,
                "observed": 3,
            }
        ],
        contamination=[
            {
                "id": "inference_jit",
                "needle": "jit",
                "count": 1,
            }
        ],
        log_window_valid=True,
    )

    assert eligible is False
    assert "repository_not_clean" in reasons
    assert "workload:completion_tokens_mismatch" in reasons
    assert "contamination:inference_jit" in reasons


def test_measured_completion_requires_timing():
    record = _record(0, "measured")
    record["tpot_ms"] = None

    assert (
        performance_evidence._measured_timing_complete([record])
        is False
    )


def _install_success_mocks(monkeypatch):
    monkeypatch.setattr(
        performance_evidence.workload_contract,
        "prepare_workload",
        lambda **kwargs: _prepared(),
    )
    monkeypatch.setattr(
        performance_evidence,
        "repository_state",
        lambda: performance_evidence.RepositoryState(
            commit="commit",
            clean=True,
            status=(),
        ),
    )


def test_eligible_run_writes_auditable_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    calls = []

    def fake_run_benchmark(args: argparse.Namespace):
        calls.append(args.requests)
        phase = "warmup" if len(calls) == 1 else "measured"
        return (
            [_record(i, phase) for i in range(args.requests)],
            float(args.requests),
        )

    monkeypatch.setattr(
        performance_evidence.bench_openai,
        "run_benchmark",
        fake_run_benchmark,
    )
    _install_success_mocks(monkeypatch)

    evidence, code = performance_evidence.run_measurement(config)

    assert code == 0
    assert evidence["performance_eligible"] is True
    assert evidence["evidence_kind"] == "local_measured"
    assert evidence["workload"]["expected_input_tokens"] == 32
    assert evidence["workload"]["expected_output_tokens"] == 4
    assert evidence["workload"]["repetition_index"] == 1
    assert calls == [3, 2]

    measured = config.output
    expected_paths = (
        measured,
        Path(f"{measured}.summary.json"),
        Path(f"{measured}.warmup.jsonl"),
        Path(f"{measured}.prompt.txt"),
        Path(f"{measured}.server-window.log"),
        Path(f"{measured}.evidence.json"),
    )
    for path in expected_paths:
        assert path.is_file()

    payload = json.loads(
        Path(f"{measured}.evidence.json").read_text()
    )
    assert payload["performance_eligible"] is True
    assert payload["repository_commit"] == "commit"
    assert payload["runtime_lane"]["model_revision"] == "model-rev"
    assert set(payload["artifacts_sha256"]) == {
        "measured",
        "summary",
        "warmup",
        "prompt",
        "server_window",
    }


def test_measured_token_mismatch_makes_run_ineligible(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    calls = []

    def fake_run_benchmark(args: argparse.Namespace):
        calls.append(args.requests)
        phase = "warmup" if len(calls) == 1 else "measured"
        records = [_record(i, phase) for i in range(args.requests)]
        if phase == "measured":
            records[0]["prompt_tokens"] = 31
        return records, 1.0

    monkeypatch.setattr(
        performance_evidence.bench_openai,
        "run_benchmark",
        fake_run_benchmark,
    )
    _install_success_mocks(monkeypatch)

    evidence, code = performance_evidence.run_measurement(config)

    assert code == 3
    assert evidence["performance_eligible"] is False
    assert "workload:prompt_tokens_mismatch" in evidence[
        "eligibility_reasons"
    ]
    assert evidence["workload_mismatches"][0]["observed"] == 31


def test_warmup_token_mismatch_stops_before_measurement(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    calls = []

    def fake_run_benchmark(args: argparse.Namespace):
        calls.append(args.requests)
        records = [_record(i, "warmup") for i in range(args.requests)]
        records[0]["completion_tokens"] = 3
        return records, 1.0

    monkeypatch.setattr(
        performance_evidence.bench_openai,
        "run_benchmark",
        fake_run_benchmark,
    )
    _install_success_mocks(monkeypatch)

    evidence, code = performance_evidence.run_measurement(config)

    assert code == 3
    assert calls == [3]
    assert "workload:completion_tokens_mismatch" in evidence[
        "eligibility_reasons"
    ]
    assert not config.output.exists()


def test_measured_jit_makes_run_ineligible(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    calls = []

    def fake_run_benchmark(args: argparse.Namespace):
        calls.append(args.requests)
        if len(calls) == 2:
            with config.server_log.open("a") as stream:
                stream.write(
                    "Triton kernel JIT compilation during inference: kernel\n"
                )
        phase = "warmup" if len(calls) == 1 else "measured"
        return (
            [_record(i, phase) for i in range(args.requests)],
            1.0,
        )

    monkeypatch.setattr(
        performance_evidence.bench_openai,
        "run_benchmark",
        fake_run_benchmark,
    )
    _install_success_mocks(monkeypatch)

    evidence, code = performance_evidence.run_measurement(config)

    assert code == 3
    assert evidence["performance_eligible"] is False
    assert evidence["evidence_kind"] == "diagnostic_measurement"
    assert "contamination:inference_jit" in evidence[
        "eligibility_reasons"
    ]


def test_existing_artifact_set_is_rejected_before_requests(
    tmp_path: Path,
):
    config = _config(tmp_path)
    config.output.write_text("old")

    with pytest.raises(
        performance_evidence.PerformanceEvidenceError,
        match="refusing to overwrite",
    ):
        performance_evidence.run_measurement(config)


def test_dirty_repository_is_rejected_before_requests(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    monkeypatch.setattr(
        performance_evidence,
        "repository_state",
        lambda: performance_evidence.RepositoryState(
            commit="commit",
            clean=False,
            status=(" M scripts/x.py",),
        ),
    )

    with pytest.raises(
        performance_evidence.PerformanceEvidenceError,
        match="clean repository",
    ):
        performance_evidence.run_measurement(config)


def test_invalid_repetition_is_rejected(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    config = performance_evidence.MeasurementConfig(
        **{
            **config.__dict__,
            "repetition_index": 4,
        }
    )
    monkeypatch.setattr(
        performance_evidence,
        "repository_state",
        lambda: performance_evidence.RepositoryState(
            commit="commit",
            clean=True,
            status=(),
        ),
    )

    with pytest.raises(
        performance_evidence.PerformanceEvidenceError,
        match="1..3",
    ):
        performance_evidence.run_measurement(config)


def test_runtime_lane_requires_version_provenance(tmp_path: Path):
    path = tmp_path / "runtime_lane.json"
    path.write_text('{"lane_id":"x"}')

    with pytest.raises(
        performance_evidence.PerformanceEvidenceError,
        match="required provenance",
    ):
        performance_evidence.load_runtime_lane(path)
