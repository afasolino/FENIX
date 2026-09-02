import argparse
import json
from pathlib import Path

import pytest

from scripts import performance_evidence


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


def _server_log(path: Path, trace: str = "0") -> None:
    path.write_text(
        json.dumps({"FENIX_TRACE": trace})
        + "\n"
        + "python -m scripts.fenix_podman run "
        + "fenix-qwen38:hardened-v1 serve /model\n"
        + "Application startup complete.\n"
    )


def _config(tmp_path: Path) -> performance_evidence.MeasurementConfig:
    server_log = tmp_path / "server.log"
    _server_log(server_log)
    runtime_lane = tmp_path / "runtime_lane.json"
    _runtime_lane(runtime_lane)
    return performance_evidence.MeasurementConfig(
        server_log=server_log,
        output=tmp_path / "measured.jsonl",
        url="http://example.invalid/v1/chat/completions",
        model="model",
        prompt="prompt",
        max_tokens=4,
        temperature=0.0,
        concurrency=1,
        warmup_requests=3,
        measured_requests=2,
        log_settle_ms=0,
        runtime_lane=runtime_lane,
    )


def test_launch_log_requires_started_non_trace_server(tmp_path: Path):
    path = tmp_path / "server.log"
    _server_log(path, trace="0")

    metadata = performance_evidence.require_performance_server(path)

    assert metadata.startup_complete is True
    assert metadata.trace_enabled is False
    assert metadata.trace_values == ("0",)
    assert metadata.runtime_images == ("fenix-qwen38:hardened-v1",)


@pytest.mark.parametrize("trace", ("1", None))
def test_trace_or_unknown_launch_is_rejected(tmp_path: Path, trace):
    path = tmp_path / "server.log"
    if trace is None:
        path.write_text("Application startup complete.\n")
    else:
        _server_log(path, trace=trace)

    with pytest.raises(
        performance_evidence.PerformanceEvidenceError,
        match="FENIX_TRACE=0",
    ):
        performance_evidence.require_performance_server(path)


def test_conflicting_trace_values_are_rejected(tmp_path: Path):
    path = tmp_path / "server.log"
    path.write_text(
        '{"FENIX_TRACE": "0"}\n'
        "FENIX_TRACE=1\n"
        "Application startup complete.\n"
    )

    with pytest.raises(performance_evidence.PerformanceEvidenceError):
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
        runtime_images=("fenix-qwen38:hardened-v1",),
    )

    eligible, reasons = performance_evidence.evaluate_eligibility(
        repository=repository,
        launch=launch,
        warmup_records=[_record(0, "warmup")],
        measured_records=[_record(0, "measured")],
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
    assert "contamination:inference_jit" in reasons


def test_measured_completion_requires_timing():
    record = _record(0, "measured")
    record["tpot_ms"] = None

    assert (
        performance_evidence._measured_timing_complete([record])
        is False
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
    monkeypatch.setattr(
        performance_evidence,
        "repository_state",
        lambda: performance_evidence.RepositoryState(
            commit="commit",
            clean=True,
            status=(),
        ),
    )

    evidence, code = performance_evidence.run_measurement(config)

    assert code == 0
    assert evidence["performance_eligible"] is True
    assert evidence["evidence_kind"] == "local_measured"
    assert calls == [3, 2]

    measured = config.output
    summary = Path(f"{measured}.summary.json")
    warmup = Path(f"{measured}.warmup.jsonl")
    window = Path(f"{measured}.server-window.log")
    manifest = Path(f"{measured}.evidence.json")

    for path in (measured, summary, warmup, window, manifest):
        assert path.is_file()

    payload = json.loads(manifest.read_text())
    assert payload["performance_eligible"] is True
    assert payload["repository_commit"] == "commit"
    assert payload["runtime_lane"]["model_revision"] == "model-rev"
    assert set(payload["artifacts_sha256"]) == {
        "measured",
        "summary",
        "warmup",
        "server_window",
    }


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
    monkeypatch.setattr(
        performance_evidence,
        "repository_state",
        lambda: performance_evidence.RepositoryState(
            commit="commit",
            clean=True,
            status=(),
        ),
    )

    evidence, code = performance_evidence.run_measurement(config)

    assert code == 3
    assert evidence["performance_eligible"] is False
    assert evidence["evidence_kind"] == "diagnostic_measurement"
    assert "contamination:inference_jit" in evidence[
        "eligibility_reasons"
    ]

    window = Path(f"{config.output}.server-window.log")
    assert "Triton kernel JIT compilation" in window.read_text()


def test_failed_warmup_stops_before_measured_phase(
    tmp_path: Path,
    monkeypatch,
):
    config = _config(tmp_path)
    calls = []

    def fake_run_benchmark(args: argparse.Namespace):
        calls.append(args.requests)
        record = _record(0, "warmup")
        record["error"] = "failure"
        return [record], 1.0

    monkeypatch.setattr(
        performance_evidence.bench_openai,
        "run_benchmark",
        fake_run_benchmark,
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

    evidence, code = performance_evidence.run_measurement(config)

    assert code == 3
    assert calls == [3]
    assert evidence["eligibility_reasons"] == ["warmup_failed"]
    assert not config.output.exists()


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


def test_runtime_lane_requires_version_provenance(tmp_path: Path):
    path = tmp_path / "runtime_lane.json"
    path.write_text('{"lane_id":"x"}')

    with pytest.raises(
        performance_evidence.PerformanceEvidenceError,
        match="required provenance",
    ):
        performance_evidence.load_runtime_lane(path)
