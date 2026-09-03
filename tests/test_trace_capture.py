import json
from pathlib import Path

import pytest

from scripts import trace_capture


def test_trace_server_requires_trace_mode_and_single_image(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        '{"FENIX_TRACE":"1"}\n'
        "fenix-qwen38:candidate serve /model\n"
        "Application startup complete.\n"
    )
    meta = trace_capture.require_trace_server(log)
    assert meta.trace_values == ("1",)
    assert meta.runtime_images == ("fenix-qwen38:candidate",)


def test_trace_server_rejects_performance_mode(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(
        '"FENIX_TRACE": "0"\n'
        "fenix-qwen38:candidate\nApplication startup complete.\n"
    )
    with pytest.raises(trace_capture.TraceCaptureError, match="FENIX_TRACE=1"):
        trace_capture.require_trace_server(log)


def test_jsonl_window_requires_complete_records(tmp_path: Path):
    source = tmp_path / "trace.jsonl"
    source.write_bytes(b'{"a":1}\n{"b":2}')
    with pytest.raises(trace_capture.TraceCaptureError, match="partial"):
        trace_capture.capture_jsonl_window(
            source, 0, source.stat().st_size, tmp_path / "out.jsonl"
        )


def test_trace_contamination_distinguishes_nonfatal_jit():
    text = (
        "Triton kernel JIT compilation during inference\n"
        "RuntimeError: failure\n"
    )
    hits = trace_capture.contamination_hits(text)
    fatal = trace_capture.fatal_contamination(hits)
    assert {item["id"] for item in hits} == {"inference_jit", "runtime_error"}
    assert [item["id"] for item in fatal] == ["runtime_error"]


def test_campaign_lock_is_non_reentrant(tmp_path: Path):
    lock = tmp_path / "campaign.lock"
    with trace_capture.campaign_lock(lock):
        with pytest.raises(trace_capture.TraceCaptureError, match="holds the lock"):
            with trace_capture.campaign_lock(lock):
                pass


def test_resolve_image_id_normalizes_bare_podman_id(monkeypatch):
    digest = "6207dfc7ab8761cc56559977d89b845549b429d0f1affd461884404931688a4b"

    def fake_run(*args, **kwargs):
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": digest + "\n",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(trace_capture.subprocess, "run", fake_run)

    assert trace_capture.resolve_image_id(
        Path("."), "fenix-qwen38:candidate"
    ) == f"sha256:{digest}"


def test_resolve_image_id_accepts_prefixed_podman_id(monkeypatch):
    digest = "6207dfc7ab8761cc56559977d89b845549b429d0f1affd461884404931688a4b"

    def fake_run(*args, **kwargs):
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": f"sha256:{digest}\n",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(trace_capture.subprocess, "run", fake_run)

    assert trace_capture.resolve_image_id(
        Path("."), "fenix-qwen38:candidate"
    ) == f"sha256:{digest}"


def test_resolve_image_id_rejects_malformed_id(monkeypatch):
    def fake_run(*args, **kwargs):
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": "not-an-image-id\n",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(trace_capture.subprocess, "run", fake_run)

    with pytest.raises(
        trace_capture.TraceCaptureError,
        match="unexpected runtime image ID",
    ):
        trace_capture.resolve_image_id(
            Path("."), "fenix-qwen38:candidate"
        )
