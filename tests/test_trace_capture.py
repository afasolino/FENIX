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
