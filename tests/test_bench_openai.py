import json

import pytest

from scripts import bench_openai


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        assert decode_unicode is True
        yield from self._lines


def data_event(payload):
    return "data: " + json.dumps(payload)


def test_parse_sse_event_handles_protocol_framing():
    assert bench_openai.parse_sse_event("") is None
    assert bench_openai.parse_sse_event("event: message") is None
    assert bench_openai.parse_sse_event("data: [DONE]") is None
    assert bench_openai.parse_sse_event(
        'data: {"choices":[]}'
    ) == {"choices": []}


def test_parse_sse_event_fails_closed_on_malformed_json():
    with pytest.raises(json.JSONDecodeError):
        bench_openai.parse_sse_event("data: {not-json}")


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ({"role": "assistant", "content": ""}, False),
        ({"content": "answer"}, True),
        ({"reasoning": "thinking"}, True),
        ({"reasoning_content": "legacy"}, True),
        ({"reasoning": ""}, False),
        ({}, False),
    ],
)
def test_generated_output_detection(delta, expected):
    assert bench_openai.delta_contains_generated_output(delta) is expected


def test_modern_reasoning_stream_establishes_first_token(monkeypatch):
    lines = [
        data_event({
            "choices": [{
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }]
        }),
        data_event({
            "choices": [{
                "delta": {"reasoning": "We"},
                "finish_reason": None,
            }]
        }),
        data_event({
            "choices": [{
                "delta": {"reasoning": " need"},
                "finish_reason": "length",
            }]
        }),
        data_event({
            "choices": [],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        bench_openai.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(lines),
    )

    clock = iter((1_000_000_000, 1_500_000_000, 2_500_000_000))
    monkeypatch.setattr(
        bench_openai.time,
        "perf_counter_ns",
        lambda: next(clock),
    )

    result = bench_openai.run_one(
        "http://example.invalid",
        "model",
        "prompt",
        2,
        0,
        "request",
    )

    assert result["first_token_ns"] == 1_500_000_000
    assert result["ttft_ms"] == 500.0
    assert result["e2e_ms"] == 1500.0
    assert result["tpot_ms"] == 1000.0
    assert result["decode_tokens_s"] == 1.0


def test_legacy_reasoning_content_remains_supported(monkeypatch):
    lines = [
        data_event({
            "choices": [{"delta": {"reasoning_content": "thinking"}}]
        }),
        data_event({
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        bench_openai.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(lines),
    )

    result = bench_openai.run_one(
        "http://example.invalid",
        "model",
        "prompt",
        2,
        0,
        "request",
    )

    assert result["first_token_ns"] is not None
    assert result["ttft_ms"] is not None


def test_completion_without_generated_delta_fails_closed(monkeypatch):
    lines = [
        data_event({
            "choices": [{
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }]
        }),
        data_event({
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        bench_openai.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(lines),
    )

    with pytest.raises(RuntimeError, match="no recognized generated delta"):
        bench_openai.run_one(
            "http://example.invalid",
            "model",
            "prompt",
            2,
            0,
            "request",
        )


def test_summary_metric_names_and_percentiles():
    results = [
        {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "ttft_ms": 100.0,
            "tpot_ms": 20.0,
            "e2e_ms": 200.0,
        },
        {
            "prompt_tokens": 14,
            "completion_tokens": 6,
            "ttft_ms": 300.0,
            "tpot_ms": 40.0,
            "e2e_ms": 400.0,
        },
    ]

    summary = bench_openai.summarize_results(
        results,
        wall_s=2.0,
        concurrency=1,
    )

    assert summary["aggregate_completion_tokens_s"] == 5.0
    assert summary["aggregate_prompt_tokens_s"] == 12.0
    assert "aggregate_prefill_tokens_s" not in summary
    assert summary["ttft_ms_p50"] == 200.0
    assert summary["tpot_ms_p50"] == 30.0
    assert summary["e2e_ms_p50"] == 300.0


def test_percentile_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="must be in"):
        bench_openai.percentile([1.0], 1.1)
