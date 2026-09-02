from pathlib import Path

import json

from scripts import bench_openai


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        assert decode_unicode is True
        yield 'data: ' + json.dumps({
            "choices": [{"delta": {"reasoning_content": "thinking"}}]
        })
        yield 'data: ' + json.dumps({
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            "choices": [],
        })
        yield "data: [DONE]"


def test_reasoning_content_establishes_first_token(monkeypatch):
    monkeypatch.setattr(
        bench_openai.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
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
    assert result["tpot_ms"] is not None
    assert result["decode_tokens_s"] is not None


def test_summary_metric_names_are_not_misleading():
    source = Path("scripts/bench_openai.py").read_text()
    assert "aggregate_completion_tokens_s" in source
    assert "aggregate_prompt_tokens_s" in source
    assert "aggregate_prefill_tokens_s" not in source
