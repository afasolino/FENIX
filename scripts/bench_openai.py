#!/usr/bin/env python3
"""Measure OpenAI-compatible streaming request latency for FENIX.

TTFT is defined as elapsed client time until the first non-empty generated
assistant delta is received. Generated output may be visible ``content`` or
reasoning output. vLLM currently emits reasoning as ``reasoning``; the legacy
``reasoning_content`` field remains accepted for compatibility.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests


DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3.8-flash-next"
DEFAULT_PROMPT = (
    "Explain deterministic sparse lookup prefetch during autoregressive decoding."
)
SSE_DATA_PREFIX = "data: "
SSE_DONE = "[DONE]"
GENERATED_DELTA_FIELDS = ("content", "reasoning", "reasoning_content")


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile for a non-empty sample."""

    if not values:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"percentile fraction must be in [0, 1], got {fraction}")

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_sse_event(raw_line: str) -> dict[str, Any] | None:
    """Parse one OpenAI-style SSE data line.

    Blank lines, non-data SSE fields, and the terminal ``[DONE]`` sentinel do
    not carry a JSON event and return ``None``. Malformed JSON is deliberately
    not suppressed: benchmark protocol drift must fail closed.
    """

    if not raw_line or not raw_line.startswith(SSE_DATA_PREFIX):
        return None

    payload = raw_line[len(SSE_DATA_PREFIX) :]
    if payload == SSE_DONE:
        return None

    event = json.loads(payload)
    if not isinstance(event, dict):
        raise ValueError("SSE data payload must decode to a JSON object")
    return event


def first_choice_delta(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the first choice delta, or an empty mapping for non-choice events."""

    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}

    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        return {}

    delta = first_choice.get("delta")
    return delta if isinstance(delta, Mapping) else {}


def delta_contains_generated_output(delta: Mapping[str, Any]) -> bool:
    """Whether a stream delta contains non-empty generated assistant output."""

    for field in GENERATED_DELTA_FIELDS:
        value = delta.get(field)
        if isinstance(value, str) and value:
            return True
    return False


def _request_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def run_one(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_id: str,
) -> dict[str, Any]:
    """Execute one streaming request and return client-observed timing."""

    start_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    usage: Mapping[str, Any] | None = None

    with requests.post(
        url,
        json=_request_payload(model, prompt, max_tokens, temperature),
        stream=True,
        timeout=3600,
    ) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            event = parse_sse_event(raw_line)
            if event is None:
                continue

            event_usage = event.get("usage")
            if isinstance(event_usage, Mapping):
                usage = event_usage

            if first_token_ns is None:
                delta = first_choice_delta(event)
                if delta_contains_generated_output(delta):
                    first_token_ns = time.perf_counter_ns()

    end_ns = time.perf_counter_ns()
    usage = usage or {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")

    if (
        isinstance(completion_tokens, int)
        and completion_tokens > 0
        and first_token_ns is None
    ):
        raise RuntimeError(
            "response reported completion tokens, but no recognized generated "
            f"delta was observed; supported fields={GENERATED_DELTA_FIELDS}"
        )

    ttft_ms = (
        None
        if first_token_ns is None
        else (first_token_ns - start_ns) / 1e6
    )
    e2e_ms = (end_ns - start_ns) / 1e6

    if (
        first_token_ns is None
        or not isinstance(completion_tokens, int)
        or completion_tokens < 2
    ):
        tpot_ms = None
        decode_tokens_s = None
    else:
        decode_duration_s = (end_ns - first_token_ns) / 1e9
        tpot_ms = (end_ns - first_token_ns) / 1e6 / (completion_tokens - 1)
        decode_tokens_s = (
            None
            if decode_duration_s <= 0
            else (completion_tokens - 1) / decode_duration_s
        )

    return {
        "request_id": request_id,
        "start_ns": start_ns,
        "first_token_ns": first_token_ns,
        "end_ns": end_ns,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot_ms,
        "decode_tokens_s": decode_tokens_s,
    }


def summarize_results(
    results: Sequence[Mapping[str, Any]],
    wall_s: float,
    concurrency: int,
) -> dict[str, Any]:
    """Summarize successful requests without relabeling token counts as phases."""

    successful = [record for record in results if "error" not in record]
    summary: dict[str, Any] = {
        "requests": len(results),
        "success": len(successful),
        "concurrency": concurrency,
        "wall_s": wall_s,
        "aggregate_completion_tokens_s": (
            sum(record.get("completion_tokens") or 0 for record in successful)
            / wall_s
            if wall_s > 0
            else None
        ),
        "aggregate_prompt_tokens_s": (
            sum(record.get("prompt_tokens") or 0 for record in successful)
            / wall_s
            if wall_s > 0
            else None
        ),
    }

    for field in ("ttft_ms", "tpot_ms", "e2e_ms"):
        samples = [
            float(record[field])
            for record in successful
            if record.get(field) is not None
        ]
        for name, fraction in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
            summary[f"{field}_{name}"] = percentile(samples, fraction)

    return summary


def _run_request_job(
    ordinal: int,
    request_id: str,
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
) -> dict[str, Any]:
    try:
        result = run_one(
            url,
            model,
            prompt,
            max_tokens,
            temperature,
            request_id,
        )
        result.update(ordinal=ordinal, concurrency=concurrency)
        return result
    except Exception as exc:
        return {
            "request_id": request_id,
            "ordinal": ordinal,
            "concurrency": concurrency,
            "error": repr(exc),
        }


def run_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, Any]], float]:
    """Run all requests with bounded client concurrency."""

    jobs = [
        (ordinal, f"fenix-{ordinal:06d}-{uuid.uuid4().hex[:8]}")
        for ordinal in range(args.requests)
    ]

    start_ns = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                _run_request_job,
                ordinal,
                request_id,
                url=args.url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                concurrency=args.concurrency,
            )
            for ordinal, request_id in jobs
        ]
        results = [future.result() for future in futures]
    end_ns = time.perf_counter_ns()

    results.sort(key=lambda record: int(record["ordinal"]))
    return results, (end_ns - start_ns) / 1e9


def write_results(
    output_path: Path,
    results: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        for record in results:
            stream.write(json.dumps(record) + "\n")

    summary_path = Path(f"{output_path}.summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.requests < 1:
        raise SystemExit("--requests must be >= 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    results, wall_s = run_benchmark(args)
    summary = summarize_results(results, wall_s, args.concurrency)
    write_results(args.out, results, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
