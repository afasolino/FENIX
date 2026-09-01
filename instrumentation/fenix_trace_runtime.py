"""Minimal, process-safe runtime trace writer for FENIX instrumentation."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_TRACE_ENABLE_VALUES = frozenset({"1", "true", "yes"})
_TRACE_LOCK = threading.Lock()
_COUNTERS: dict[str, int] = {}


def tracing_enabled() -> bool:
    """Return whether runtime tracing is enabled for the current process."""

    return os.getenv("FENIX_TRACE", "0").lower() in _TRACE_ENABLE_VALUES


def trace_directory() -> Path:
    """Return and create the configured trace directory."""

    path = Path(os.getenv("FENIX_TRACE_DIR", "/fenix-traces"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def next_sequence_id(stream_name: str) -> int:
    """Return a process-local monotonically increasing sequence identifier."""

    with _TRACE_LOCK:
        value = _COUNTERS.get(stream_name, 0)
        _COUNTERS[stream_name] = value + 1
        return value


def emit_trace(stream_name: str, record: dict[str, Any]) -> None:
    """Append one compact JSON record to a trace stream."""

    if not tracing_enabled():
        return

    payload = dict(record)
    payload.setdefault("timestamp_ns", time.monotonic_ns())
    payload.setdefault("pid", os.getpid())

    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    destination = trace_directory() / f"{stream_name}.jsonl"

    with _TRACE_LOCK:
        with destination.open("a") as stream:
            stream.write(serialized + "\n")


# Backwards-compatible names used by the pinned runtime injection template.
enabled = tracing_enabled
next_id = next_sequence_id
emit = emit_trace
