#!/usr/bin/env python3
"""Trace-window isolation and provenance utilities for FENIX campaigns."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


STARTUP_MARKER = "Application startup complete."
TRACE_PATTERNS = (
    re.compile(r'"FENIX_TRACE"\s*:\s*"([01])"'),
    re.compile(r"\bFENIX_TRACE=([01])\b"),
)
FENIX_IMAGE_PATTERN = re.compile(
    r"\b(?:localhost/)?fenix-qwen38:[A-Za-z0-9_.-]+\b"
)
TRACE_MARKERS: tuple[tuple[str, str], ...] = (
    ("inference_jit", "Triton kernel JIT compilation during inference"),
    ("cuda_oom", "CUDA out of memory"),
    ("torch_oom", "torch.OutOfMemoryError"),
    ("allocator_mapping_oom", "memory mapping failed with OOM"),
    ("python_traceback", "Traceback (most recent call last)"),
    ("runtime_error", "RuntimeError:"),
    ("shm_stall", "No available shared memory broadcast block found in 60 seconds"),
)
FATAL_TRACE_MARKERS = frozenset(
    {
        "cuda_oom",
        "torch_oom",
        "allocator_mapping_oom",
        "python_traceback",
        "runtime_error",
    }
)


class TraceCaptureError(RuntimeError):
    """Raised when trace isolation or provenance cannot be established."""


@dataclass(frozen=True)
class TraceLaunchMetadata:
    startup_complete: bool
    trace_values: tuple[str, ...]
    runtime_images: tuple[str, ...]


@dataclass(frozen=True)
class RepositoryState:
    commit: str
    clean: bool
    status: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def repository_state(root: Path) -> RepositoryState:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    raw_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    status = tuple(line for line in raw_status.splitlines() if line.strip())
    return RepositoryState(commit=commit, clean=not status, status=status)


def inspect_trace_launch(text: str) -> TraceLaunchMetadata:
    values: set[str] = set()
    for pattern in TRACE_PATTERNS:
        values.update(pattern.findall(text))
    images = tuple(sorted(set(FENIX_IMAGE_PATTERN.findall(text))))
    return TraceLaunchMetadata(
        startup_complete=STARTUP_MARKER in text,
        trace_values=tuple(sorted(values)),
        runtime_images=images,
    )


def require_trace_server(server_log: Path) -> TraceLaunchMetadata:
    if not server_log.is_file():
        raise TraceCaptureError(f"server log does not exist: {server_log}")
    launch = inspect_trace_launch(server_log.read_text(errors="replace"))
    failures: list[str] = []
    if not launch.startup_complete:
        failures.append("server startup marker is missing")
    if launch.trace_values != ("1",):
        failures.append(
            "trace server must expose exactly FENIX_TRACE=1; "
            f"observed={launch.trace_values or 'missing'}"
        )
    if len(launch.runtime_images) != 1:
        failures.append(
            "server log must identify exactly one FENIX runtime image; "
            f"observed={launch.runtime_images or 'missing'}"
        )
    if failures:
        raise TraceCaptureError("; ".join(failures))
    return launch


def resolve_image_id(root: Path, image: str) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.fenix_podman",
            "image",
            "inspect",
            image,
            "--format",
            "{{.Id}}",
        ],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise TraceCaptureError(
            f"cannot resolve runtime image ID for {image}: {completed.stderr.strip()}"
        )
    image_id = completed.stdout.strip()
    if not image_id.startswith("sha256:"):
        raise TraceCaptureError(f"unexpected runtime image ID: {image_id!r}")
    return image_id


def load_runtime_lane(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    try:
        return {
            "lane_id": payload["lane_id"],
            "runtime_repository": payload["runtime"]["repository"],
            "runtime_revision": payload["runtime"]["revision"],
            "base_container_image": payload["runtime"]["container_image"],
            "model_repository": payload["model"]["repository"],
            "model_revision": payload["model"]["revision"],
        }
    except (KeyError, TypeError) as exc:
        raise TraceCaptureError(
            f"runtime lane is missing required provenance fields: {path}"
        ) from exc


def file_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def read_byte_window(path: Path, start: int, end: int) -> bytes:
    if start < 0 or end < start:
        raise TraceCaptureError(
            f"invalid byte window for {path}: start={start}, end={end}"
        )
    if not path.is_file():
        if start == end == 0:
            return b""
        raise TraceCaptureError(f"trace source disappeared: {path}")
    size = path.stat().st_size
    if size < end:
        raise TraceCaptureError(
            f"trace source shrank during capture: {path}, size={size}, end={end}"
        )
    with path.open("rb") as stream:
        stream.seek(start)
        return stream.read(end - start)


def capture_jsonl_window(
    source: Path,
    start: int,
    end: int,
    destination: Path,
    *,
    required: bool = True,
) -> list[dict[str, Any]]:
    payload = read_byte_window(source, start, end)
    if not payload:
        if required:
            raise TraceCaptureError(f"trace window is empty: {source}")
        atomic_write_text(destination, "")
        return []
    if not payload.endswith(b"\n"):
        raise TraceCaptureError(f"trace window ends with a partial JSONL record: {source}")

    records: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TraceCaptureError(
                f"invalid JSONL in {source} window at record {index}"
            ) from exc
        if not isinstance(value, dict):
            raise TraceCaptureError(
                f"non-object JSONL record in {source} window at record {index}"
            )
        records.append(value)
    atomic_write_text(destination, payload.decode("utf-8"))
    return records


def contamination_hits(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for marker_id, needle in TRACE_MARKERS:
        count = text.count(needle)
        if count:
            hits.append({"id": marker_id, "needle": needle, "count": count})
    return hits


def fatal_contamination(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in hits if item.get("id") in FATAL_TRACE_MARKERS]


@contextmanager
def campaign_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TraceCaptureError(
                f"another trace campaign holds the lock: {path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
