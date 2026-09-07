"""Shared fail-closed utilities for the FENIX H3 conventional hierarchy study."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

GIB = 1024 ** 3
_EXECUTION_REPOSITORY_IDENTITY: dict[str, Any] | None = None


class H3Error(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise H3Error(f"{path}: expected JSON object")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise H3Error(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise H3Error(f"{path}:{line_number}: expected object")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_execution_repository_identity(identity: dict[str, Any] | None) -> None:
    """Set CLI-scoped FENIX implementation provenance for emitted H3 artifacts."""
    global _EXECUTION_REPOSITORY_IDENTITY
    _EXECUTION_REPOSITORY_IDENTITY = dict(identity) if identity is not None else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if _EXECUTION_REPOSITORY_IDENTITY is not None:
        kind = payload.get("artifact_kind")
        if isinstance(kind, str) and kind.startswith("fenix_h3_"):
            existing = payload.get("execution_repository")
            if existing is not None and existing != _EXECUTION_REPOSITORY_IDENTITY:
                raise H3Error(f"{path}: conflicting execution-repository provenance")
            payload = dict(payload)
            payload["execution_repository"] = dict(_EXECUTION_REPOSITORY_IDENTITY)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)



def execution_repository_identity(payload: dict[str, Any], label: str = "artifact") -> dict[str, Any]:
    identity = payload.get("execution_repository")
    if not isinstance(identity, dict):
        raise H3Error(f"{label}: execution_repository provenance is missing")
    required = ("head", "tree", "required_base_commit")
    missing = [key for key in required if not identity.get(key)]
    if missing or identity.get("clean") is not True:
        raise H3Error(f"{label}: incomplete execution_repository provenance: {missing}")
    return identity


def require_same_execution_repository(*labeled_payloads: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    if not labeled_payloads:
        raise H3Error("no artifacts supplied for execution-repository provenance check")
    observed: list[tuple[str, dict[str, Any]]] = [
        (label, execution_repository_identity(payload, label))
        for label, payload in labeled_payloads
    ]
    reference = observed[0][1]
    fingerprint_keys = ("head", "tree", "required_base_commit")
    reference_fp = tuple(reference.get(key) for key in fingerprint_keys)
    for label, identity in observed[1:]:
        fingerprint = tuple(identity.get(key) for key in fingerprint_keys)
        if fingerprint != reference_fp:
            raise H3Error(
                f"{label}: execution-repository identity differs from {observed[0][0]}: "
                f"{fingerprint} != {reference_fp}"
            )
    return dict(reference)


def require_contract_sha(payload: dict[str, Any], expected_sha256: str, label: str) -> None:
    observed = payload.get("contract_sha256")
    if observed is None:
        observed = (payload.get("provenance") or {}).get("contract_sha256")
    if observed != expected_sha256:
        raise H3Error(f"{label}: H3 contract SHA mismatch: {observed} != {expected_sha256}")

def stable_u64(seed: int, *parts: object) -> int:
    text = "|".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def parse_layer_id(value: object) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    marker = ".layers."
    if marker not in text:
        raise H3Error(f"cannot extract layer ID from {text!r}")
    tail = text.split(marker, 1)[1]
    token = tail.split(".", 1)[0]
    if not token.isdigit():
        raise H3Error(f"cannot extract layer ID from {text!r}")
    return int(token)
