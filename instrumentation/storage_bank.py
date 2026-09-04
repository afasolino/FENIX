#!/usr/bin/env python3
"""Versioned raw-storage banks for FENIX externalized model state.

The bank format intentionally contains no Python or torch serialization. Large
runtime tensors are represented by raw contiguous bytes plus a small JSON
manifest with exact source provenance and SHA-256 checksums. This keeps cold
storage independently inspectable and mmap-friendly.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping


BANK_SCHEMA_VERSION = 1
COPY_CHUNK_BYTES = 16 * 1024 * 1024


class StorageBankError(RuntimeError):
    """Raised when a storage-bank input or artifact violates the contract."""


@dataclass(frozen=True)
class ByteRange:
    path: Path
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class BankArtifact:
    data_path: Path
    manifest_path: Path
    data_sha256: str
    data_bytes: int


def sha256_file(path: Path, *, chunk_bytes: int = COPY_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def copy_range(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    start: int,
    end: int,
    chunk_bytes: int = COPY_CHUNK_BYTES,
) -> int:
    if start < 0 or end < start:
        raise StorageBankError(f"invalid source range: start={start}, end={end}")
    if chunk_bytes < 1:
        raise StorageBankError("chunk_bytes must be positive")

    source.seek(start)
    remaining = end - start
    copied = 0
    while remaining:
        chunk = source.read(min(chunk_bytes, remaining))
        if not chunk:
            raise StorageBankError(
                f"source ended early after {copied} of {end - start} bytes"
            )
        destination.write(chunk)
        copied += len(chunk)
        remaining -= len(chunk)
    return copied


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_bank_manifest(
    manifest_path: Path,
    *,
    verify_data_sha256: bool = True,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise StorageBankError(f"bank manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise StorageBankError(f"invalid bank manifest JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise StorageBankError("bank manifest must be a JSON object")
    if manifest.get("schema_version") != BANK_SCHEMA_VERSION:
        raise StorageBankError(
            f"unsupported bank schema: {manifest.get('schema_version')!r}"
        )

    data_file = manifest.get("data_file")
    data_bytes = manifest.get("data_bytes")
    data_sha256 = manifest.get("data_sha256")
    if not isinstance(data_file, str) or not data_file:
        raise StorageBankError("bank manifest data_file is invalid")
    if not isinstance(data_bytes, int) or isinstance(data_bytes, bool) or data_bytes < 0:
        raise StorageBankError("bank manifest data_bytes is invalid")
    if not isinstance(data_sha256, str) or len(data_sha256) != 64:
        raise StorageBankError("bank manifest data_sha256 is invalid")

    data_path = manifest_path.parent / data_file
    if not data_path.is_file():
        raise StorageBankError(f"bank data file does not exist: {data_path}")
    observed_bytes = data_path.stat().st_size
    if observed_bytes != data_bytes:
        raise StorageBankError(
            f"bank data length mismatch: expected={data_bytes}, observed={observed_bytes}"
        )
    if verify_data_sha256:
        observed_sha256 = sha256_file(data_path)
        if observed_sha256 != data_sha256:
            raise StorageBankError(
                "bank data SHA-256 mismatch: "
                f"expected={data_sha256}, observed={observed_sha256}"
            )
    return manifest


def ensure_artifacts_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise StorageBankError(
            "refusing to overwrite existing storage-bank artifacts: "
            + ", ".join(existing)
        )
