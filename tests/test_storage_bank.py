from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from instrumentation.storage_bank import (
    StorageBankError,
    atomic_write_json,
    copy_range,
    validate_bank_manifest,
)


def test_copy_range_copies_exact_interval(tmp_path: Path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(bytes(range(64)))
    with source.open("rb") as src, destination.open("wb") as dst:
        copied = copy_range(src, dst, start=7, end=31, chunk_bytes=5)
    assert copied == 24
    assert destination.read_bytes() == bytes(range(7, 31))


def test_copy_range_fails_if_source_ends_early(tmp_path: Path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"abc")
    with source.open("rb") as src, destination.open("wb") as dst:
        with pytest.raises(StorageBankError, match="ended early"):
            copy_range(src, dst, start=1, end=8)


def test_validate_bank_manifest_checks_length_and_hash(tmp_path: Path):
    data = tmp_path / "bank.bin"
    data.write_bytes(b"FENIX")
    manifest = tmp_path / "bank.manifest.json"
    atomic_write_json(
        manifest,
        {
            "schema_version": 1,
            "data_file": data.name,
            "data_bytes": 5,
            "data_sha256": hashlib.sha256(b"FENIX").hexdigest(),
        },
    )
    assert validate_bank_manifest(manifest)["data_bytes"] == 5
    data.write_bytes(b"fenix")
    with pytest.raises(StorageBankError, match="SHA-256 mismatch"):
        validate_bank_manifest(manifest)


def test_validate_bank_manifest_can_skip_full_hash(tmp_path: Path):
    data = tmp_path / "bank.bin"
    data.write_bytes(b"abc")
    manifest = tmp_path / "bank.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_file": "bank.bin",
                "data_bytes": 3,
                "data_sha256": "0" * 64,
            }
        )
    )
    assert validate_bank_manifest(manifest, verify_data_sha256=False)["data_bytes"] == 3
