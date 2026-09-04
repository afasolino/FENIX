#!/usr/bin/env python3
"""Build a contiguous PLE bank directly from safetensors checkpoint bytes."""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from instrumentation.storage_bank import (
    BANK_SCHEMA_VERSION,
    BankArtifact,
    StorageBankError,
    atomic_write_json,
    copy_range,
    ensure_artifacts_absent,
    sha256_file,
)


DEFAULT_INDEX = "model.safetensors.index.json"
PLE_SHARD_RE = re.compile(
    r"^(?P<prefix>.+\.ple_embedding\.ngram_embedding)\.shard_(?P<index>\d+)\.weight$"
)


@dataclass(frozen=True)
class SafeTensorRecord:
    name: str
    filename: str
    dtype: str
    shape: tuple[int, ...]
    file_start: int
    file_end: int
    shard_index: int

    @property
    def data_bytes(self) -> int:
        return self.file_end - self.file_start

    @property
    def rows(self) -> int:
        return self.shape[0]


@dataclass(frozen=True)
class SafeTensorHeader:
    data_start: int
    tensors: Mapping[str, Mapping[str, Any]]


def read_safetensors_header(path: Path) -> SafeTensorHeader:
    if not path.is_file():
        raise StorageBankError(f"safetensors shard does not exist: {path}")
    size = path.stat().st_size
    if size < 8:
        raise StorageBankError(f"safetensors shard is too small: {path}")

    with path.open("rb") as stream:
        raw = stream.read(8)
        header_bytes = struct.unpack("<Q", raw)[0]
        if header_bytes < 2 or header_bytes > size - 8:
            raise StorageBankError(
                f"invalid safetensors header length {header_bytes}: {path}"
            )
        encoded = stream.read(header_bytes)
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageBankError(f"invalid safetensors header JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise StorageBankError(f"safetensors header is not an object: {path}")

    tensors = {
        name: value
        for name, value in payload.items()
        if name != "__metadata__"
    }
    return SafeTensorHeader(data_start=8 + header_bytes, tensors=tensors)


def load_weight_map(model_dir: Path, index_name: str = DEFAULT_INDEX) -> dict[str, str]:
    index_path = model_dir / index_name
    if not index_path.is_file():
        raise StorageBankError(f"checkpoint index does not exist: {index_path}")
    try:
        payload = json.loads(index_path.read_text())
    except json.JSONDecodeError as exc:
        raise StorageBankError(f"invalid checkpoint index JSON: {index_path}") from exc
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise StorageBankError("checkpoint index weight_map is missing or empty")
    output: dict[str, str] = {}
    for name, filename in weight_map.items():
        if not isinstance(name, str) or not isinstance(filename, str):
            raise StorageBankError("checkpoint weight_map must map strings to strings")
        output[name] = filename
    return output


def discover_ple_shards(
    model_dir: Path,
    *,
    tensor_prefix: str | None = None,
    index_name: str = DEFAULT_INDEX,
) -> tuple[str, list[SafeTensorRecord]]:
    weight_map = load_weight_map(model_dir, index_name=index_name)
    matches: list[tuple[str, int, str, str]] = []
    prefixes: set[str] = set()

    for name, filename in weight_map.items():
        matched = PLE_SHARD_RE.match(name)
        if matched is None:
            continue
        prefix = matched.group("prefix")
        if tensor_prefix is not None and prefix != tensor_prefix:
            continue
        prefixes.add(prefix)
        matches.append((prefix, int(matched.group("index")), name, filename))

    if not matches:
        suffix = f" for prefix {tensor_prefix!r}" if tensor_prefix else ""
        raise StorageBankError(f"checkpoint contains no PLE embedding shards{suffix}")
    if tensor_prefix is None and len(prefixes) != 1:
        raise StorageBankError(
            "checkpoint contains multiple PLE embedding prefixes; pass --tensor-prefix: "
            + ", ".join(sorted(prefixes))
        )
    selected_prefix = tensor_prefix or next(iter(prefixes))

    matches.sort(key=lambda item: item[1])
    indices = [item[1] for item in matches]
    if indices != list(range(len(indices))):
        raise StorageBankError(
            "PLE shard indices must be contiguous from zero; observed=" + repr(indices)
        )

    headers: dict[str, SafeTensorHeader] = {}
    records: list[SafeTensorRecord] = []
    dtype: str | None = None
    columns: int | None = None

    for _prefix, shard_index, name, filename in matches:
        shard_path = model_dir / filename
        header = headers.get(filename)
        if header is None:
            header = read_safetensors_header(shard_path)
            headers[filename] = header
        tensor = header.tensors.get(name)
        if not isinstance(tensor, dict):
            raise StorageBankError(
                f"checkpoint index references missing tensor {name!r} in {filename}"
            )
        raw_dtype = tensor.get("dtype")
        raw_shape = tensor.get("shape")
        raw_offsets = tensor.get("data_offsets")
        if not isinstance(raw_dtype, str) or not raw_dtype:
            raise StorageBankError(f"tensor dtype is invalid: {name}")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 2
            or any(not isinstance(value, int) or value < 1 for value in raw_shape)
        ):
            raise StorageBankError(f"PLE shard must be rank-2 and non-empty: {name}")
        if (
            not isinstance(raw_offsets, list)
            or len(raw_offsets) != 2
            or any(not isinstance(value, int) for value in raw_offsets)
        ):
            raise StorageBankError(f"tensor data_offsets are invalid: {name}")
        start, end = raw_offsets
        if start < 0 or end <= start:
            raise StorageBankError(f"tensor data_offsets are invalid: {name}")
        file_start = header.data_start + start
        file_end = header.data_start + end
        if file_end > shard_path.stat().st_size:
            raise StorageBankError(f"tensor range exceeds safetensors shard: {name}")

        shape = (int(raw_shape[0]), int(raw_shape[1]))
        tensor_bytes = file_end - file_start
        if tensor_bytes % shape[0]:
            raise StorageBankError(f"PLE shard bytes are not divisible by rows: {name}")
        row_bytes = tensor_bytes // shape[0]
        if row_bytes % shape[1]:
            raise StorageBankError(
                f"PLE shard row bytes are not divisible by embedding width: {name}"
            )

        if dtype is None:
            dtype = raw_dtype
        elif dtype != raw_dtype:
            raise StorageBankError(
                f"PLE shard dtype mismatch: expected={dtype}, observed={raw_dtype}"
            )
        if columns is None:
            columns = shape[1]
        elif columns != shape[1]:
            raise StorageBankError(
                f"PLE shard embedding width mismatch: expected={columns}, observed={shape[1]}"
            )

        records.append(
            SafeTensorRecord(
                name=name,
                filename=filename,
                dtype=raw_dtype,
                shape=shape,
                file_start=file_start,
                file_end=file_end,
                shard_index=shard_index,
            )
        )

    return selected_prefix, records


def build_ple_bank(
    *,
    model_dir: Path,
    output_dir: Path,
    tensor_prefix: str | None = None,
    index_name: str = DEFAULT_INDEX,
) -> BankArtifact:
    model_dir = model_dir.resolve()
    output_dir = output_dir.resolve()
    prefix, records = discover_ple_shards(
        model_dir,
        tensor_prefix=tensor_prefix,
        index_name=index_name,
    )

    data_path = output_dir / "ple.bin"
    manifest_path = output_dir / "ple.manifest.json"
    ensure_artifacts_absent((data_path, manifest_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / ".ple.bin.tmp"
    if temporary.exists():
        raise StorageBankError(f"stale temporary PLE bank exists: {temporary}")

    output_offset = 0
    manifest_records: list[dict[str, Any]] = []
    try:
        with temporary.open("xb") as destination:
            for record in records:
                source_path = model_dir / record.filename
                with source_path.open("rb") as source:
                    copied = copy_range(
                        source,
                        destination,
                        start=record.file_start,
                        end=record.file_end,
                    )
                if copied != record.data_bytes:
                    raise StorageBankError(
                        f"short copy for {record.name}: {copied}/{record.data_bytes}"
                    )
                manifest_records.append(
                    {
                        **asdict(record),
                        "shape": list(record.shape),
                        "output_start": output_offset,
                        "output_end": output_offset + copied,
                    }
                )
                output_offset += copied
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, data_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    data_sha256 = sha256_file(data_path)
    row_bytes_set = {
        record.data_bytes // record.rows
        for record in records
    }
    if len(row_bytes_set) != 1:
        data_path.unlink(missing_ok=True)
        raise StorageBankError(
            "PLE shards do not have one row width: " + repr(sorted(row_bytes_set))
        )
    row_bytes = next(iter(row_bytes_set))
    total_rows = sum(record.rows for record in records)

    index_path = model_dir / index_name
    manifest = {
        "schema_version": BANK_SCHEMA_VERSION,
        "artifact_kind": "fenix_ple_storage_bank",
        "data_file": data_path.name,
        "data_bytes": data_path.stat().st_size,
        "data_sha256": data_sha256,
        "source": {
            "model_directory": str(model_dir),
            "checkpoint_index": index_name,
            "checkpoint_index_sha256": sha256_file(index_path),
            "tensor_prefix": prefix,
        },
        "layout": {
            "dtype": records[0].dtype,
            "embedding_width": records[0].shape[1],
            "row_bytes": row_bytes,
            "total_rows": total_rows,
            "shard_count": len(records),
        },
        "shards": manifest_records,
    }
    atomic_write_json(manifest_path, manifest)
    return BankArtifact(
        data_path=data_path,
        manifest_path=manifest_path,
        data_sha256=data_sha256,
        data_bytes=data_path.stat().st_size,
    )
