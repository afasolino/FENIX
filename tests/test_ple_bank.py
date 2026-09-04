from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from instrumentation.ple_bank import (
    StorageBankError,
    build_ple_bank,
    discover_ple_shards,
    read_safetensors_header,
)
from instrumentation.storage_bank import validate_bank_manifest


def _write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]):
    offset = 0
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(payload))


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    prefix = "model.language_model.model.layers.0.ple_embedding.ngram_embedding"
    a = f"{prefix}.shard_0.weight"
    b = f"{prefix}.shard_1.weight"
    _write_safetensors(
        model / "part-1.safetensors",
        [(a, "U8", [2, 3], b"abcdef")],
    )
    _write_safetensors(
        model / "part-2.safetensors",
        [(b, "U8", [1, 3], b"XYZ")],
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {a: "part-1.safetensors", b: "part-2.safetensors"}})
    )
    return model


def test_read_safetensors_header_reports_data_start(tmp_path: Path):
    path = tmp_path / "one.safetensors"
    _write_safetensors(path, [("x", "U8", [1, 3], b"abc")])
    header = read_safetensors_header(path)
    assert header.data_start > 8
    assert header.tensors["x"]["data_offsets"] == [0, 3]


def test_discover_ple_shards_orders_numeric_indices(tmp_path: Path):
    model = _model(tmp_path)
    prefix, records = discover_ple_shards(model)
    assert prefix.endswith("ple_embedding.ngram_embedding")
    assert [record.shard_index for record in records] == [0, 1]
    assert [record.rows for record in records] == [2, 1]


def test_build_ple_bank_is_raw_row_concatenation(tmp_path: Path):
    model = _model(tmp_path)
    out = tmp_path / "bank"
    artifact = build_ple_bank(model_dir=model, output_dir=out)
    assert artifact.data_path.read_bytes() == b"abcdefXYZ"
    manifest = validate_bank_manifest(artifact.manifest_path)
    assert manifest["layout"] == {
        "dtype": "U8",
        "embedding_width": 3,
        "row_bytes": 3,
        "shard_count": 2,
        "total_rows": 3,
    }
    assert [item["output_start"] for item in manifest["shards"]] == [0, 6]


def test_build_refuses_overwrite(tmp_path: Path):
    model = _model(tmp_path)
    out = tmp_path / "bank"
    build_ple_bank(model_dir=model, output_dir=out)
    with pytest.raises(StorageBankError, match="refusing to overwrite"):
        build_ple_bank(model_dir=model, output_dir=out)


def test_multiple_ple_prefixes_require_explicit_selection(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    names = [
        "model.a.ple_embedding.ngram_embedding.shard_0.weight",
        "model.b.ple_embedding.ngram_embedding.shard_0.weight",
    ]
    _write_safetensors(
        model / "part.safetensors",
        [(names[0], "U8", [1, 1], b"a"), (names[1], "U8", [1, 1], b"b")],
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "part.safetensors" for name in names}})
    )
    with pytest.raises(StorageBankError, match="multiple PLE embedding prefixes"):
        discover_ple_shards(model)
