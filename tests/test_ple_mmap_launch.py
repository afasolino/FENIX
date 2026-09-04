from pathlib import Path

import pytest

from scripts.launch_vllm import build_command, build_environment


def _build(**overrides):
    kwargs = dict(
        model_directory=Path("/model"),
        runtime_directory=Path("/runtime"),
        trace_directory=Path("/traces"),
        gpu_ids=["0"],
        port=8000,
        cpu_offload_gib=40,
        hot_experts=16,
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=1073741824,
        trace_enabled=False,
    )
    kwargs.update(overrides)
    return build_command(**kwargs)


def test_resident_ple_is_default_and_has_no_bank_mount():
    environment, command = _build()

    assert environment["FENIX_PLE_STORAGE_MODE"] == "resident"
    assert "FENIX_PLE_BANK_MANIFEST" not in environment
    assert not any("/fenix-ple-bank" in value for value in command)


def test_mmap_ple_mounts_bank_read_only_and_records_provenance():
    manifest = Path("/storage/ple.manifest.json")
    environment, command = _build(
        ple_storage_mode="mmap",
        ple_bank_manifest=manifest,
    )

    assert environment["FENIX_PLE_STORAGE_MODE"] == "mmap"
    assert environment["FENIX_PLE_BANK_MANIFEST"] == (
        "/fenix-ple-bank/ple.manifest.json"
    )
    assert environment["FENIX_PLE_MODEL_INDEX"] == (
        "/model/model.safetensors.index.json"
    )
    assert "/storage:/fenix-ple-bank:ro" in command


def test_mmap_ple_requires_manifest():
    with pytest.raises(ValueError, match="requires a bank manifest"):
        build_environment(
            16,
            False,
            Path("/runtime"),
            ple_storage_mode="mmap",
        )


def test_resident_ple_rejects_bank_manifest():
    with pytest.raises(ValueError, match="does not accept"):
        build_environment(
            16,
            False,
            Path("/runtime"),
            ple_storage_mode="resident",
            ple_bank_manifest=Path("/storage/ple.manifest.json"),
        )
