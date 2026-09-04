#!/usr/bin/env python3
"""Runtime-only mmap reader for the FENIX PLE storage bank.

This module is copied into the pinned vLLM overlay as ``vllm.fenix_ple_bank_runtime``.
It deliberately avoids safetensors parsing and model-loader dependencies: the
versioned FENIX bank manifest is the only storage contract consumed at runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


TRUTHY = {"1", "true", "yes", "on"}
SUPPORTED_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
}


class FenixPleBankError(RuntimeError):
    """Raised when the externalized PLE bank violates the runtime contract."""


def _sha256_file(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FenixPleBankError(f"PLE bank manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FenixPleBankError(f"invalid PLE bank manifest JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FenixPleBankError("PLE bank manifest must be a JSON object")
    if payload.get("schema_version") != 1:
        raise FenixPleBankError(
            f"unsupported PLE bank schema: {payload.get('schema_version')!r}"
        )
    if payload.get("artifact_kind") != "fenix_ple_storage_bank":
        raise FenixPleBankError(
            f"unexpected PLE bank artifact kind: {payload.get('artifact_kind')!r}"
        )
    return payload


def _require_int(mapping: dict[str, Any], key: str, *, minimum: int = 0) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise FenixPleBankError(f"PLE bank manifest field {key!r} is invalid")
    return value


class FenixPleBankEmbedding(nn.Module):
    """File-backed PLE row matrix used only inside the CPU-offload worker.

    The bank is mapped as raw ``uint8`` because the checkpoint uses one-byte
    FP8 rows. The worker performs byte-exact row gathers into the existing
    shared CPU output buffer; the GPU side retains the original FP8 scale and
    dequantizes exactly as in the resident path.
    """

    _fenix_ple_bank = True

    def __init__(
        self,
        *,
        manifest_path: Path,
        expected_rows: int,
        embedding_width: int,
        model_index_path: Path,
    ) -> None:
        super().__init__()
        manifest_path = manifest_path.resolve()
        manifest = _load_manifest(manifest_path)
        layout = manifest.get("layout")
        source = manifest.get("source")
        if not isinstance(layout, dict):
            raise FenixPleBankError("PLE bank manifest layout is missing")
        if not isinstance(source, dict):
            raise FenixPleBankError("PLE bank manifest source is missing")

        dtype_name = layout.get("dtype")
        torch_dtype = SUPPORTED_DTYPES.get(dtype_name)
        if torch_dtype is None:
            raise FenixPleBankError(
                f"unsupported PLE bank dtype {dtype_name!r}; "
                f"supported={sorted(SUPPORTED_DTYPES)}"
            )
        total_rows = _require_int(layout, "total_rows", minimum=1)
        row_bytes = _require_int(layout, "row_bytes", minimum=1)
        manifest_width = _require_int(layout, "embedding_width", minimum=1)
        if total_rows != expected_rows:
            raise FenixPleBankError(
                "PLE bank row count does not match runtime geometry: "
                f"bank={total_rows}, runtime={expected_rows}"
            )
        if manifest_width != embedding_width:
            raise FenixPleBankError(
                "PLE bank embedding width does not match runtime geometry: "
                f"bank={manifest_width}, runtime={embedding_width}"
            )
        if row_bytes != embedding_width:
            raise FenixPleBankError(
                "F8_E4M3 PLE bank must contain one byte per embedding value: "
                f"row_bytes={row_bytes}, embedding_width={embedding_width}"
            )

        data_file = manifest.get("data_file")
        data_bytes = manifest.get("data_bytes")
        if not isinstance(data_file, str) or not data_file:
            raise FenixPleBankError("PLE bank data_file is invalid")
        if not isinstance(data_bytes, int) or isinstance(data_bytes, bool):
            raise FenixPleBankError("PLE bank data_bytes is invalid")
        if data_bytes != total_rows * row_bytes:
            raise FenixPleBankError(
                "PLE bank byte length disagrees with row geometry: "
                f"data_bytes={data_bytes}, rows*row_bytes={total_rows * row_bytes}"
            )
        data_path = manifest_path.parent / data_file
        if not data_path.is_file():
            raise FenixPleBankError(f"PLE bank data file does not exist: {data_path}")
        observed_bytes = data_path.stat().st_size
        if observed_bytes != data_bytes:
            raise FenixPleBankError(
                "PLE bank data length mismatch: "
                f"manifest={data_bytes}, observed={observed_bytes}"
            )

        checkpoint_sha = source.get("checkpoint_index_sha256")
        if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
            raise FenixPleBankError("PLE bank checkpoint-index SHA-256 is invalid")
        if not model_index_path.is_file():
            raise FenixPleBankError(
                f"runtime checkpoint index does not exist: {model_index_path}"
            )
        runtime_checkpoint_sha = _sha256_file(model_index_path)
        if runtime_checkpoint_sha != checkpoint_sha:
            raise FenixPleBankError(
                "PLE bank was built from a different checkpoint index: "
                f"bank={checkpoint_sha}, runtime={runtime_checkpoint_sha}"
            )

        # Do not hash the 51+ GB data file here. The bank builder already stores
        # and verifies its SHA-256; re-reading it at every server launch would
        # warm the page cache and destroy the intended cold-storage condition.
        self.org_vocab_size = total_rows
        self.embedding_dim = embedding_width
        self.row_bytes = row_bytes
        self.torch_dtype = torch_dtype
        self.manifest_path = manifest_path
        self.data_path = data_path
        self.data_sha256 = str(manifest.get("data_sha256", ""))
        self.checkpoint_index_sha256 = checkpoint_sha

        flat = torch.from_file(
            str(data_path),
            shared=False,
            size=data_bytes,
            dtype=torch.uint8,
        )
        self._rows = flat.view(total_rows, row_bytes)
        self._fd = os.open(data_path, os.O_RDONLY)
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_RANDOM"):
            try:
                os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_RANDOM)
            except OSError:
                # The hint is an optimization only; mmap correctness is unchanged.
                pass

    @classmethod
    def from_environment(
        cls,
        *,
        expected_rows: int,
        embedding_width: int,
    ) -> "FenixPleBankEmbedding":
        manifest_raw = os.getenv("FENIX_PLE_BANK_MANIFEST", "").strip()
        if not manifest_raw:
            raise FenixPleBankError(
                "FENIX_PLE_STORAGE_MODE=mmap requires FENIX_PLE_BANK_MANIFEST"
            )
        model_index_raw = os.getenv(
            "FENIX_PLE_MODEL_INDEX",
            "/model/model.safetensors.index.json",
        ).strip()
        return cls(
            manifest_path=Path(manifest_raw),
            expected_rows=expected_rows,
            embedding_width=embedding_width,
            model_index_path=Path(model_index_raw),
        )

    def gather_into(
        self,
        row_ids: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        """Gather raw FP8 rows directly into a preallocated uint8 destination."""
        ids = row_ids.reshape(-1)
        if ids.device.type != "cpu":
            raise FenixPleBankError(
                f"PLE bank row IDs must be on CPU, got {ids.device}"
            )
        if ids.dtype != torch.int64:
            ids = ids.to(dtype=torch.int64)
        if destination.device.type != "cpu":
            raise FenixPleBankError(
                f"PLE bank destination must be on CPU, got {destination.device}"
            )
        if destination.dtype != torch.uint8:
            raise FenixPleBankError(
                f"PLE bank destination must be uint8, got {destination.dtype}"
            )
        expected_shape = (ids.numel(), self.row_bytes)
        if tuple(destination.shape) != expected_shape:
            raise FenixPleBankError(
                "PLE bank destination shape mismatch: "
                f"expected={expected_shape}, observed={tuple(destination.shape)}"
            )
        if not destination.is_contiguous():
            raise FenixPleBankError("PLE bank destination must be contiguous")
        if ids.numel():
            minimum = int(ids.min().item())
            maximum = int(ids.max().item())
            if minimum < 0 or maximum >= self.org_vocab_size:
                raise FenixPleBankError(
                    "PLE bank row ID out of range: "
                    f"min={minimum}, max={maximum}, rows={self.org_vocab_size}"
                )
        torch.index_select(self._rows, 0, ids, out=destination)

    def close(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            os.close(fd)
            self._fd = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "FenixPleBankEmbedding",
    "FenixPleBankError",
]
