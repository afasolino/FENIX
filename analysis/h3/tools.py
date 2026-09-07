"""Fail-closed provenance helpers for pinned H3 upstream tools."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from analysis.h3.common import H3Error, load_json, sha256_file


def _capture(command: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise H3Error(f"command failed rc={proc.returncode}: {' '.join(command)}\n{proc.stdout}")
    return proc.stdout.strip()


def git_checkout_identity(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    root = Path(_capture(["git", "rev-parse", "--show-toplevel"], cwd=path)).resolve()
    head = _capture(["git", "rev-parse", "HEAD"], cwd=root)
    tree = _capture(["git", "rev-parse", "HEAD^{tree}"], cwd=root)
    status = _capture(["git", "status", "--porcelain"], cwd=root)
    origin = _capture(["git", "remote", "get-url", "origin"], cwd=root).rstrip("/")
    expected_origin = str(expected["repository"]).rstrip("/")
    expected_head = str(expected["commit"])
    if head != expected_head:
        raise H3Error(f"tool commit mismatch at {root}: {head} != {expected_head}")
    if status:
        raise H3Error(f"dirty mature-tool checkout: {root}")
    if origin != expected_origin:
        raise H3Error(f"tool origin mismatch at {root}: {origin} != {expected_origin}")
    return {"root": str(root), "head": head, "tree": tree, "origin": origin, "clean": True}


def fio_identity(fio_binary: Path, lock_path: Path) -> dict[str, Any]:
    fio = fio_binary.resolve()
    if not fio.is_file():
        raise H3Error(f"fio binary missing: {fio}")
    lock = load_json(lock_path)
    expected = lock["tools"]["fio"]
    identity = git_checkout_identity(fio.parent, expected)
    version = _capture([str(fio), "--version"])
    return {
        **identity,
        "binary": str(fio),
        "binary_sha256": sha256_file(fio),
        "version": version,
        "lock_sha256": sha256_file(lock_path),
    }
