"""Project-local Podman invocation for FENIX."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PodmanPaths:
    storage: Path
    run: Path
    tmp: Path


def _short_runtime_base(repository_root: Path) -> Path:
    """Return a deterministic short user-owned Podman runtime directory.

    Rootless containers/storage rejects overly long runroot paths. Git linked
    worktrees can easily exceed that limit even when the persistent OCI store
    remains project-local. Keep only ephemeral runroot/tmp data in the system
    temporary directory; include uid + repository hash to avoid cross-worktree
    collisions.
    """

    root = repository_root.resolve()
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:8]
    return Path(tempfile.gettempdir()) / f"fenix-pd-{os.getuid()}-{digest}"


def podman_paths(repository_root: Path) -> PodmanPaths:
    root = repository_root.resolve()
    persistent = root / ".runtime" / "podman"
    transient = _short_runtime_base(root)
    return PodmanPaths(
        storage=(persistent / "storage").resolve(),
        run=(transient / "run").resolve(),
        tmp=(transient / "tmp").resolve(),
    )


def ensure_podman_paths(repository_root: Path) -> PodmanPaths:
    paths = podman_paths(repository_root)
    for path in (paths.storage, paths.run, paths.tmp):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def podman_command(repository_root: Path, *arguments: str) -> list[str]:
    paths = ensure_podman_paths(repository_root)
    return [
        "podman",
        "--root", str(paths.storage),
        "--runroot", str(paths.run),
        "--tmpdir", str(paths.tmp),
        "--storage-driver", "overlay",
        "--storage-opt", "overlay.mount_program=/usr/bin/fuse-overlayfs",
        *arguments,
    ]
