"""Project-local Podman invocation for FENIX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PodmanPaths:
    storage: Path
    run: Path
    tmp: Path


def podman_paths(repository_root: Path) -> PodmanPaths:
    base = repository_root / ".runtime" / "podman"
    return PodmanPaths(
        storage=(base / "storage").resolve(),
        run=(base / "run").resolve(),
        tmp=(base / "tmp").resolve(),
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
