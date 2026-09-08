#!/usr/bin/env python3
"""Invoke Podman with the FENIX project-local OCI store."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualification.podman_runtime import podman_command


def repository_root_from_cwd(cwd: Path) -> Path:
    """Return the Git worktree root, rejecting non-repositories and subdirs."""

    cwd = cwd.resolve()
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("run from the FENIX repository root")
    root = Path(completed.stdout.strip()).resolve()
    if root != cwd:
        raise RuntimeError("run from the FENIX repository root")
    return root


def main() -> int:
    try:
        root = repository_root_from_cwd(Path.cwd())
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m scripts.fenix_podman <podman arguments>")
    return subprocess.call(podman_command(root, *sys.argv[1:]), cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
