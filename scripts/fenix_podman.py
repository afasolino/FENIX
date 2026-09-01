#!/usr/bin/env python3
"""Invoke Podman with the FENIX project-local OCI store."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualification.podman_runtime import podman_command


def main() -> int:
    root = Path.cwd().resolve()
    if not (root / ".git").is_dir():
        raise SystemExit("run from the FENIX repository root")
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m scripts.fenix_podman <podman arguments>")
    return subprocess.call(podman_command(root, *sys.argv[1:]), cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
