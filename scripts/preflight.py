"""Capture normalized FENIX host, accelerator, storage, and toolchain metadata."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def run_command(
    command: list[str],
    *,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Run a diagnostic command without making its failure fatal."""

    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return {"command": command, "available": False}
    except Exception as exc:
        return {
            "command": command,
            "available": True,
            "error": repr(exc),
        }

    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return None


def parse_meminfo(path: Path) -> dict[str, int]:
    """Return /proc-style memory values in bytes."""

    result: dict[str, int] = {}
    text = read_text(path)
    if text is None:
        return result

    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        fields = raw_value.strip().split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        multiplier = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
        result[key] = value * multiplier

    return result


def numa_nodes() -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    root = Path("/sys/devices/system/node")

    for node_path in sorted(root.glob("node[0-9]*")):
        nodes.append(
            {
                "node": int(node_path.name.removeprefix("node")),
                "cpulist": read_text(node_path / "cpulist"),
                "memory_bytes": parse_meminfo(node_path / "meminfo"),
            }
        )

    return nodes


def git_metadata(repository: Path) -> dict[str, Any]:
    commands = {
        "status": ["git", "-C", str(repository), "status", "--porcelain=v1", "-b"],
        "head": ["git", "-C", str(repository), "rev-parse", "HEAD"],
        "origin": ["git", "-C", str(repository), "remote", "get-url", "origin"],
    }
    return {name: run_command(command) for name, command in commands.items()}


def collect(repository: Path) -> dict[str, Any]:
    """Collect the preflight record."""

    gpu_query = (
        "index,name,uuid,driver_version,memory.total,memory.free,compute_cap,"
        "pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,"
        "pcie.link.width.current,pcie.link.width.max"
    )

    diagnostics = {
        "nvidia_smi": run_command(
            [
                "nvidia-smi",
                f"--query-gpu={gpu_query}",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvcc": run_command(["nvcc", "--version"]),
        "lscpu": run_command(["lscpu", "-J"]),
        "numactl": run_command(["numactl", "--hardware"]),
        "block_devices": run_command(["lsblk", "-b", "-O", "-J"]),
        "repository_mount": run_command(
            ["findmnt", "-J", "-T", str(repository)]
        ),
        "docker_context": run_command(["docker", "context", "show"]),
        "docker_default_info": run_command(
            ["docker", "--context", "default", "info"]
        ),
        "nvidia_container_cli": run_command(
            ["nvidia-container-cli", "--version"]
        ),
        "pip_freeze": run_command([sys.executable, "-m", "pip", "freeze"]),
    }

    return {
        "schema_version": 2,
        "captured_utc": datetime.now(UTC).isoformat(),
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "repository": str(repository),
        "git": git_metadata(repository),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "uname": list(platform.uname()),
            "logical_cpu_count": os.cpu_count(),
            "memory_bytes": parse_meminfo(Path("/proc/meminfo")),
            "numa_nodes": numa_nodes(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
        },
        "environment": {
            "path": os.environ.get("PATH"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repository = Path.cwd().resolve()
    if not (repository / ".git").is_dir():
        raise SystemExit("preflight must be run from the FENIX repository root")

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "environment.json"
    destination.write_text(json.dumps(collect(repository), indent=2))
    print(destination)


if __name__ == "__main__":
    main()
