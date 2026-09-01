"""Pinned runtime-lane acquisition and source qualification."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_INCOMPATIBLE = "SOURCE_INCOMPATIBLE"
ENVIRONMENT_BLOCKED = "ENVIRONMENT_BLOCKED"
READY_FOR_MODEL_FETCH = "READY_FOR_MODEL_FETCH"


@dataclass(frozen=True)
class CommandResult:
    available: bool
    returncode: int | None
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.available and self.returncode == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "ok": self.ok,
        }


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: int = 60,
) -> CommandResult:
    """Run a qualification command without raising on ordinary command failure."""

    executable = shutil.which(command[0])
    if executable is None:
        return CommandResult(False, None, "", f"{command[0]} not found")

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            True,
            None,
            exc.stdout or "",
            f"command timed out after {timeout_seconds}s",
        )

    return CommandResult(
        True,
        completed.returncode,
        completed.stdout.strip(),
        completed.stderr.strip(),
    )


def load_lane_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the runtime-lane configuration."""

    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported runtime-lane schema")
    if not config.get("runtime", {}).get("revision"):
        raise ValueError("runtime revision is missing")
    if not config.get("model", {}).get("revision"):
        raise ValueError("model revision is missing")
    return config


def _git_status(checkout: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _non_git_worktree_entries(checkout: Path) -> list[Path]:
    return [
        path
        for path in checkout.iterdir()
        if path.name != ".git"
    ]


def is_incomplete_no_checkout_clone(checkout: Path) -> bool:
    """Recognize only the empty worktree created by the old fetch bug.

    ``git clone --no-checkout`` creates a repository whose HEAD points at the
    remote default branch while no files have been materialized. Git reports
    every tracked path as a staged deletion. This predicate deliberately
    rejects any state containing non-deletion changes or any non-.git worktree
    entries so that ordinary user edits can never be mistaken for the known
    initialization artifact.
    """

    if not (checkout / ".git").is_dir():
        return False
    if _non_git_worktree_entries(checkout):
        return False

    status = _git_status(checkout)
    return bool(status) and all(line.startswith("D  ") for line in status)


def repair_incomplete_no_checkout_clone(checkout: Path) -> None:
    """Clear the exact empty-worktree state produced by the old fetch bug."""

    if not is_incomplete_no_checkout_clone(checkout):
        raise RuntimeError(
            "runtime checkout is not the recognized incomplete no-checkout "
            "state; refusing automatic repair"
        )

    subprocess.run(
        ["git", "reset", "--hard", "HEAD"],
        cwd=checkout,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if _git_status(checkout):
        raise RuntimeError("runtime checkout remained dirty after repair")


def ensure_runtime_checkout(
    repository_root: Path,
    config: dict[str, Any],
    *,
    repair_incomplete_clone: bool = False,
) -> Path:
    """Fetch the pinned runtime into the ignored external tree.

    New clones are initialized before cleanliness is assessed. Existing
    third-party checkouts remain fail-closed: local modifications are never
    reset. The only automatic recovery path is opt-in and recognizes the exact
    empty ``--no-checkout`` state created by the previous FENIX fetch bug.
    """

    runtime = config["runtime"]
    checkout = repository_root / runtime["checkout"]
    revision = str(runtime["revision"])
    repository = str(runtime["repository"])
    created = False

    if checkout.exists() and not (checkout / ".git").is_dir():
        raise RuntimeError(
            f"runtime checkout exists but is not a Git repository: {checkout}"
        )

    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", repository, str(checkout)],
            cwd=repository_root,
            check=True,
        )
        created = True

    if not created:
        status = _git_status(checkout)
        if status:
            if repair_incomplete_clone and is_incomplete_no_checkout_clone(checkout):
                repair_incomplete_no_checkout_clone(checkout)
            else:
                raise RuntimeError(
                    "third-party runtime checkout contains local modifications; "
                    "refusing to alter it"
                )

    subprocess.run(
        ["git", "fetch", "--prune", "origin"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", revision],
        cwd=checkout,
        check=True,
    )

    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    if actual != revision:
        raise RuntimeError(
            f"runtime revision mismatch: expected {revision}, got {actual}"
        )

    final_status = _git_status(checkout)
    if final_status:
        raise RuntimeError(
            "runtime checkout is dirty after pinned revision checkout"
        )

    return checkout


def inspect_runtime_source(
    checkout: Path,
    config: dict[str, Any],
) -> dict[str, object]:
    """Verify the exact source structures needed by the FENIX experiment."""

    expected_revision = str(config["runtime"]["revision"])
    head = run_command(["git", "rev-parse", "HEAD"], cwd=checkout)

    checks: list[dict[str, object]] = []
    failures: list[str] = []

    if not head.ok or head.stdout != expected_revision:
        failures.append(
            "runtime checkout is not at the configured revision"
        )

    for specification in config["source_checks"]:
        relative = Path(specification["path"])
        path = checkout / relative
        missing_markers: list[str] = []

        if path.is_file():
            content = path.read_text(errors="replace")
            for marker in specification.get("markers", []):
                if marker not in content:
                    missing_markers.append(marker)
        else:
            missing_markers.extend(specification.get("markers", []))

        passed = path.is_file() and not missing_markers
        if not passed:
            failures.append(str(specification["id"]))

        checks.append(
            {
                "id": specification["id"],
                "path": str(relative),
                "exists": path.is_file(),
                "missing_markers": missing_markers,
                "passed": passed,
            }
        )

    return {
        "revision_expected": expected_revision,
        "revision_actual": head.stdout if head.ok else None,
        "checks": checks,
        "passed": not failures,
        "failures": failures,
    }


def _parse_gpu_rows(result: CommandResult) -> list[dict[str, object]]:
    if not result.ok:
        return []

    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        index, name, compute_capability, memory_total_mib, driver = fields
        try:
            rows.append(
                {
                    "index": int(index),
                    "name": name,
                    "compute_capability": float(compute_capability),
                    "memory_total_mib": float(memory_total_mib),
                    "driver_version": driver,
                }
            )
        except ValueError:
            continue
    return rows


def inspect_host_environment(
    config: dict[str, Any],
) -> dict[str, object]:
    """Assess only prerequisites for the configured execution lane."""

    target = config["target"]
    execution = config["execution"]

    gpu_result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = _parse_gpu_rows(gpu_result)

    host_failures: list[str] = []

    if platform.system() != target["operating_system"]:
        host_failures.append(
            f"operating_system={platform.system()}"
        )
    if platform.machine() != target["machine"]:
        host_failures.append(f"machine={platform.machine()}")

    if len(gpus) != int(target["gpu_count"]):
        host_failures.append(f"gpu_count={len(gpus)}")
    elif gpus:
        gpu = gpus[0]
        if target["gpu_name_contains"] not in str(gpu["name"]):
            host_failures.append(f"gpu_name={gpu['name']}")
        if float(gpu["compute_capability"]) < float(
            target["minimum_compute_capability"]
        ):
            host_failures.append(
                f"compute_capability={gpu['compute_capability']}"
            )

    docker = run_command(["docker", "--context", "default", "info"])
    if execution["backend"] == "docker" and not docker.ok:
        host_failures.append("docker_default_context_unavailable")

    nvidia_container_cli = run_command(
        ["nvidia-container-cli", "--version"]
    )
    if (
        execution.get("require_nvidia_container_cli")
        and not nvidia_container_cli.ok
    ):
        host_failures.append("nvidia_container_cli_unavailable")

    return {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "gpus": gpus,
        "gpu_query": gpu_result.as_dict(),
        "docker_default_info": docker.as_dict(),
        "nvidia_container_cli": nvidia_container_cli.as_dict(),
        "passed": not host_failures,
        "failures": host_failures,
    }


def qualify(
    repository_root: Path,
    config: dict[str, Any],
) -> dict[str, object]:
    """Return a source-only qualification report for the configured lane."""

    checkout = repository_root / config["runtime"]["checkout"]
    source = inspect_runtime_source(checkout, config)
    host = inspect_host_environment(config)

    if not source["passed"]:
        status = SOURCE_INCOMPATIBLE
    elif not host["passed"]:
        status = ENVIRONMENT_BLOCKED
    else:
        status = READY_FOR_MODEL_FETCH

    return {
        "schema_version": 1,
        "qualification_kind": "source_and_environment",
        "runtime_qualified": False,
        "lane_id": config["lane_id"],
        "campaign_base_commit": config["campaign_base_commit"],
        "status": status,
        "model_fetch_allowed": status == READY_FOR_MODEL_FETCH,
        "source": source,
        "host": host,
        "tp1_policy": {
            "tensor_parallel_size": 1,
            "distributed_executor_backend": config["execution"][
                "tp1_distributed_executor_backend"
            ],
            "reason": config["execution"]["tp1_reason"],
            "does_not_establish_runtime_qualification": True,
        },
        "known_upstream_state": config["known_upstream_state"],
    }
