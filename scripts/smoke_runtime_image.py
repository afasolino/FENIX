#!/usr/bin/env python3
"""Smoke-test the built FENIX runtime image on the target GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualification.podman_runtime import podman_command
from qualification.runtime_lane import load_lane_config, run_command


SMOKE_PREFIX = "FENIX_SMOKE_JSON="


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/runtime_lane.json"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/raw/runtime_qualification/image_smoke.json"),
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    if not (root / ".git").is_dir():
        raise SystemExit("run from the FENIX repository root")

    config = load_lane_config(args.config)
    execution = config["execution"]
    runtime = config["runtime"]
    image = str(execution["runtime_image"])

    base = {
        "schema_version": 1,
        "qualification_kind": "built_runtime_image_smoke",
        "lane_id": config["lane_id"],
        "runtime_source_revision": runtime["revision"],
        "base_container_image": runtime["container_image"],
        "image": image,
        "passed": False,
    }

    exists = run_command(podman_command(root, "image", "exists", image))
    if not exists.ok:
        report = {**base, "failure": "built_runtime_image_missing", "image_exists": exists.as_dict()}
        _write_report(args.out, report)
        print(json.dumps(report, indent=2))
        return 2

    inspect = run_command(
        podman_command(root, "image", "inspect", "--format", "{{.Id}}", image)
    )

    code = """
import importlib.metadata
import json
import torch

result = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "vllm": importlib.metadata.version("vllm"),
    "humming_kernels": importlib.metadata.version("humming-kernels"),
    "cuda_available": torch.cuda.is_available(),
    "ple_worker_import": False,
}
try:
    from vllm.v1.ple_offload.worker import PleOffloadWorker
    result["ple_worker_import"] = PleOffloadWorker is not None
except Exception as exc:
    result["ple_worker_import_error"] = repr(exc)

if torch.cuda.is_available():
    result["device"] = torch.cuda.get_device_name(0)
    result["compute_capability"] = list(torch.cuda.get_device_capability(0))
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    result["sum"] = x.sum().item()

print("FENIX_SMOKE_JSON=" + json.dumps(result, sort_keys=True))
"""

    command = podman_command(
        root,
        "run",
        "--rm",
        "--security-opt=label=disable",
        "--device", str(execution["cdi_device"]),
        "--entrypoint", "python3",
        image,
        "-c", code,
    )
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith(SMOKE_PREFIX):
            payload = json.loads(line[len(SMOKE_PREFIX):])

    failures: list[str] = []
    if completed.returncode != 0:
        failures.append("container_execution_failed")
    if payload is None:
        failures.append("smoke_payload_missing")
    else:
        if payload.get("cuda_available") is not True:
            failures.append("cuda_unavailable")
        if config["target"]["gpu_name_contains"] not in str(payload.get("device", "")):
            failures.append("unexpected_gpu")
        capability = payload.get("compute_capability")
        if not (
            isinstance(capability, list)
            and len(capability) == 2
            and float(f"{capability[0]}.{capability[1]}")
            >= float(config["target"]["minimum_compute_capability"])
        ):
            failures.append("compute_capability_below_target")
        if payload.get("sum") != 523776.0:
            failures.append("cuda_calculation_mismatch")
        if payload.get("vllm") != runtime["reported_vllm_version"]:
            failures.append("vllm_version_mismatch")
        if payload.get("humming_kernels") != runtime["humming_kernels_version"]:
            failures.append("humming_kernels_version_mismatch")
        if payload.get("ple_worker_import") is not True:
            failures.append("ple_worker_import_failed")

    report = {
        **base,
        "passed": not failures,
        "image_id": inspect.stdout if inspect.ok else None,
        "container_returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "observed": payload,
        "failures": failures,
    }
    _write_report(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
