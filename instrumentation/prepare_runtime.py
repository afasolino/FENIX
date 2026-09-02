#!/usr/bin/env python3
"""Create a transactionally instrumented copy of the pinned external runtime.

The pinned third-party checkout is never edited. Instrumentation is applied to
a staging copy under FENIX .runtime/, then the overlay checksum manifest is
regenerated from the final staged bytes before the copy is promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PIN = "7b5f0465db90fc49d6324904f48ad995ebdcb62f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regenerate_overlay_manifest(runtime_root: Path) -> dict[str, object]:
    """Regenerate the manifest using the exact install_overlay.py semantics."""

    overlay = runtime_root / "runtime/vllm-overlay"
    manifest_path = overlay / "SHA256SUMS.json"
    if not overlay.is_dir():
        raise RuntimeError(f"overlay directory is missing: {overlay}")

    observed = {
        str(path.relative_to(overlay)): sha256(path)
        for path in overlay.rglob("*")
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(
        json.dumps(observed, indent=2, sort_keys=True) + "\n"
    )
    return {
        "path": str(manifest_path.relative_to(runtime_root)),
        "file_count": len(observed),
        "sha256": sha256(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", default=".runtime/instrumented/qwen38")
    args = parser.parse_args()

    source = Path(args.runtime).resolve()
    output = Path(args.output).resolve()

    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != PIN:
        raise SystemExit(f"runtime HEAD {head} != {PIN}")

    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1"],
        text=True,
    ).strip()
    if status:
        raise SystemExit("pinned third-party runtime checkout is dirty")

    staging = output.parent / f"{output.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    output.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(source, staging, ignore=shutil.ignore_patterns(".git"))

    helper = Path(__file__).with_name("prepare_runtime_inplace.py")
    if not helper.is_file():
        shutil.rmtree(staging)
        raise SystemExit(f"missing instrumentation helper: {helper}")

    completed = subprocess.run(
        [
            str(Path(__file__).resolve().parents[1] / ".venv/bin/python"),
            str(helper),
            "--runtime",
            str(staging),
            "--skip-git-pin-check",
            "--source-pin",
            head,
        ],
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        shutil.rmtree(staging)
        return completed.returncode

    hardener_source = Path(__file__).with_name(
        "harden_runtime_image.py"
    )
    if not hardener_source.is_file():
        shutil.rmtree(staging)
        raise SystemExit(
            f"missing runtime hardener: {hardener_source}"
        )

    hardener_target = staging / "runtime/fenix_harden_runtime_image.py"
    shutil.copy2(hardener_source, hardener_target)

    dockerfile = staging / "docker/Dockerfile"
    docker_text = dockerfile.read_text()

    docker_anchor = (
        "RUN python3 /opt/qwen38/runtime/install_overlay.py "
        "/opt/qwen38/runtime/vllm-overlay\n"
    )
    docker_replacement = (
        docker_anchor
        + "COPY runtime/fenix_harden_runtime_image.py "
        "/opt/qwen38/runtime/fenix_harden_runtime_image.py\n"
        + "RUN python3 "
        "/opt/qwen38/runtime/fenix_harden_runtime_image.py\n"
    )

    if docker_text.count(docker_anchor) != 1:
        shutil.rmtree(staging)
        raise SystemExit(
            "candidate Dockerfile overlay-install anchor mismatch"
        )

    dockerfile.write_text(
        docker_text.replace(
            docker_anchor,
            docker_replacement,
            1,
        )
    )

    instrumentation_manifest = staging / "fenix-instrumentation-manifest.json"
    if not instrumentation_manifest.is_file():
        shutil.rmtree(staging)
        raise SystemExit("instrumentation manifest missing from staging tree")

    overlay_manifest = regenerate_overlay_manifest(staging)

    provenance = json.loads(instrumentation_manifest.read_text())
    provenance["overlay_manifest"] = overlay_manifest
    provenance["source_checkout_modified"] = False
    instrumentation_manifest.write_text(json.dumps(provenance, indent=2) + "\n")

    if output.exists():
        shutil.rmtree(output)
    staging.rename(output)

    print(
        json.dumps(
            {
                "runtime_head": head,
                "output": str(output),
                "overlay_manifest": overlay_manifest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
