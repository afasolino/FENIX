"""Run the local FENIX repository quality and reproducibility gates."""

from __future__ import annotations

import argparse
import compileall
import json
import subprocess
import sys
from pathlib import Path


MAX_TRACKABLE_FILE_BYTES = 10 * 1024 * 1024
IGNORED_TOP_LEVEL = {".git", ".venv", "external"}


def repository_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in IGNORED_TOP_LEVEL for part in path.relative_to(root).parts)
    ]


def validate_json(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if path.suffix != ".json":
            continue
        try:
            json.loads(path.read_text())
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return errors


def validate_methodology(root: Path) -> list[str]:
    errors: list[str] = []

    required = (
        root / "docs/decisions/0001-literature-grounded-local-falsification.md",
        root / "docs/evidence_policy.md",
        root / "docs/decisions/0002-runtime-lane-qualification.md",
        root / "configs/runtime_lane.json",
        root / "configs/campaign.json",
        root / "repro.lock.json",
        root / "requirements-analysis.lock.txt",
        root / "scripts/__init__.py",
    )
    for path in required:
        if not path.exists():
            errors.append(f"missing required repository artifact: {path}")

    campaign_path = root / "configs/campaign.json"
    if campaign_path.exists():
        campaign = json.loads(campaign_path.read_text())
        if campaign.get("schema_version") != 2:
            errors.append("campaign schema_version must be 2")
        if "host_dram_sweep" in campaign:
            errors.append("legacy host_dram_sweep configuration is still present")
        if "capacity_tradeoff" not in campaign.get("experiments", {}):
            errors.append("capacity_tradeoff experiment is missing")

    verdict_path = root / "results/processed/initial_verdict.json"
    if verdict_path.exists():
        verdict = json.loads(verdict_path.read_text())
        if verdict.get("verdict") != "INCONCLUSIVE":
            errors.append("initial verdict must remain INCONCLUSIVE")
        if verdict.get("proceed_to_hardware_architecture") is not False:
            errors.append("initial hardware-architecture gate must be closed")

    legacy_fetch = root / "scripts/fetch_assets.sh"
    if legacy_fetch.exists():
        errors.append(
            "legacy scripts/fetch_assets.sh couples runtime and model acquisition"
        )

    for required_script in (
        "scripts/fetch_runtime.py",
        "scripts/qualify_runtime.py",
        "scripts/fetch_model.py",
    ):
        if not (root / required_script).exists():
            errors.append(f"missing runtime-gate script: {required_script}")

    return errors


def validate_file_sizes(root: Path, files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        relative = path.relative_to(root)
        if relative.parts[:2] in {
            ("results", "raw"),
            ("traces", "raw"),
        }:
            continue
        if path.stat().st_size > MAX_TRACKABLE_FILE_BYTES:
            errors.append(
                f"unexpected large repository file: {relative} "
                f"({path.stat().st_size} bytes)"
            )
    return errors


def run_tests(root: Path) -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        check=False,
    )
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    if not (root / ".git").is_dir():
        print("ERROR: run from the FENIX repository root", file=sys.stderr)
        return 2

    files = repository_files(root)
    errors = []
    errors.extend(validate_json(files))
    errors.extend(validate_methodology(root))
    errors.extend(validate_file_sizes(root, files))

    compile_ok = compileall.compile_dir(
        str(root / "analysis"),
        quiet=1,
        force=True,
    )
    compile_ok &= compileall.compile_dir(
        str(root / "instrumentation"),
        quiet=1,
        force=True,
    )
    compile_ok &= compileall.compile_dir(
        str(root / "scripts"),
        quiet=1,
        force=True,
    )
    if not compile_ok:
        errors.append("Python compilation failed")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if not args.skip_tests:
        test_returncode = run_tests(root)
        if test_returncode != 0:
            return test_returncode

    print("FENIX repository checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
