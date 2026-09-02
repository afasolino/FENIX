#!/usr/bin/env python3
"""Prepare and build a FENIX Qwen3.8 runtime image safely.

The default target is a disposable candidate image. The already-qualified
``fenix-qwen38:locked`` tag is protected and can only be targeted with the
explicit ``--allow-qualified-image-overwrite`` escape hatch.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_IMAGE = "fenix-qwen38:candidate"
QUALIFIED_IMAGE = "fenix-qwen38:locked"
DEFAULT_OUTPUT = Path(".runtime/instrumented/qwen38-candidate")
PINNED_RUNTIME = Path("external/runtime/qwen38")


class RuntimeBuildError(RuntimeError):
    """Raised when a runtime build violates FENIX build policy."""


@dataclass(frozen=True)
class BuildPlan:
    root: Path
    output: Path
    image: str
    prepare_command: tuple[str, ...]
    build_command: tuple[str, ...]


def repository_root() -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeBuildError("not inside a Git repository")
    return Path(completed.stdout.strip()).resolve()


def _ensure_within_root(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeBuildError(
            f"{label} must remain inside the FENIX repository: {resolved}"
        ) from exc
    return resolved


def validate_image_target(
    image: str,
    *,
    allow_qualified_image_overwrite: bool,
) -> str:
    normalized = image.strip()
    if not normalized:
        raise RuntimeBuildError("runtime image tag must not be empty")
    if any(character.isspace() for character in normalized):
        raise RuntimeBuildError(
            f"runtime image tag must not contain whitespace: {image!r}"
        )

    if (
        normalized == QUALIFIED_IMAGE
        and not allow_qualified_image_overwrite
    ):
        raise RuntimeBuildError(
            f"refusing to overwrite qualified image {QUALIFIED_IMAGE!r}; "
            "pass --allow-qualified-image-overwrite only when intentionally "
            "invalidating and requalifying that evidence"
        )

    return normalized


def build_plan(
    *,
    root: Path,
    python: Path,
    image: str,
    output: Path,
    allow_qualified_image_overwrite: bool = False,
) -> BuildPlan:
    root = root.resolve()
    python = python.resolve()
    image = validate_image_target(
        image,
        allow_qualified_image_overwrite=allow_qualified_image_overwrite,
    )
    output = _ensure_within_root(output, root, "instrumented runtime output")
    runtime = _ensure_within_root(
        root / PINNED_RUNTIME,
        root,
        "pinned runtime checkout",
    )

    dockerfile = output / "docker/Dockerfile"
    prepare_command = (
        str(python),
        "instrumentation/prepare_runtime.py",
        "--runtime",
        str(runtime.relative_to(root)),
        "--output",
        str(output.relative_to(root)),
    )
    build_command = (
        str(python),
        "-m",
        "scripts.fenix_podman",
        "build",
        "-t",
        image,
        "-f",
        str(dockerfile.relative_to(root)),
        str(output.relative_to(root)),
    )
    return BuildPlan(
        root=root,
        output=output,
        image=image,
        prepare_command=prepare_command,
        build_command=build_command,
    )


def _run(command: Sequence[str], *, cwd: Path) -> int:
    completed = subprocess.run(command, cwd=cwd)
    return completed.returncode


def execute_plan(plan: BuildPlan) -> int:
    prepare_rc = _run(plan.prepare_command, cwd=plan.root)
    if prepare_rc != 0:
        print(
            "Transactional runtime preparation failed; image build skipped.",
            file=sys.stderr,
        )
        return prepare_rc

    return _run(plan.build_command, cwd=plan.root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"image tag to build (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"project-local staging directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--allow-qualified-image-overwrite",
        action="store_true",
        help=(
            "allow targeting fenix-qwen38:locked; this invalidates the "
            "assumption that the existing qualified image is preserved"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the prepare/build commands without executing them",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        root = repository_root()
        python = root / ".venv/bin/python"
        if not python.is_file():
            raise RuntimeBuildError(
                f"expected FENIX interpreter is missing: {python}"
            )

        output = args.output
        if not output.is_absolute():
            output = root / output

        plan = build_plan(
            root=root,
            python=python,
            image=args.image,
            output=output,
            allow_qualified_image_overwrite=(
                args.allow_qualified_image_overwrite
            ),
        )
    except RuntimeBuildError as exc:
        print(f"runtime build policy error: {exc}", file=sys.stderr)
        return 2

    print("runtime image:", plan.image)
    print("instrumented output:", plan.output)
    print("prepare:", shlex.join(plan.prepare_command))
    print("build:", shlex.join(plan.build_command))

    if args.dry_run:
        return 0
    return execute_plan(plan)


if __name__ == "__main__":
    raise SystemExit(main())
