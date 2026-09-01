#!/usr/bin/env python3
"""Acquire the pinned model only after the runtime-lane gate allows it."""

from __future__ import annotations

# Support both ``python scripts/<name>.py`` and ``python -m scripts.<name>``.
# Direct script execution places only ``scripts/`` on sys.path, so add the
# repository root before importing sibling FENIX packages.
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
import subprocess

from qualification.runtime_lane import (
    READY_FOR_MODEL_FETCH,
    load_lane_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime_lane.json"),
    )
    parser.add_argument(
        "--qualification",
        type=Path,
        default=Path("results/raw/runtime_qualification/report.json"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the large model download; otherwise print the gated command",
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    if not (root / ".git").is_dir():
        raise SystemExit("run from the FENIX repository root")

    config = load_lane_config(args.config)

    if not args.qualification.exists():
        raise SystemExit(
            "runtime qualification report is absent; model fetch is blocked"
        )

    report = json.loads(args.qualification.read_text())
    if (
        report.get("status") != READY_FOR_MODEL_FETCH
        or report.get("model_fetch_allowed") is not True
        or report.get("lane_id") != config["lane_id"]
    ):
        raise SystemExit(
            "runtime lane is not READY_FOR_MODEL_FETCH; model fetch is blocked"
        )

    hf = shutil.which("hf")
    if hf is None:
        raise SystemExit(
            "Hugging Face CLI 'hf' is unavailable; no model download was attempted"
        )

    model = config["model"]
    destination = root / model["checkout"]
    command = [
        hf,
        "download",
        model["repository"],
        "--revision",
        model["revision"],
        "--local-dir",
        str(destination),
    ]

    print(" ".join(command))
    if not args.execute:
        print("Dry run only. Re-run with --execute to acquire model weights.")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
