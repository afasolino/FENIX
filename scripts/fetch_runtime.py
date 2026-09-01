#!/usr/bin/env python3
"""Fetch only the pinned FENIX runtime source; never download model weights."""

from __future__ import annotations

# Support both ``python scripts/<name>.py`` and ``python -m scripts.<name>``.
# Direct script execution places only ``scripts/`` on sys.path, so add the
# repository root before importing sibling FENIX packages.
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from qualification.runtime_lane import (
    ensure_runtime_checkout,
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
        "--repair-incomplete-clone",
        action="store_true",
        help=(
            "repair only the exact empty --no-checkout state produced by the "
            "previous FENIX runtime-fetch bug"
        ),
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    if not (root / ".git").is_dir():
        raise SystemExit("run from the FENIX repository root")

    config = load_lane_config(args.config)
    checkout = ensure_runtime_checkout(
        root,
        config,
        repair_incomplete_clone=args.repair_incomplete_clone,
    )
    print(checkout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
