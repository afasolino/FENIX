#!/usr/bin/env python3
"""Run the source-only FENIX runtime-lane qualification gate."""

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

from qualification.runtime_lane import load_lane_config, qualify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime_lane.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/raw/runtime_qualification/report.json"),
    )
    args = parser.parse_args()

    root = Path.cwd().resolve()
    if not (root / ".git").is_dir():
        raise SystemExit("run from the FENIX repository root")

    config = load_lane_config(args.config)
    report = qualify(root, config)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    return 0 if report["model_fetch_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
