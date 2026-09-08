#!/usr/bin/env python3
"""Build the amended H3 resident-vs-mmap placement semantic control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.h3.common import H3Error, set_execution_repository_identity
from analysis.h3.placement_invariance_control import build_placement_invariance_control
from scripts.h3_campaign import _execution_repository_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path("configs/h3/contract_v6.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        set_execution_repository_identity(_execution_repository_identity())
        result = build_placement_invariance_control(args.input, args.contract, args.out)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except H3Error as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 3
    finally:
        set_execution_repository_identity(None)


if __name__ == "__main__":
    raise SystemExit(main())
