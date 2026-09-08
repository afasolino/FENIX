#!/usr/bin/env python3
"""Rootless H3 page-cache-oracle analysis amendment CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.h3.common import H3Error, set_execution_repository_identity
from analysis.h3.oracle_gate import campaign_gate_with_oracle, decide_with_oracle
from analysis.h3.pagecache_oracle import build_pagecache_oracle
from scripts.h3_campaign import _execution_repository_identity

DEFAULT_CONTRACT = Path("configs/h3/contract_v6.json")
DEFAULT_AMENDMENT = Path("configs/h3/pagecache_oracle_amendment_v1.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pagecache-oracle")
    p.add_argument("--window", type=Path, required=True)
    p.add_argument("--capacity-gib", type=float, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: build_pagecache_oracle(
        a.window, a.contract, a.amendment, a.capacity_gib, a.out
    ))

    p = sub.add_parser("decide")
    p.add_argument("--residency", type=Path, required=True)
    p.add_argument("--lpddr", type=Path, required=True)
    p.add_argument("--fio", type=Path, required=True)
    p.add_argument("--actual-lpddr", type=Path, required=True)
    p.add_argument("--pagecache", type=Path, action="append", default=[])
    p.add_argument("--pagecache-oracle", type=Path)
    p.add_argument("--qd", type=int, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: decide_with_oracle(
        a.residency, a.lpddr, a.fio, a.contract, a.qd, a.out, a.amendment,
        a.actual_lpddr, a.pagecache_oracle, a.pagecache,
    ))

    p = sub.add_parser("campaign-gate")
    p.add_argument("--decision", type=Path, action="append", required=True)
    p.add_argument("--prerequisites", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: campaign_gate_with_oracle(
        a.decision, a.prerequisites, a.contract, a.amendment, a.out
    ))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        set_execution_repository_identity(_execution_repository_identity())
        result = args.func(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except H3Error as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
