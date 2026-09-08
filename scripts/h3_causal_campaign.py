#!/usr/bin/env python3
"""Rootless H3 causal expert-prefetch qualification and re-decision CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.h3.causal_decision import decide_with_causal_prefetch
from analysis.h3.common import H3Error, set_execution_repository_identity
from analysis.h3.expert_prefetch_causality import build_expert_prefetch_causality
from scripts.h3_campaign import _execution_repository_identity

DEFAULT_CONTRACT = Path("configs/h3/contract_v6.json")
DEFAULT_AMENDMENT = Path("configs/h3/prefetch_causality_amendment_v1.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze-causality")
    p.add_argument("--moe", type=Path, action="append", required=True)
    p.add_argument("--fio-root", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: build_expert_prefetch_causality(
        a.moe, a.fio_root, a.contract, a.amendment, a.out
    ))

    p = sub.add_parser("decide")
    p.add_argument("--residency", type=Path, required=True)
    p.add_argument("--lpddr", type=Path, required=True)
    p.add_argument("--fio", type=Path, required=True)
    p.add_argument("--actual-lpddr", type=Path, required=True)
    p.add_argument("--causality", type=Path, required=True)
    p.add_argument("--pagecache-oracle", type=Path)
    p.add_argument("--qd", type=int, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: decide_with_causal_prefetch(
        a.residency,
        a.lpddr,
        a.fio,
        a.actual_lpddr,
        a.causality,
        a.contract,
        a.amendment,
        a.qd,
        a.out,
        a.pagecache_oracle,
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
