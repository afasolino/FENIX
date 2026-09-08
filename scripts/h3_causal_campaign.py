#!/usr/bin/env python3
"""Rootless H3 causal expert-prefetch qualification and finalization CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.h3.causal_decision import decide_with_causal_prefetch
from analysis.h3.causal_gate import causal_campaign_gate
from analysis.h3.causal_matrix import build_causal_matrix
from analysis.h3.common import H3Error, load_json, set_execution_repository_identity
from analysis.h3.expert_prefetch_causality import build_expert_prefetch_causality
from scripts.h3_campaign import _execution_repository_identity

DEFAULT_CONTRACT = Path("configs/h3/contract_v6.json")
DEFAULT_AMENDMENT = Path("configs/h3/prefetch_causality_amendment_v1.json")
DEFAULT_PROMOTION = Path("configs/h3/causal_promotion_gate_v1.json")


def _discover_prerequisites(measurement_root: Path) -> Path:
    measurement_root = measurement_root.resolve()
    candidates: list[Path] = []
    preferred = list(measurement_root.rglob("*prereq*.json")) + list(
        measurement_root.rglob("*prerequisite*.json")
    )
    seen: set[Path] = set()
    for path in preferred:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            payload = load_json(path)
        except Exception:
            continue
        if payload.get("artifact_kind") == "fenix_h3_prerequisite_manifest":
            candidates.append(path)
    if not candidates:
        for path in measurement_root.rglob("*.json"):
            path = path.resolve()
            if path in seen or not path.is_file():
                continue
            try:
                payload = load_json(path)
            except Exception:
                continue
            if payload.get("artifact_kind") == "fenix_h3_prerequisite_manifest":
                candidates.append(path)
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise H3Error(
            "cannot auto-resolve exactly one frozen prerequisite manifest; "
            f"found={unique}. Supply --prerequisites explicitly."
        )
    return unique[0]


def _finalize(args: argparse.Namespace) -> dict:
    measurement_root = args.measurement_root.resolve()
    out_root = args.out_root.resolve()
    prerequisites = (
        args.prerequisites.resolve()
        if args.prerequisites is not None
        else _discover_prerequisites(measurement_root)
    )
    matrix = build_causal_matrix(
        measurement_root=measurement_root,
        oracle_root=args.oracle_root,
        causality_path=args.causality,
        contract_path=args.contract,
        amendment_path=args.amendment,
        capacity_gib=args.capacity_gib,
        queue_depth=args.qd,
        out_root=out_root,
    )
    matrix_path = out_root / "causal-matrix.json"
    gate_path = out_root / "campaign-gate.json"
    gate = causal_campaign_gate(
        matrix_path=matrix_path,
        prerequisites_path=prerequisites,
        causality_path=args.causality,
        contract_path=args.contract,
        amendment_path=args.amendment,
        promotion_contract_path=args.promotion_contract,
        out_path=gate_path,
    )
    return {
        "artifact_kind": "fenix_h3_causal_finalization",
        "matrix_manifest": str(matrix_path),
        "matrix_decision_count": matrix["decision_count"],
        "prerequisites": str(prerequisites),
        "campaign_gate": str(gate_path),
        "preliminary_memory_service_verdict": gate["preliminary_memory_service_verdict"],
        "paper_promotion_verdict": gate["paper_promotion_verdict"],
        "proceed_to_h4": gate["proceed_to_h4"],
        "combined_directional_bounds": gate["combined_directional_bounds"],
        "limiting_matrix_row": gate["limiting_matrix_row"],
    }


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

    p = sub.add_parser("matrix")
    p.add_argument("--measurement-root", type=Path, required=True)
    p.add_argument("--oracle-root", type=Path, required=True)
    p.add_argument("--causality", type=Path, required=True)
    p.add_argument("--capacity-gib", type=float, default=7.0)
    p.add_argument("--qd", type=int, default=32)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    p.add_argument("--out-root", type=Path, required=True)
    p.set_defaults(func=lambda a: build_causal_matrix(
        a.measurement_root,
        a.oracle_root,
        a.causality,
        a.contract,
        a.amendment,
        a.capacity_gib,
        a.qd,
        a.out_root,
    ))

    p = sub.add_parser("gate")
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--prerequisites", type=Path, required=True)
    p.add_argument("--causality", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    p.add_argument("--promotion-contract", type=Path, default=DEFAULT_PROMOTION)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: causal_campaign_gate(
        a.matrix,
        a.prerequisites,
        a.causality,
        a.contract,
        a.amendment,
        a.promotion_contract,
        a.out,
    ))

    p = sub.add_parser("finalize")
    p.add_argument("--measurement-root", type=Path, required=True)
    p.add_argument("--oracle-root", type=Path, required=True)
    p.add_argument("--causality", type=Path, required=True)
    p.add_argument("--capacity-gib", type=float, default=7.0)
    p.add_argument("--qd", type=int, default=32)
    p.add_argument("--prerequisites", type=Path)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    p.add_argument("--promotion-contract", type=Path, default=DEFAULT_PROMOTION)
    p.add_argument("--out-root", type=Path, required=True)
    p.set_defaults(func=_finalize)
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
    finally:
        set_execution_repository_identity(None)


if __name__ == "__main__":
    raise SystemExit(main())
