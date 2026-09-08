#!/usr/bin/env python3
"""Run the causal H3 gate with the prerequisite-provenance amendment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.h3.causal_prerequisite_provenance import (
    causal_campaign_gate_with_prerequisite_provenance,
)
from analysis.h3.common import H3Error, set_execution_repository_identity
from scripts.h3_campaign import _execution_repository_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--prerequisites", type=Path, required=True)
    parser.add_argument("--causality", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path("configs/h3/contract_v6.json"))
    parser.add_argument("--causality-amendment", type=Path, default=Path("configs/h3/prefetch_causality_amendment_v1.json"))
    parser.add_argument("--promotion-contract", type=Path, default=Path("configs/h3/causal_promotion_gate_v1.json"))
    parser.add_argument("--prerequisite-amendment", type=Path, default=Path("configs/h3/prerequisite_provenance_amendment_v1.json"))
    parser.add_argument("--base-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        set_execution_repository_identity(_execution_repository_identity())
        result = causal_campaign_gate_with_prerequisite_provenance(
            matrix_path=args.matrix,
            prerequisites_path=args.prerequisites,
            causality_path=args.causality,
            contract_path=args.contract,
            causality_amendment_path=args.causality_amendment,
            promotion_contract_path=args.promotion_contract,
            prerequisite_amendment_path=args.prerequisite_amendment,
            base_out_path=args.base_out,
            out_path=args.out,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except H3Error as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 3
    finally:
        set_execution_repository_identity(None)


if __name__ == "__main__":
    raise SystemExit(main())
