"""Verify runtime PLE row IDs against the independent reference algorithm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.ple_address import rows_for_history


def verify_trace(trace_path: Path) -> dict[str, object]:
    grouped: dict[tuple[object, object, object], dict[str, object]] = {}

    for line_number, line in enumerate(trace_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue

        record = json.loads(line)
        history = record.get("history_token_ids")
        observed_rows = record.get("physical_row_ids")
        if history is None or observed_rows is None:
            raise ValueError(
                f"{trace_path}:{line_number}: normalized record lacks "
                "history_token_ids or physical_row_ids"
            )

        key = (
            record.get("request_id"),
            record.get("token_position"),
            record.get("step_id"),
        )
        candidate = {
            "history_token_ids": history,
            "physical_row_ids": observed_rows,
        }
        previous = grouped.setdefault(key, candidate)
        if previous != candidate:
            raise ValueError(
                f"{trace_path}:{line_number}: inconsistent records for token key {key}"
            )

    mismatches: list[dict[str, object]] = []
    for key, record in grouped.items():
        expected = rows_for_history(
            [int(token) for token in record["history_token_ids"]]
        )
        observed = [int(row) for row in record["physical_row_ids"]]
        if expected != observed and len(mismatches) < 20:
            mismatches.append(
                {
                    "request_id": key[0],
                    "token_position": key[1],
                    "expected": expected,
                    "observed": observed,
                }
            )

    checked_tokens = len(grouped)
    mismatch_count = sum(
        rows_for_history([int(token) for token in record["history_token_ids"]])
        != [int(row) for row in record["physical_row_ids"]]
        for record in grouped.values()
    )

    return {
        "schema_version": 1,
        "checked_tokens": checked_tokens,
        "mismatches": mismatch_count,
        "passed": checked_tokens > 0 and mismatch_count == 0,
        "examples": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = verify_trace(args.trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
