"""Merge profiler-derived PLE timing into normalized row records.

The timing CSV must contain one row per PLE sequence identifier with:

    step_id,consumption_ns[,service_start_ns,service_end_ns,exposed_stall_ns]

All normalized PLE records must match a timing row. Duplicate timing identifiers,
missing identifiers, and impossible time ordering are rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

_OPTIONAL_TIMING_FIELDS = (
    "service_start_ns",
    "service_end_ns",
    "exposed_stall_ns",
)


def load_timing(path: Path) -> dict[int, dict[str, str]]:
    timing: dict[int, dict[str, str]] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"step_id", "consumption_ns"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path}: timing CSV is missing columns: {sorted(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            step_id = int(row["step_id"])
            if step_id in timing:
                raise ValueError(
                    f"{path}:{line_number}: duplicate step_id {step_id}"
                )
            timing[step_id] = row

    if not timing:
        raise ValueError("timing CSV contains no data rows")
    return timing


def merge_records(
    ple_path: Path,
    timing: dict[int, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    matched_step_ids: set[int] = set()

    for line_number, line in enumerate(ple_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue

        record = json.loads(line)
        if record.get("step_id") is None:
            raise ValueError(f"{ple_path}:{line_number}: missing step_id")

        step_id = int(record["step_id"])
        timing_record = timing.get(step_id)
        if timing_record is None:
            raise ValueError(
                f"{ple_path}:{line_number}: no timing row for step_id {step_id}"
            )

        consumption_ns = int(timing_record["consumption_ns"])
        address_known_ns = int(record["address_known_ns"])
        if consumption_ns < address_known_ns:
            raise ValueError(
                f"step_id {step_id}: consumption precedes address availability"
            )

        record["consumption_ns"] = consumption_ns
        for field in _OPTIONAL_TIMING_FIELDS:
            value = timing_record.get(field)
            if value not in (None, ""):
                record[field] = int(value)

        matched_step_ids.add(step_id)
        output.append(record)

    if not output:
        raise ValueError("PLE trace contains no normalized records")

    unused = sorted(set(timing).difference(matched_step_ids))
    if unused:
        raise ValueError(
            "timing CSV contains unmatched step IDs: "
            + ", ".join(str(value) for value in unused[:20])
        )

    return output, {
        "records": len(output),
        "matched_step_ids": len(matched_step_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ple", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    timing = load_timing(args.timing_csv)
    records, summary = merge_records(args.ple, timing)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
