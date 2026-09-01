"""Normalize concurrency-one PLE runtime batches into token/head records.

Exact request correlation is intentionally fail-closed. A runtime PLE address
batch must map to exactly one measured client request interval; ambiguous or
unmatched batches abort normalization rather than disappearing silently.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def correlate_request(
    address_known_ns: int,
    clients: list[dict[str, Any]],
) -> dict[str, Any]:
    matches = [
        record
        for record in clients
        if int(record["start_ns"]) <= address_known_ns <= int(record["end_ns"])
    ]
    if len(matches) != 1:
        raise ValueError(
            "PLE batch cannot be correlated uniquely to a client request: "
            f"address_known_ns={address_known_ns}, matches={len(matches)}"
        )
    return matches[0]


def classify_phase(
    segment_length: int,
    address_known_ns: int,
    client_record: dict[str, Any],
) -> str:
    if segment_length > 1:
        return "prefill"

    first_token_ns = client_record.get("first_token_ns")
    if first_token_ns is not None and address_known_ns >= int(first_token_ns):
        return "decode"

    return "unknown"


def normalize(
    runtime_path: Path,
    client_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    runtime_events = [
        record
        for record in load_jsonl(runtime_path)
        if record.get("kind") == "address_batch"
    ]
    clients = [
        record
        for record in load_jsonl(client_path)
        if "error" not in record
    ]

    if not runtime_events:
        raise ValueError("runtime trace contains no PLE address batches")
    if not clients:
        raise ValueError("client trace contains no successful requests")
    if any(int(record.get("concurrency", 1)) != 1 for record in clients):
        raise ValueError(
            "exact request correlation currently requires concurrency=1"
        )

    absolute_position: defaultdict[str, int] = defaultdict(int)
    output: list[dict[str, Any]] = []

    for event in runtime_events:
        address_known_ns = int(event["address_known_ns"])
        request = correlate_request(address_known_ns, clients)
        request_id = str(request["request_id"])

        query_start = [int(value) for value in event["query_start_loc"]]
        input_ids = [int(value) for value in event["input_ids"]]
        physical_rows = event["physical_row_ids"]
        contexts = event.get("ngram_context") or []
        row_bytes = event.get("row_bytes")

        if len(query_start) < 2:
            raise ValueError("query_start_loc must contain at least two offsets")

        for slot in range(len(query_start) - 1):
            begin, end = query_start[slot], query_start[slot + 1]
            if not (0 <= begin <= end <= len(input_ids)):
                raise ValueError(
                    f"invalid query segment [{begin}, {end}) for {len(input_ids)} tokens"
                )

            segment = input_ids[begin:end]
            history = [
                int(value)
                for value in (contexts[slot] if slot < len(contexts) else [])
            ]
            phase = classify_phase(len(segment), address_known_ns, request)

            for local_index, token in enumerate(segment):
                history.append(int(token))
                row_index = begin + local_index
                if row_index >= len(physical_rows):
                    raise ValueError(
                        "physical_row_ids is shorter than the packed input token stream"
                    )

                rows_for_token = [int(row) for row in physical_rows[row_index]]
                token_position = absolute_position[request_id]
                absolute_position[request_id] += 1

                for head, row in enumerate(rows_for_token):
                    output.append(
                        {
                            "request_id": request_id,
                            "token_position": token_position,
                            "phase": phase,
                            "ple_head": head,
                            "physical_row_id": row,
                            "bytes": row_bytes,
                            "address_known_ns": address_known_ns,
                            "consumption_ns": None,
                            "concurrency": 1,
                            "history_token_ids": list(history),
                            "physical_row_ids": rows_for_token,
                            "step_id": event.get("step_id"),
                        }
                    )

    summary = {
        "row_records": len(output),
        "runtime_batches": len(runtime_events),
        "client_requests": len(clients),
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records, summary = normalize(args.runtime, args.client)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
