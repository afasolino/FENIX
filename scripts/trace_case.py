#!/usr/bin/env python3
"""Execute and atomically publish one predeclared FENIX trace case."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from analysis import expert_locality, ple_locality, process_moe_trace, process_ple_trace
from analysis import trace_characterization
from scripts import bench_openai, trace_capture, trace_contract


class TraceCaseError(RuntimeError):
    """Raised when a trace case cannot be published as valid trace evidence."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(record), sort_keys=True) + "\n" for record in records)
    trace_capture.atomic_write_text(path, text)


def _run_prompt_job(
    ordinal: int,
    prompt: str,
    *,
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
) -> dict[str, Any]:
    request_id = f"fenix-trace-{ordinal:04d}-{uuid.uuid4().hex[:8]}"
    try:
        result = bench_openai.run_one(
            url, model, prompt, max_tokens, temperature, request_id
        )
        result.update(ordinal=ordinal, concurrency=concurrency)
        return result
    except Exception as exc:
        return {
            "request_id": request_id,
            "ordinal": ordinal,
            "concurrency": concurrency,
            "error": repr(exc),
        }


def run_prompt_benchmark(
    prompts: Sequence[str],
    *,
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
) -> tuple[list[dict[str, Any]], float]:
    if not prompts:
        raise TraceCaseError("trace prompt set is empty")
    if concurrency < 1:
        raise TraceCaseError("concurrency must be >= 1")

    start_ns = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _run_prompt_job,
                ordinal,
                prompt,
                url=url,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                concurrency=concurrency,
            )
            for ordinal, prompt in enumerate(prompts)
        ]
        results = [future.result() for future in futures]
    end_ns = time.perf_counter_ns()
    results.sort(key=lambda record: int(record["ordinal"]))
    return results, (end_ns - start_ns) / 1e9


def validate_client_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_requests: int,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
) -> list[str]:
    reasons: list[str] = []
    if len(records) != expected_requests:
        reasons.append(f"client_request_count:{len(records)}!={expected_requests}")
    if any("error" in record for record in records):
        reasons.append("client_request_error")
    for record in records:
        if "error" in record:
            continue
        ordinal = record.get("ordinal")
        if record.get("prompt_tokens") != input_tokens:
            reasons.append(f"prompt_tokens_mismatch:{ordinal}")
        if record.get("completion_tokens") != output_tokens:
            reasons.append(f"completion_tokens_mismatch:{ordinal}")
        if record.get("concurrency") != concurrency:
            reasons.append(f"concurrency_mismatch:{ordinal}")
        if record.get("start_ns") is None or record.get("end_ns") is None:
            reasons.append(f"client_timing_missing:{ordinal}")
    return sorted(set(reasons))


def validate_ple_events(
    events: Sequence[Mapping[str, Any]],
    campaign_path: Path,
) -> list[str]:
    payload = json.loads(campaign_path.read_text())
    try:
        model = payload["model"]
        expected_rows = (
            (int(model["ngram_size"]) - 1)
            * int(model["heads_per_ngram"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TraceCaseError("campaign PLE geometry is incomplete") from exc
    if expected_rows < 1:
        raise TraceCaseError("campaign PLE row count per token must be positive")

    reasons: list[str] = []
    for index, event in enumerate(events):
        if event.get("kind") != "address_batch":
            reasons.append(f"unexpected_ple_event_kind:{index}")
            continue
        input_ids = event.get("input_ids")
        rows = event.get("physical_row_ids")
        row_bytes = event.get("row_bytes")
        if not isinstance(input_ids, list) or not isinstance(rows, list):
            reasons.append(f"invalid_ple_packed_arrays:{index}")
            continue
        if len(input_ids) != len(rows):
            reasons.append(f"ple_token_row_count_mismatch:{index}")
        if any(not isinstance(item, list) or len(item) != expected_rows for item in rows):
            reasons.append(f"ple_rows_per_token_mismatch:{index}")
        if not isinstance(row_bytes, int) or isinstance(row_bytes, bool) or row_bytes <= 0:
            reasons.append(f"invalid_ple_row_bytes:{index}")
    return sorted(set(reasons))


def _trace_capacities(campaign_path: Path) -> list[float]:
    payload = json.loads(campaign_path.read_text())
    raw = payload.get("trace_analysis", {}).get("ple_cache_capacities_gib")
    if not isinstance(raw, list) or not raw:
        raise TraceCaseError("trace_analysis.ple_cache_capacities_gib is missing")
    values = [float(value) for value in raw]
    if any(value <= 0 for value in values):
        raise TraceCaseError("PLE trace cache capacities must be positive")
    return values


def _artifact_hashes(case_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(case_dir)): trace_capture.sha256_file(path)
        for path in sorted(case_dir.rglob("*"))
        if path.is_file() and path.name != "evidence.json"
    }


def _write_prompts(
    case_dir: Path,
    prepared: trace_contract.PreparedTracePrompts,
) -> None:
    prompt_dir = case_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ordinal, (prompt, digest) in enumerate(
        zip(prepared.prompts, prepared.prompt_hashes, strict=True)
    ):
        relative = Path("prompts") / f"{ordinal:04d}.txt"
        trace_capture.atomic_write_text(case_dir / relative, prompt)
        manifest.append(
            {
                "ordinal": ordinal,
                "path": str(relative),
                "sha256": digest,
                "characters": len(prompt),
            }
        )
    trace_capture.write_json(
        case_dir / "prompts.json",
        {
            "prompt_set_sha256": prepared.prompt_set_sha256,
            "prompt_tokens": prepared.prompt_tokens,
            "max_model_len": prepared.max_model_len,
            "tokenize_url": prepared.tokenize_url,
            "prompts": manifest,
        },
    )


def _normalize_exact_case(
    case_dir: Path,
    client_records: Sequence[Mapping[str, Any]],
    campaign_path: Path,
) -> dict[str, Any]:
    clients = case_dir / "client.jsonl"
    ple_raw = case_dir / "ple_runtime.jsonl"
    moe_raw = case_dir / "moe_runtime.jsonl"
    ple_out = case_dir / "ple_normalized.jsonl"
    moe_out = case_dir / "moe_normalized.jsonl"

    ple_records, ple_summary = process_ple_trace.normalize(ple_raw, clients)
    moe_records, moe_summary = process_moe_trace.normalize(moe_raw, clients)
    write_jsonl(ple_out, ple_records)
    write_jsonl(moe_out, moe_records)

    expected_ids = {
        str(record["request_id"])
        for record in client_records
        if "error" not in record
    }
    joint = trace_characterization.analyze(
        ple_out, moe_out, expected_request_ids=expected_ids
    )
    ple_analysis = ple_locality.analyze_trace(
        ple_out, _trace_capacities(campaign_path)
    )
    if any(
        record.get("trace_scope") == "selection_only"
        for record in moe_records
    ):
        expert_keys: set[tuple[int, int]] = set()
        expert_selections = 0
        for record in moe_records:
            layer = expert_locality.parse_layer_id(record["layer"])
            selected = [
                int(value)
                for value in record.get("selected_expert_ids", [])
            ]
            expert_selections += len(selected)
            expert_keys.update((layer, expert) for expert in selected)
        expert_analysis = {
            "schema_version": 2,
            "evidence_kind": "local_measured_trace_analysis",
            "expert_selections": expert_selections,
            "unique_layer_experts": len(expert_keys),
            "reuse_distance": {
                "exact_stack_reuse_deferred": True,
                "reason": (
                    "full prefill selection batches can contain tens of "
                    "millions of expert references; H1 computes a streaming "
                    "reuse-gap distribution instead"
                ),
            },
        }
    else:
        expert_analysis = expert_locality.analyze_trace(moe_raw)
    trace_capture.write_json(case_dir / "joint_characterization.json", joint)
    trace_capture.write_json(case_dir / "ple_locality.json", ple_analysis)
    trace_capture.write_json(case_dir / "expert_locality.json", expert_analysis)
    return {
        "ple": ple_summary,
        "moe": moe_summary,
        "joint_request_count": joint["request_count"],
    }


def run_case(
    *,
    root: Path,
    contract: trace_contract.TraceContract,
    case: trace_contract.TraceCase,
    campaign_path: Path,
    runtime_lane_path: Path,
    server_log: Path,
    out_root: Path,
    ple_source: Path,
    moe_source: Path,
    url: str,
    model: str,
    tokenize_url: str | None,
    settle_ms: int,
) -> Path:
    final_dir = out_root / case.case_id
    if final_dir.exists():
        raise TraceCaseError(f"refusing to overwrite trace case: {final_dir}")

    repository = trace_capture.repository_state(root)
    if not repository.clean:
        raise TraceCaseError(
            f"trace collection requires a clean repository: {repository.status}"
        )
    launch = trace_capture.require_trace_server(server_log)
    image_tag = launch.runtime_images[0]
    image_id = trace_capture.resolve_image_id(root, image_tag)
    runtime_lane = trace_capture.load_runtime_lane(runtime_lane_path)
    prepared = trace_contract.prepare_trace_prompts(
        contract=contract,
        input_tokens=case.input_tokens,
        chat_url=url,
        model=model,
        tokenize_url=tokenize_url,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    partial = out_root / f".{case.case_id}.partial-{os.getpid()}"
    if partial.exists():
        raise TraceCaseError(f"temporary case directory already exists: {partial}")
    partial.mkdir()
    started_at = utc_now()

    try:
        _write_prompts(partial, prepared)
        starts = {
            "ple": trace_capture.file_offset(ple_source),
            "moe": trace_capture.file_offset(moe_source),
            "server": trace_capture.file_offset(server_log),
        }
        records, wall_s = run_prompt_benchmark(
            prepared.prompts,
            url=url,
            model=model,
            max_tokens=contract.output_tokens,
            temperature=contract.temperature,
            concurrency=case.concurrency,
        )
        write_jsonl(partial / "client.jsonl", records)
        trace_capture.write_json(
            partial / "client.jsonl.summary.json",
            bench_openai.summarize_results(records, wall_s, case.concurrency),
        )
        if settle_ms:
            time.sleep(settle_ms / 1000.0)
        ends = {
            "ple": trace_capture.file_offset(ple_source),
            "moe": trace_capture.file_offset(moe_source),
            "server": trace_capture.file_offset(server_log),
        }

        ple_events = trace_capture.capture_jsonl_window(
            ple_source, starts["ple"], ends["ple"], partial / "ple_runtime.jsonl"
        )
        moe_events = trace_capture.capture_jsonl_window(
            moe_source, starts["moe"], ends["moe"], partial / "moe_runtime.jsonl"
        )
        server_bytes = trace_capture.read_byte_window(
            server_log, starts["server"], ends["server"]
        )
        if server_bytes and not server_bytes.endswith(b"\n"):
            raise TraceCaseError("server-log window ends with a partial line")
        server_text = server_bytes.decode("utf-8", errors="replace")
        trace_capture.atomic_write_text(partial / "server-window.log", server_text)

        reasons = validate_client_records(
            records,
            expected_requests=contract.requests_per_input_length,
            input_tokens=case.input_tokens,
            output_tokens=contract.output_tokens,
            concurrency=case.concurrency,
        )
        reasons.extend(validate_ple_events(ple_events, campaign_path))
        if any(
            "layer" not in event or not event.get("selected_expert_ids")
            for event in moe_events
        ):
            reasons.append("invalid_moe_event")

        contamination = trace_capture.contamination_hits(server_text)
        fatal = trace_capture.fatal_contamination(contamination)
        if fatal:
            reasons.extend(f"fatal_contamination:{item['id']}" for item in fatal)

        normalization: dict[str, Any] | None = None
        if not reasons and case.correlation_mode == "exact_request_correlation":
            normalization = _normalize_exact_case(partial, records, campaign_path)
        if reasons:
            raise TraceCaseError("; ".join(sorted(set(reasons))))

        evidence = {
            "schema_version": 1,
            "evidence_kind": "local_measured_trace",
            "trace_valid": True,
            "performance_eligible": False,
            "can_establish_motivation": False,
            "repository_commit": repository.commit,
            "repository_clean": repository.clean,
            "campaign_sha256": trace_capture.sha256_file(campaign_path),
            "runtime_lane": runtime_lane,
            "launch": {
                "trace_values": list(launch.trace_values),
                "runtime_image": image_tag,
                "runtime_image_id": image_id,
            },
            "case": {
                "case_id": case.case_id,
                "experiment": contract.experiment,
                "workload_profile": contract.workload_profile,
                "input_tokens": case.input_tokens,
                "output_tokens": contract.output_tokens,
                "requests": contract.requests_per_input_length,
                "temperature": contract.temperature,
                "concurrency": case.concurrency,
                "repetition_index": case.repetition_index,
                "correlation_mode": case.correlation_mode,
                "prompt_set_sha256": prepared.prompt_set_sha256,
            },
            "trace_windows": {
                "ple": {"start": starts["ple"], "end": ends["ple"], "records": len(ple_events)},
                "moe": {"start": starts["moe"], "end": ends["moe"], "records": len(moe_events)},
                "server_log": {"start": starts["server"], "end": ends["server"]},
            },
            "server_log_window": {
                "contamination_hits": contamination,
                "fatal_contamination_hits": fatal,
            },
            "normalization": normalization,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
        }
        evidence["artifacts_sha256"] = _artifact_hashes(partial)
        trace_capture.write_json(partial / "evidence.json", evidence)
        partial.replace(final_dir)
        return final_dir
    except Exception as exc:
        failure_dir = out_root / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_name = (
            f"{case.case_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        )
        trace_capture.write_json(
            failure_dir / failure_name,
            {
                "case_id": case.case_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "started_at_utc": started_at,
                "failed_at_utc": utc_now(),
            },
        )
        shutil.rmtree(partial, ignore_errors=True)
        raise
