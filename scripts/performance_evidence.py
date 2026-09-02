#!/usr/bin/env python3
"""Run a fail-closed, campaign-conformant FENIX performance measurement.

This orchestrator separates warmup from measured requests, enforces the
versioned workload contract, and emits a machine-readable evidence manifest.
Use ``scripts.bench_openai`` directly for diagnostic or trace-mode client
measurements; this command only succeeds when the result is eligible for
performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import bench_openai, workload_contract


STARTUP_MARKER = "Application startup complete."
TRACE_PATTERNS = (
    re.compile(r'"FENIX_TRACE"\s*:\s*"([01])"'),
    re.compile(r"\bFENIX_TRACE=([01])\b"),
)
FENIX_IMAGE_PATTERN = re.compile(
    r"\b(?:localhost/)?fenix-qwen38:[A-Za-z0-9_.-]+\b"
)
CONTAMINATION_MARKERS: tuple[tuple[str, str], ...] = (
    (
        "inference_jit",
        "Triton kernel JIT compilation during inference",
    ),
    (
        "cuda_oom",
        "CUDA out of memory",
    ),
    (
        "torch_oom",
        "torch.OutOfMemoryError",
    ),
    (
        "allocator_mapping_oom",
        "memory mapping failed with OOM",
    ),
    (
        "python_traceback",
        "Traceback (most recent call last)",
    ),
    (
        "runtime_error",
        "RuntimeError:",
    ),
    (
        "shm_stall",
        "No available shared memory broadcast block found in 60 seconds",
    ),
)


class PerformanceEvidenceError(RuntimeError):
    """Raised when a precondition prevents a performance measurement."""


@dataclass(frozen=True)
class RepositoryState:
    commit: str
    clean: bool
    status: tuple[str, ...]


@dataclass(frozen=True)
class LaunchMetadata:
    startup_complete: bool
    trace_values: tuple[str, ...]
    runtime_images: tuple[str, ...]

    @property
    def trace_enabled(self) -> bool | None:
        if len(self.trace_values) != 1:
            return None
        return self.trace_values[0] == "1"


@dataclass(frozen=True)
class MeasurementConfig:
    server_log: Path
    output: Path
    url: str
    model: str
    log_settle_ms: int
    runtime_lane: Path
    campaign: Path
    experiment: str
    repetition_index: int
    tokenize_url: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(json.dumps(record) + "\n" for record in records)
    atomic_write_text(path, content)


def repository_state() -> RepositoryState:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    status = tuple(
        line for line in status_result.stdout.splitlines() if line.strip()
    )
    return RepositoryState(
        commit=commit_result.stdout.strip(),
        clean=not status,
        status=status,
    )


def inspect_launch_log(text: str) -> LaunchMetadata:
    trace_values: set[str] = set()
    for pattern in TRACE_PATTERNS:
        trace_values.update(pattern.findall(text))

    runtime_images = tuple(sorted(set(FENIX_IMAGE_PATTERN.findall(text))))
    return LaunchMetadata(
        startup_complete=STARTUP_MARKER in text,
        trace_values=tuple(sorted(trace_values)),
        runtime_images=runtime_images,
    )


def require_performance_server(server_log: Path) -> LaunchMetadata:
    if not server_log.is_file():
        raise PerformanceEvidenceError(
            f"server log does not exist: {server_log}"
        )

    launch = inspect_launch_log(
        server_log.read_text(errors="replace")
    )
    failures: list[str] = []

    if not launch.startup_complete:
        failures.append("server startup marker is missing")
    if launch.trace_values != ("0",):
        failures.append(
            "performance server must expose exactly FENIX_TRACE=0; "
            f"observed={launch.trace_values or 'missing'}"
        )
    if len(launch.runtime_images) != 1:
        failures.append(
            "server log must identify exactly one FENIX runtime image; "
            f"observed={launch.runtime_images or 'missing'}"
        )

    if failures:
        raise PerformanceEvidenceError("; ".join(failures))
    return launch


def contamination_hits(log_window: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for marker_id, needle in CONTAMINATION_MARKERS:
        count = log_window.count(needle)
        if count:
            hits.append(
                {
                    "id": marker_id,
                    "needle": needle,
                    "count": count,
                }
            )
    return hits


def read_log_window(path: Path, start: int, end: int) -> str:
    if start < 0 or end < start:
        raise PerformanceEvidenceError(
            f"invalid server-log byte window: start={start}, end={end}"
        )

    with path.open("rb") as stream:
        stream.seek(start)
        payload = stream.read(end - start)
    return payload.decode("utf-8", errors="replace")


def load_runtime_lane(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PerformanceEvidenceError(
            f"runtime lane configuration does not exist: {path}"
        )

    payload = json.loads(path.read_text())
    try:
        return {
            "lane_id": payload["lane_id"],
            "runtime_repository": payload["runtime"]["repository"],
            "runtime_revision": payload["runtime"]["revision"],
            "base_container_image": payload["runtime"]["container_image"],
            "model_repository": payload["model"]["repository"],
            "model_revision": payload["model"]["revision"],
        }
    except (KeyError, TypeError) as exc:
        raise PerformanceEvidenceError(
            f"runtime lane is missing required provenance fields: {path}"
        ) from exc


def _benchmark_namespace(
    config: MeasurementConfig,
    contract: workload_contract.ExperimentContract,
    prompt: str,
    requests: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        url=config.url,
        model=config.model,
        prompt=prompt,
        max_tokens=contract.output_tokens,
        temperature=contract.temperature,
        concurrency=contract.concurrency,
        requests=requests,
    )


def run_phase(
    config: MeasurementConfig,
    contract: workload_contract.ExperimentContract,
    prompt: str,
    requests: int,
    phase: str,
) -> tuple[list[dict[str, Any]], float]:
    results, wall_s = bench_openai.run_benchmark(
        _benchmark_namespace(
            config,
            contract,
            prompt,
            requests,
        )
    )
    annotated: list[dict[str, Any]] = []
    for record in results:
        item = dict(record)
        item["phase"] = phase
        annotated.append(item)
    return annotated, wall_s


def _successful(records: Sequence[Mapping[str, Any]]) -> bool:
    return bool(records) and all("error" not in record for record in records)


def _measured_timing_complete(
    records: Sequence[Mapping[str, Any]],
) -> bool:
    if not _successful(records):
        return False

    for record in records:
        completion_tokens = record.get("completion_tokens")
        if not isinstance(completion_tokens, int) or completion_tokens < 1:
            return False
        if record.get("ttft_ms") is None:
            return False
        if record.get("e2e_ms") is None:
            return False
        if completion_tokens >= 2 and record.get("tpot_ms") is None:
            return False
    return True


def _workload_reason_ids(
    mismatches: Sequence[Mapping[str, object]],
) -> list[str]:
    fields = sorted(
        {
            str(mismatch.get("field"))
            for mismatch in mismatches
            if mismatch.get("field")
        }
    )
    return [f"workload:{field}_mismatch" for field in fields]


def evaluate_eligibility(
    *,
    repository: RepositoryState,
    launch: LaunchMetadata,
    warmup_records: Sequence[Mapping[str, Any]],
    measured_records: Sequence[Mapping[str, Any]],
    workload_mismatches: Sequence[Mapping[str, object]],
    contamination: Sequence[Mapping[str, Any]],
    log_window_valid: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if not repository.clean:
        reasons.append("repository_not_clean")
    if launch.trace_values != ("0",):
        reasons.append("trace_mode_not_disabled")
    if not launch.startup_complete:
        reasons.append("server_not_started")
    if not _successful(warmup_records):
        reasons.append("warmup_failed")
    if not _successful(measured_records):
        reasons.append("measured_requests_failed")
    if not _measured_timing_complete(measured_records):
        reasons.append("measured_timing_incomplete")
    reasons.extend(_workload_reason_ids(workload_mismatches))
    if not log_window_valid:
        reasons.append("server_log_window_invalid")
    if contamination:
        reasons.extend(
            f"contamination:{item['id']}" for item in contamination
        )

    return not reasons, reasons


def _artifact_paths(output: Path) -> dict[str, Path]:
    return {
        "measured": output,
        "summary": Path(f"{output}.summary.json"),
        "warmup": Path(f"{output}.warmup.jsonl"),
        "prompt": Path(f"{output}.prompt.txt"),
        "server_window": Path(f"{output}.server-window.log"),
        "evidence": Path(f"{output}.evidence.json"),
    }


def _ensure_artifacts_absent(paths: Mapping[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise PerformanceEvidenceError(
            "refusing to overwrite existing performance artifacts: "
            + ", ".join(existing)
        )


def _artifact_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "evidence" and path.is_file()
    }


def _workload_manifest(
    *,
    config: MeasurementConfig,
    contract: workload_contract.ExperimentContract,
    prepared: workload_contract.PreparedWorkload,
) -> dict[str, Any]:
    return {
        "experiment": contract.experiment,
        "workload_profile": contract.workload_profile,
        "url": config.url,
        "tokenize_url": prepared.tokenize_url,
        "model": config.model,
        "prompt_sha256": prepared.prompt_sha256,
        "prompt_characters": len(prepared.prompt),
        "preflight_prompt_tokens": prepared.prompt_tokens,
        "server_max_model_len": prepared.max_model_len,
        "expected_input_tokens": contract.input_tokens,
        "expected_output_tokens": contract.output_tokens,
        "temperature": contract.temperature,
        "concurrency": contract.concurrency,
        "warmup_requests": contract.warmup_requests,
        "measured_requests": contract.measured_requests,
        "repetition_index": config.repetition_index,
        "required_repetitions": contract.repetitions,
    }


def _diagnostic_manifest(
    *,
    repository: RepositoryState,
    runtime_lane: Mapping[str, Any],
    launch: LaunchMetadata,
    workload: Mapping[str, Any],
    reasons: Sequence[str],
    started_at: str,
    paths: Mapping[str, Path],
    workload_mismatches: Sequence[Mapping[str, object]],
    warmup_wall_s: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "evidence_kind": "diagnostic_measurement",
        "performance_eligible": False,
        "eligibility_reasons": list(reasons),
        "repository_commit": repository.commit,
        "repository_clean": repository.clean,
        "runtime_lane": dict(runtime_lane),
        "launch": {
            "trace_values": list(launch.trace_values),
            "runtime_images": list(launch.runtime_images),
        },
        "workload": dict(workload),
        "workload_mismatches": list(workload_mismatches),
        "warmup_wall_s": warmup_wall_s,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "artifacts_sha256": _artifact_hashes(paths),
    }


def run_measurement(config: MeasurementConfig) -> tuple[dict[str, Any], int]:
    if config.log_settle_ms < 0:
        raise PerformanceEvidenceError("log_settle_ms must be >= 0")

    paths = _artifact_paths(config.output)
    _ensure_artifacts_absent(paths)

    repository = repository_state()
    if not repository.clean:
        raise PerformanceEvidenceError(
            "performance measurements require a clean repository; "
            f"status={repository.status}"
        )

    launch = require_performance_server(config.server_log)
    runtime_lane = load_runtime_lane(config.runtime_lane)

    try:
        contract = workload_contract.load_experiment_contract(
            config.campaign,
            config.experiment,
        )
        workload_contract.validate_repetition_index(
            contract,
            config.repetition_index,
        )
        prepared = workload_contract.prepare_workload(
            contract=contract,
            chat_url=config.url,
            model=config.model,
            tokenize_url=config.tokenize_url,
        )
    except workload_contract.WorkloadContractError as exc:
        raise PerformanceEvidenceError(str(exc)) from exc

    started_at = utc_now()
    atomic_write_text(paths["prompt"], prepared.prompt)
    workload_manifest = _workload_manifest(
        config=config,
        contract=contract,
        prepared=prepared,
    )

    warmup_records, warmup_wall_s = run_phase(
        config,
        contract,
        prepared.prompt,
        contract.warmup_requests,
        "warmup",
    )
    write_jsonl(paths["warmup"], warmup_records)

    warmup_mismatches = workload_contract.record_token_mismatches(
        warmup_records,
        contract=contract,
        phase="warmup",
    )
    if not _successful(warmup_records) or warmup_mismatches:
        reasons = []
        if not _successful(warmup_records):
            reasons.append("warmup_failed")
        reasons.extend(_workload_reason_ids(warmup_mismatches))
        evidence = _diagnostic_manifest(
            repository=repository,
            runtime_lane=runtime_lane,
            launch=launch,
            workload=workload_manifest,
            reasons=reasons,
            started_at=started_at,
            paths=paths,
            workload_mismatches=warmup_mismatches,
            warmup_wall_s=warmup_wall_s,
        )
        write_json(paths["evidence"], evidence)
        return evidence, 3

    measured_log_start = config.server_log.stat().st_size
    measured_records, measured_wall_s = run_phase(
        config,
        contract,
        prepared.prompt,
        contract.measured_requests,
        "measured",
    )

    if config.log_settle_ms:
        time.sleep(config.log_settle_ms / 1000.0)

    measured_log_end = config.server_log.stat().st_size
    log_window_valid = measured_log_end >= measured_log_start
    log_window = (
        read_log_window(
            config.server_log,
            measured_log_start,
            measured_log_end,
        )
        if log_window_valid
        else ""
    )
    atomic_write_text(paths["server_window"], log_window)

    measured_mismatches = workload_contract.record_token_mismatches(
        measured_records,
        contract=contract,
        phase="measured",
    )
    contamination = contamination_hits(log_window)
    eligible, reasons = evaluate_eligibility(
        repository=repository,
        launch=launch,
        warmup_records=warmup_records,
        measured_records=measured_records,
        workload_mismatches=measured_mismatches,
        contamination=contamination,
        log_window_valid=log_window_valid,
    )

    summary = bench_openai.summarize_results(
        measured_records,
        measured_wall_s,
        contract.concurrency,
    )
    summary["run_class"] = "performance"
    summary["performance_eligible"] = eligible
    summary["eligibility_reasons"] = reasons
    bench_openai.write_results(
        paths["measured"],
        measured_records,
        summary,
    )

    evidence_kind = (
        "local_measured" if eligible else "diagnostic_measurement"
    )
    evidence = {
        "schema_version": 2,
        "evidence_kind": evidence_kind,
        "performance_eligible": eligible,
        "eligibility_reasons": reasons,
        "repository_commit": repository.commit,
        "repository_clean": repository.clean,
        "runtime_lane": runtime_lane,
        "launch": {
            "trace_values": list(launch.trace_values),
            "trace_enabled": launch.trace_enabled,
            "runtime_images": list(launch.runtime_images),
            "server_log": str(config.server_log),
        },
        "workload": workload_manifest,
        "workload_mismatches": measured_mismatches,
        "timing": {
            "warmup_wall_s": warmup_wall_s,
            "measured_wall_s": measured_wall_s,
        },
        "server_log_window": {
            "start_byte": measured_log_start,
            "end_byte": measured_log_end,
            "valid": log_window_valid,
            "contamination_hits": contamination,
        },
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
    }

    evidence["artifacts_sha256"] = _artifact_hashes(paths)
    write_json(paths["evidence"], evidence)
    return evidence, 0 if eligible else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default=bench_openai.DEFAULT_URL)
    parser.add_argument("--model", default=bench_openai.DEFAULT_MODEL)
    parser.add_argument("--tokenize-url")
    parser.add_argument("--log-settle-ms", type=int, default=250)
    parser.add_argument(
        "--runtime-lane",
        type=Path,
        default=Path("configs/runtime_lane.json"),
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=workload_contract.DEFAULT_CAMPAIGN,
    )
    parser.add_argument(
        "--experiment",
        default=workload_contract.DEFAULT_EXPERIMENT,
    )
    parser.add_argument(
        "--repetition-index",
        type=int,
        required=True,
        help="1-based repetition index within the predeclared campaign",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = MeasurementConfig(
        server_log=args.server_log,
        output=args.out,
        url=args.url,
        model=args.model,
        log_settle_ms=args.log_settle_ms,
        runtime_lane=args.runtime_lane,
        campaign=args.campaign,
        experiment=args.experiment,
        repetition_index=args.repetition_index,
        tokenize_url=args.tokenize_url,
    )

    try:
        evidence, return_code = run_measurement(config)
    except PerformanceEvidenceError as exc:
        print(
            json.dumps(
                {
                    "performance_eligible": False,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 2

    print(json.dumps(evidence, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
