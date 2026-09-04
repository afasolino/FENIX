#!/usr/bin/env python3
"""Collect exact C=1 Qwen3.8 traces for the natural workload-robustness suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import bench_openai, trace_capture, trace_case, workload_contract

DEFAULT_CONTRACT = Path("configs/h1_h2_workload_robustness_v1.json")
DEFAULT_MODEL_CAMPAIGN = Path("configs/campaign.json")
DEFAULT_CORPUS = Path("external/workloads/h1_h2_workload_robustness_v1/corpus.jsonl")
DEFAULT_SOURCE_MANIFEST = Path(
    "external/workloads/h1_h2_workload_robustness_v1/source_manifest.json"
)
DEFAULT_RUNTIME_LANE = Path("configs/runtime_lane.json")
DEFAULT_PLE_TRACE = Path("traces/raw/ple_runtime.jsonl")
DEFAULT_MOE_TRACE = Path("traces/raw/moe_runtime.jsonl")
DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3.8-flash-next"


class RobustTraceError(RuntimeError):
    """Raised when a robustness trace case cannot be promoted."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RobustTraceError(f"{path}: expected JSON object")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != 1:
        raise RobustTraceError("unsupported robustness contract schema")
    if payload.get("artifact_kind") != "fenix_h1_h2_workload_robustness_contract":
        raise RobustTraceError("unexpected robustness contract kind")
    trace = payload.get("trace")
    strata = payload.get("strata")
    if not isinstance(trace, dict) or not isinstance(strata, dict):
        raise RobustTraceError("robustness trace/strata contract is incomplete")
    order = trace.get("strata_order")
    if not isinstance(order, list) or set(order) != set(strata):
        raise RobustTraceError("trace.strata_order and strata differ")
    if int(trace.get("concurrency", 0)) != 1:
        raise RobustTraceError("robustness H1 requires concurrency=1")
    if trace.get("correlation_mode") != "exact_request_correlation":
        raise RobustTraceError("robustness H1 requires exact request correlation")
    return payload


def load_corpus(
    corpus_path: Path,
    source_manifest_path: Path,
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not corpus_path.is_file() or not source_manifest_path.is_file():
        raise RobustTraceError(
            "frozen workload corpus is missing; run scripts.prepare_workload_robustness"
        )
    manifest = _load_json(source_manifest_path)
    if manifest.get("artifact_kind") != "fenix_h1_h2_frozen_workload_corpus":
        raise RobustTraceError("unexpected frozen corpus manifest kind")
    observed_hash = trace_capture.sha256_file(corpus_path)
    if manifest.get("corpus_sha256") != observed_hash:
        raise RobustTraceError("frozen corpus SHA256 does not match source manifest")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(corpus_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RobustTraceError(f"{corpus_path}:{line_number}: expected object")
        records.append(value)

    expected = {
        name: int(spec["requests"]) for name, spec in contract["strata"].items()
    }
    observed: dict[str, int] = {name: 0 for name in expected}
    for record in records:
        stratum = str(record.get("stratum"))
        if stratum not in observed:
            raise RobustTraceError(f"corpus contains undeclared stratum: {stratum}")
        observed[stratum] += 1
    if observed != expected:
        raise RobustTraceError(
            f"corpus stratum counts differ: observed={observed} expected={expected}"
        )
    return records, manifest


def _tokenize(
    tokenize_url: str,
    model: str,
    prompt: str,
) -> workload_contract.TokenizationResult:
    return workload_contract.tokenize_prompt(
        tokenize_url,
        model,
        prompt,
        timeout_s=120.0,
    )


def _fit_long_context(
    context: str,
    suffix: str,
    *,
    target_tokens: int,
    minimum_tokens: int,
    tokenize_url: str,
    model: str,
) -> tuple[str, int, int | None, dict[str, Any]]:
    if target_tokens < minimum_tokens:
        raise RobustTraceError("long-context target is below declared minimum")

    cache: dict[int, workload_contract.TokenizationResult] = {}

    def candidate(chars: int) -> tuple[str, workload_contract.TokenizationResult]:
        chars = max(0, min(chars, len(context)))
        prompt = context[:chars].rstrip()
        if prompt:
            prompt += "\n\n"
        prompt += suffix
        if chars not in cache:
            cache[chars] = _tokenize(tokenize_url, model, prompt)
        return prompt, cache[chars]

    full_prompt, full = candidate(len(context))
    if full.count <= target_tokens:
        if full.count < minimum_tokens:
            raise RobustTraceError(
                f"LongBench sample is too short after rendering: {full.count} < {minimum_tokens}"
            )
        return full_prompt, full.count, full.max_model_len, {
            "truncation": "none",
            "context_characters_used": len(context),
        }

    low, high = 0, len(context)
    best_chars = 0
    best_prompt, best = candidate(0)
    if best.count > target_tokens:
        raise RobustTraceError("long-context question/options alone exceed target")

    # Maximize natural context prefix without adding filler.
    for _ in range(32):
        if low > high:
            break
        middle = (low + high) // 2
        prompt, result = candidate(middle)
        if result.count <= target_tokens:
            if result.count >= best.count:
                best_chars, best_prompt, best = middle, prompt, result
            low = middle + 1
        else:
            high = middle - 1

    if best.count < minimum_tokens:
        raise RobustTraceError(
            f"cannot fit long context into declared band: {best.count} < {minimum_tokens}"
        )
    return best_prompt, best.count, best.max_model_len, {
        "truncation": "context_prefix",
        "context_characters_used": best_chars,
        "context_characters_available": len(context),
    }


def _fit_session_suffix(
    prompt: str,
    *,
    max_tokens: int,
    tokenize_url: str,
    model: str,
) -> tuple[str, int, int | None, dict[str, Any]]:
    result = _tokenize(tokenize_url, model, prompt)
    if result.count <= max_tokens:
        return prompt, result.count, result.max_model_len, {"truncation": "none"}

    blocks = [block for block in prompt.split("\n\n") if block.strip()]
    for start in range(1, len(blocks)):
        candidate = "\n\n".join(blocks[start:])
        candidate_result = _tokenize(tokenize_url, model, candidate)
        if candidate_result.count <= max_tokens:
            return candidate, candidate_result.count, candidate_result.max_model_len, {
                "truncation": "drop_oldest_dialogue_blocks",
                "dropped_blocks": start,
            }

    # One final turn can itself be very long. Keep its suffix rather than the
    # oldest prefix because session inference normally preserves recent state.
    text = blocks[-1] if blocks else prompt
    low, high = 1, len(text)
    best = None
    best_result = None
    while low <= high:
        keep = (low + high) // 2
        candidate = text[-keep:]
        candidate_result = _tokenize(tokenize_url, model, candidate)
        if candidate_result.count <= max_tokens:
            best, best_result = candidate, candidate_result
            low = keep + 1
        else:
            high = keep - 1
    if best is None or best_result is None:
        raise RobustTraceError("cannot fit session prompt under input-token limit")
    return best, best_result.count, best_result.max_model_len, {
        "truncation": "recent_character_suffix",
        "characters_kept": len(best),
    }


def prepare_stratum_prompts(
    records: Sequence[Mapping[str, Any]],
    *,
    stratum: str,
    contract: Mapping[str, Any],
    url: str,
    model: str,
    tokenize_url: str | None = None,
) -> tuple[list[str], list[dict[str, Any]], str]:
    spec = contract["strata"][stratum]
    trace = contract["trace"]
    resolved_tokenize_url = tokenize_url or workload_contract.derive_tokenize_url(url)
    max_input_tokens = int(trace["max_input_tokens"])
    server_max_model_len = int(trace["server_max_model_len"])
    max_output_tokens = int(spec.get("max_output_tokens", trace["default_max_output_tokens"]))

    selected = sorted(
        (record for record in records if record.get("stratum") == stratum),
        key=lambda record: int(record["ordinal"]),
    )
    if len(selected) != int(spec["requests"]):
        raise RobustTraceError(
            f"{stratum}: expected {spec['requests']} frozen prompts, found {len(selected)}"
        )

    prompts: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    for expected_ordinal, record in enumerate(selected):
        if int(record["ordinal"]) != expected_ordinal:
            raise RobustTraceError(f"{stratum}: corpus ordinals are not contiguous")

        render_mode = str(record.get("render_mode"))
        fit_meta: dict[str, Any]
        if render_mode == "long_context_prefix_fit":
            prompt, token_count, max_model_len, fit_meta = _fit_long_context(
                str(record["context"]),
                str(record["suffix"]),
                target_tokens=int(spec["target_input_tokens"]),
                minimum_tokens=int(spec["minimum_input_tokens"]),
                tokenize_url=resolved_tokenize_url,
                model=model,
            )
        elif render_mode == "session_suffix_fit":
            prompt, token_count, max_model_len, fit_meta = _fit_session_suffix(
                str(record["prompt"]),
                max_tokens=max_input_tokens,
                tokenize_url=resolved_tokenize_url,
                model=model,
            )
        elif render_mode == "native":
            prompt = str(record["prompt"])
            result = _tokenize(resolved_tokenize_url, model, prompt)
            token_count, max_model_len = result.count, result.max_model_len
            fit_meta = {"truncation": "none"}
            if token_count > max_input_tokens:
                raise RobustTraceError(
                    f"{record['sample_id']}: native prompt exceeds {max_input_tokens} tokens; "
                    "do not repair representative prompts with synthetic filler/truncation"
                )
        else:
            raise RobustTraceError(f"unsupported render_mode: {render_mode}")

        if max_model_len is not None and max_model_len != server_max_model_len:
            raise RobustTraceError(
                f"runtime max_model_len drift: observed={max_model_len} expected={server_max_model_len}"
            )
        if token_count + max_output_tokens > server_max_model_len:
            raise RobustTraceError(
                f"{record['sample_id']}: prompt+max_output exceeds runtime max_model_len"
            )

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        meta = {
            key: value
            for key, value in record.items()
            if key not in {"prompt", "context", "suffix"}
        }
        meta.update(
            {
                "rendered_prompt_tokens": token_count,
                "rendered_prompt_sha256": prompt_hash,
                "rendered_characters": len(prompt),
                "max_output_tokens": max_output_tokens,
                "fit": fit_meta,
            }
        )
        prompts.append(prompt)
        manifest_rows.append(meta)

    prompt_set_hash = hashlib.sha256(
        "\n".join(row["rendered_prompt_sha256"] for row in manifest_rows).encode()
    ).hexdigest()
    return prompts, manifest_rows, prompt_set_hash


def _write_prompt_artifacts(
    case_dir: Path,
    prompts: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_set_sha256: str,
    tokenize_url: str,
) -> None:
    prompt_dir = case_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for ordinal, (prompt, metadata) in enumerate(zip(prompts, rows, strict=True)):
        relative = Path("prompts") / f"{ordinal:04d}.txt"
        trace_capture.atomic_write_text(case_dir / relative, prompt)
        item = dict(metadata)
        item.update({"ordinal": ordinal, "path": str(relative)})
        manifest_rows.append(item)
    trace_capture.write_json(
        case_dir / "prompts.json",
        {
            "schema_version": 1,
            "prompt_set_sha256": prompt_set_sha256,
            "tokenize_url": tokenize_url,
            "prompts": manifest_rows,
        },
    )


def validate_client_records_variable(
    records: Sequence[Mapping[str, Any]],
    prompt_rows: Sequence[Mapping[str, Any]],
    *,
    concurrency: int,
) -> list[str]:
    reasons: list[str] = []
    if len(records) != len(prompt_rows):
        reasons.append(f"client_request_count:{len(records)}!={len(prompt_rows)}")
    for index, (record, prompt) in enumerate(zip(records, prompt_rows)):
        if "error" in record:
            reasons.append(f"client_request_error:{index}")
            continue
        if int(record.get("ordinal", -1)) != index:
            reasons.append(f"client_ordinal_mismatch:{index}")
        if record.get("prompt_tokens") != prompt.get("rendered_prompt_tokens"):
            reasons.append(f"prompt_tokens_mismatch:{index}")
        completion = record.get("completion_tokens")
        maximum = int(prompt["max_output_tokens"])
        if not isinstance(completion, int) or isinstance(completion, bool) or not (1 <= completion <= maximum):
            reasons.append(f"completion_tokens_outside_natural_range:{index}")
        if record.get("concurrency") != concurrency:
            reasons.append(f"concurrency_mismatch:{index}")
        if record.get("start_ns") is None or record.get("end_ns") is None:
            reasons.append(f"client_timing_missing:{index}")
    return sorted(set(reasons))


def run_stratum(
    *,
    root: Path,
    stratum: str,
    contract_path: Path,
    model_campaign_path: Path,
    runtime_lane_path: Path,
    corpus_path: Path,
    source_manifest_path: Path,
    server_log: Path,
    out_root: Path,
    ple_source: Path,
    moe_source: Path,
    url: str,
    model: str,
    tokenize_url: str | None,
    settle_ms: int,
) -> Path:
    contract = load_contract(contract_path)
    corpus, source_manifest = load_corpus(corpus_path, source_manifest_path, contract)
    case_id = f"s-{stratum}-r01"
    final_dir = out_root / case_id
    if final_dir.exists():
        raise RobustTraceError(f"refusing to overwrite trace case: {final_dir}")

    repository = trace_capture.repository_state(root)
    if not repository.clean:
        raise RobustTraceError(
            f"trace collection requires a clean repository: {repository.status}"
        )
    launch = trace_capture.require_trace_server(server_log)
    image_tag = launch.runtime_images[0]
    image_id = trace_capture.resolve_image_id(root, image_tag)
    runtime_lane = trace_capture.load_runtime_lane(runtime_lane_path)

    resolved_tokenize_url = tokenize_url or workload_contract.derive_tokenize_url(url)
    prompts, prompt_rows, prompt_set_hash = prepare_stratum_prompts(
        corpus,
        stratum=stratum,
        contract=contract,
        url=url,
        model=model,
        tokenize_url=resolved_tokenize_url,
    )
    output_tokens = int(
        contract["strata"][stratum].get(
            "max_output_tokens",
            contract["trace"]["default_max_output_tokens"],
        )
    )

    out_root.mkdir(parents=True, exist_ok=True)
    partial = out_root / f".{case_id}.partial-{os.getpid()}"
    if partial.exists():
        raise RobustTraceError(f"temporary case directory exists: {partial}")
    partial.mkdir()
    started = datetime.now(timezone.utc).isoformat()

    try:
        _write_prompt_artifacts(
            partial,
            prompts,
            prompt_rows,
            prompt_set_sha256=prompt_set_hash,
            tokenize_url=resolved_tokenize_url,
        )
        starts = {
            "ple": trace_capture.file_offset(ple_source),
            "moe": trace_capture.file_offset(moe_source),
            "server": trace_capture.file_offset(server_log),
        }
        records, wall_s = trace_case.run_prompt_benchmark(
            prompts,
            url=url,
            model=model,
            max_tokens=output_tokens,
            temperature=float(contract["trace"]["temperature"]),
            concurrency=1,
        )
        trace_case.write_jsonl(partial / "client.jsonl", records)
        trace_capture.write_json(
            partial / "client.jsonl.summary.json",
            bench_openai.summarize_results(records, wall_s, 1),
        )
        if settle_ms:
            time.sleep(settle_ms / 1000.0)
        ends = {
            "ple": trace_capture.file_offset(ple_source),
            "moe": trace_capture.file_offset(moe_source),
            "server": trace_capture.file_offset(server_log),
        }

        ple_events = trace_capture.capture_jsonl_window(
            ple_source,
            starts["ple"],
            ends["ple"],
            partial / "ple_runtime.jsonl",
        )
        moe_events = trace_capture.capture_jsonl_window(
            moe_source,
            starts["moe"],
            ends["moe"],
            partial / "moe_runtime.jsonl",
        )
        server_bytes = trace_capture.read_byte_window(
            server_log, starts["server"], ends["server"]
        )
        if server_bytes and not server_bytes.endswith(b"\n"):
            raise RobustTraceError("server-log window ends with a partial line")
        server_text = server_bytes.decode("utf-8", errors="replace")
        trace_capture.atomic_write_text(partial / "server-window.log", server_text)

        reasons = validate_client_records_variable(records, prompt_rows, concurrency=1)
        reasons.extend(trace_case.validate_ple_events(ple_events, model_campaign_path))
        if not moe_events:
            reasons.append("missing_moe_events")
        elif any(
            "layer" not in event or not event.get("selected_expert_ids")
            for event in moe_events
        ):
            reasons.append("invalid_moe_event")

        contamination = trace_capture.contamination_hits(server_text)
        fatal = trace_capture.fatal_contamination(contamination)
        reasons.extend(f"fatal_contamination:{item['id']}" for item in fatal)
        if reasons:
            raise RobustTraceError("; ".join(sorted(set(reasons))))

        normalization = trace_case._normalize_exact_case(
            partial, records, model_campaign_path
        )

        completion_tokens = [
            int(record["completion_tokens"]) for record in records if "error" not in record
        ]
        input_tokens = [int(row["rendered_prompt_tokens"]) for row in prompt_rows]
        evidence = {
            "schema_version": 1,
            "evidence_kind": "local_measured_trace",
            "trace_valid": True,
            "performance_eligible": False,
            "can_establish_h3": False,
            "repository_commit": repository.commit,
            "repository_clean": repository.clean,
            "contract_sha256": trace_capture.sha256_file(contract_path),
            "model_campaign_sha256": trace_capture.sha256_file(model_campaign_path),
            "frozen_corpus_sha256": source_manifest["corpus_sha256"],
            "source_manifest_sha256": trace_capture.sha256_file(source_manifest_path),
            "workload_sources": {
                "selection_seed": source_manifest.get("selection_seed"),
                "selection_rule": source_manifest.get("selection_rule"),
                "source_revisions": source_manifest.get("source_revisions"),
                "tool_environment": source_manifest.get("tool_environment"),
            },
            "runtime_lane": runtime_lane,
            "launch": {
                "trace_values": list(launch.trace_values),
                "runtime_image": image_tag,
                "runtime_image_id": image_id,
            },
            "case": {
                "case_id": case_id,
                "experiment": "h1_h2_workload_robustness_v1",
                "stratum": stratum,
                "requests": len(prompts),
                "temperature": float(contract["trace"]["temperature"]),
                "max_output_tokens": output_tokens,
                "concurrency": 1,
                "repetition_index": 1,
                "correlation_mode": "exact_request_correlation",
                "prompt_set_sha256": prompt_set_hash,
                "input_token_min": min(input_tokens),
                "input_token_max": max(input_tokens),
                "input_token_mean": sum(input_tokens) / len(input_tokens),
                "completion_token_min": min(completion_tokens),
                "completion_token_max": max(completion_tokens),
                "completion_token_mean": sum(completion_tokens) / len(completion_tokens),
            },
            "trace_windows": {
                "ple": {
                    "start": starts["ple"],
                    "end": ends["ple"],
                    "records": len(ple_events),
                },
                "moe": {
                    "start": starts["moe"],
                    "end": ends["moe"],
                    "records": len(moe_events),
                },
                "server_log": {"start": starts["server"], "end": ends["server"]},
            },
            "normalization": normalization,
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        evidence["artifacts_sha256"] = trace_case._artifact_hashes(partial)
        trace_capture.write_json(partial / "evidence.json", evidence)
        partial.replace(final_dir)
        return final_dir
    except Exception as exc:
        failure_dir = out_root / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        trace_capture.write_json(
            failure_dir
            / f"{case_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json",
            {
                "case_id": case_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "started_at_utc": started,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        shutil.rmtree(partial, ignore_errors=True)
        raise


def verify_complete(
    out_root: Path,
    contract_path: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    source_manifest = _load_json(source_manifest_path)
    failures = []
    observed = []
    fingerprints = set()
    for stratum in contract["trace"]["strata_order"]:
        case_id = f"s-{stratum}-r01"
        evidence_path = out_root / case_id / "evidence.json"
        if not evidence_path.is_file():
            failures.append(f"missing_case:{case_id}")
            continue
        evidence = _load_json(evidence_path)
        observed.append(case_id)
        if evidence.get("trace_valid") is not True:
            failures.append(f"invalid_case:{case_id}")
        if evidence.get("contract_sha256") != trace_capture.sha256_file(contract_path):
            failures.append(f"contract_hash_mismatch:{case_id}")
        if evidence.get("frozen_corpus_sha256") != source_manifest.get("corpus_sha256"):
            failures.append(f"corpus_hash_mismatch:{case_id}")
        if evidence.get("case", {}).get("stratum") != stratum:
            failures.append(f"stratum_mismatch:{case_id}")
        case_dir = out_root / case_id
        for relative, expected_hash in evidence.get("artifacts_sha256", {}).items():
            artifact = case_dir / relative
            if not artifact.is_file():
                failures.append(f"missing_artifact:{case_id}:{relative}")
            elif trace_capture.sha256_file(artifact) != expected_hash:
                failures.append(f"artifact_hash_mismatch:{case_id}:{relative}")
        fingerprints.add(
            (
                evidence.get("repository_commit"),
                evidence.get("launch", {}).get("runtime_image_id"),
                evidence.get("source_manifest_sha256"),
            )
        )
    if len(fingerprints) > 1:
        failures.append("cross_case_provenance_mismatch")
    return {
        "schema_version": 1,
        "artifact_kind": "fenix_h1_h2_robust_trace_verification",
        "complete": not failures and len(observed) == len(contract["trace"]["strata_order"]),
        "expected_cases": len(contract["trace"]["strata_order"]),
        "observed_cases": len(observed),
        "case_ids": observed,
        "failures": sorted(set(failures)),
    }


def plan(contract_path: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    return {
        "artifact_kind": "fenix_h1_h2_workload_robustness_trace_plan",
        "strata_order": contract["trace"]["strata_order"],
        "cases": [
            {
                "case_id": f"s-{name}-r01",
                "stratum": name,
                "requests": int(contract["strata"][name]["requests"]),
                "max_output_tokens": int(
                    contract["strata"][name].get(
                        "max_output_tokens",
                        contract["trace"]["default_max_output_tokens"],
                    )
                ),
                "render_policy": contract["strata"][name].get("overlength_policy", "native_fail_closed"),
            }
            for name in contract["trace"]["strata_order"]
        ],
        "total_requests": sum(
            int(contract["strata"][name]["requests"])
            for name in contract["trace"]["strata_order"]
        ),
        "server_max_model_len": contract["trace"]["server_max_model_len"],
        "extended_context_feasibility": contract["extended_context_feasibility"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--model-campaign", type=Path, default=DEFAULT_MODEL_CAMPAIGN)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--runtime-lane", type=Path, default=DEFAULT_RUNTIME_LANE)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--ple-trace", type=Path, default=DEFAULT_PLE_TRACE)
    parser.add_argument("--moe-trace", type=Path, default=DEFAULT_MOE_TRACE)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--tokenize-url")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--settle-ms", type=int, default=500)
    parser.add_argument("--stratum")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--verify-complete", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        contract = load_contract(args.contract)
        if args.plan:
            print(json.dumps(plan(args.contract), indent=2))
            return 0
        if args.out_root is None:
            raise RobustTraceError("--out-root is required")
        if args.verify_complete:
            result = verify_complete(
                args.out_root, args.contract, args.source_manifest
            )
            print(json.dumps(result, indent=2))
            return 0 if result["complete"] else 3
        if args.server_log is None:
            raise RobustTraceError("--server-log is required for collection")
        if args.settle_ms < 0:
            raise RobustTraceError("--settle-ms must be >= 0")

        root = Path.cwd().resolve()
        if not (root / ".git").is_dir():
            raise RobustTraceError("run from the FENIX repository root")

        selected_strata = contract["trace"]["strata_order"]
        if args.stratum:
            if args.stratum not in contract["strata"]:
                raise RobustTraceError(f"undeclared stratum: {args.stratum}")
            selected_strata = [args.stratum]

        completed = []
        with trace_capture.campaign_lock(args.out_root / ".trace-campaign.lock"):
            for stratum in selected_strata:
                case_id = f"s-{stratum}-r01"
                existing = args.out_root / case_id / "evidence.json"
                if existing.is_file() and args.resume:
                    evidence = _load_json(existing)
                    source_manifest = _load_json(args.source_manifest)
                    if (
                        evidence.get("trace_valid") is True
                        and evidence.get("contract_sha256")
                        == trace_capture.sha256_file(args.contract)
                        and evidence.get("frozen_corpus_sha256")
                        == source_manifest.get("corpus_sha256")
                        and evidence.get("source_manifest_sha256")
                        == trace_capture.sha256_file(args.source_manifest)
                    ):
                        print(json.dumps({"skipped_valid": case_id}))
                        completed.append(str(args.out_root / case_id))
                        continue
                    raise RobustTraceError(
                        f"{case_id}: --resume found incompatible existing evidence"
                    )
                destination = run_stratum(
                    root=root,
                    stratum=stratum,
                    contract_path=args.contract,
                    model_campaign_path=args.model_campaign,
                    runtime_lane_path=args.runtime_lane,
                    corpus_path=args.corpus,
                    source_manifest_path=args.source_manifest,
                    server_log=args.server_log,
                    out_root=args.out_root,
                    ple_source=args.ple_trace,
                    moe_source=args.moe_trace,
                    url=args.url,
                    model=args.model,
                    tokenize_url=args.tokenize_url,
                    settle_ms=args.settle_ms,
                )
                completed.append(str(destination))
                print(json.dumps({"completed": case_id, "path": str(destination)}))

        verification = verify_complete(
            args.out_root, args.contract, args.source_manifest
        )
        print(
            json.dumps(
                {"completed_cases": completed, "verification": verification},
                indent=2,
            )
        )
        return 0 if verification["complete"] else 3
    except (
        RobustTraceError,
        trace_case.TraceCaseError,
        trace_capture.TraceCaptureError,
        workload_contract.WorkloadContractError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
