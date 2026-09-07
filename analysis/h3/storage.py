"""Bounded real-storage calibration and uncertainty propagation for H3."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from analysis.h3.common import (
    H3Error, iter_jsonl, load_json, require_same_execution_repository,
    sha256_file, stable_u64, write_json,
)


def _population_bytes(miss_path: Path) -> dict[str, int]:
    totals = {"mixed": 0, "ple": 0, "expert": 0}
    records = {"mixed": 0, "ple": 0, "expert": 0}
    for row in iter_jsonl(miss_path):
        kind = str(row["kind"])
        length = int(row["length_bytes"])
        if kind not in ("ple", "expert"):
            raise H3Error(f"unsupported miss kind {kind}")
        totals["mixed"] += length
        totals[kind] += length
        records["mixed"] += 1
        records[kind] += 1
    if totals["ple"] == 0 or totals["expert"] == 0:
        raise H3Error("storage calibration requires both PLE and expert misses")
    return {
        **{f"{key}_bytes": value for key, value in totals.items()},
        **{f"{key}_records": value for key, value in records.items()},
    }


def _verify_binding(
    binding_path: Path, ple_backing: Path, expert_backing: Path
) -> dict[str, Any]:
    binding = load_json(binding_path)
    if binding.get("artifact_kind") != "fenix_h3_storage_binding":
        raise H3Error("unexpected storage-binding artifact")
    if str(binding["ple"]["path"]) != str(ple_backing.resolve()):
        raise H3Error("PLE backing does not match storage binding")
    if str(binding["expert"]["path"]) != str(expert_backing.resolve()):
        raise H3Error("expert backing does not match storage binding")
    if not binding.get("same_device_major_minor"):
        raise H3Error(
            "mixed fio validation requires PLE and expert backings on one resolved block device"
        )
    return binding


def _request_blocks(miss_path: Path, seeds: list[int]) -> list[dict[str, Any]]:
    """Return deterministic non-overlapping blocks of complete source requests.

    H3 source concurrency is one, so a request is the natural blocking unit for
    storage uncertainty: it preserves session order and never cuts a request in
    half. PLE, expert and mixed samples then share exactly the same request block.
    """
    if not seeds:
        raise H3Error("storage calibration requires at least one window seed")
    requests: list[dict[str, Any]] = []
    current_id: str | None = None
    start_record = 0
    last_closed: set[str] = set()
    total_records = 0
    for record_index, row in enumerate(iter_jsonl(miss_path)):
        request_id = str(row.get("request_id", ""))
        if not request_id:
            raise H3Error("v5 storage sampling requires request_id in every miss record")
        if current_id is None:
            current_id = request_id
            start_record = record_index
        elif request_id != current_id:
            requests.append({"request_id": current_id, "start_record": start_record, "end_record": record_index})
            last_closed.add(current_id)
            if request_id in last_closed:
                raise H3Error("source miss stream re-enters a completed request; concurrency=1 request blocking is invalid")
            current_id = request_id
            start_record = record_index
        total_records = record_index + 1
    if current_id is not None:
        requests.append({"request_id": current_id, "start_record": start_record, "end_record": total_records})
    if not requests:
        raise H3Error("storage calibration miss stream is empty")
    unique_seeds = list(dict.fromkeys(int(seed) for seed in seeds))
    window_count = min(len(unique_seeds), len(requests))
    blocks: list[dict[str, Any]] = []
    for index, seed in enumerate(unique_seeds[:window_count]):
        request_begin = (index * len(requests)) // window_count
        request_end = ((index + 1) * len(requests)) // window_count
        selected = requests[request_begin:request_end]
        if not selected:
            continue
        blocks.append({
            "seed": seed,
            "block_index": index,
            "request_start_index": request_begin,
            "request_end_index": request_end,
            "request_ids": [row["request_id"] for row in selected],
            "start_record": int(selected[0]["start_record"]),
            "end_record": int(selected[-1]["end_record"]),
        })
    if not blocks:
        raise H3Error("storage calibration could not form a complete-request block")
    return blocks


def build_samples(
    miss_path: Path,
    ple_backing: Path,
    expert_backing: Path,
    binding_path: Path,
    out_dir: Path,
    seeds: Iterable[int],
    ple_target_bytes: int,
    expert_target_bytes: int,
    mixed_target_bytes: int,
    contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Build genuinely paired fio samples from common source-sequence windows.

    Each block is non-overlapping in the original miss stream.  PLE and expert
    sub-samples are capped independently inside that block; the mixed sample is
    exactly the source-order union of those admitted operations.  Therefore a
    mixed interaction measurement can be compared to same-window class service
    without inventing pairing from equal seed labels.
    """
    binding = _verify_binding(binding_path, ple_backing, expert_backing)
    population = _population_bytes(miss_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_list = [int(seed) for seed in seeds]
    blocks = _request_blocks(miss_path, seed_list)
    targets = {"ple": int(ple_target_bytes), "expert": int(expert_target_bytes)}
    expected_mixed_target = targets["ple"] + targets["expert"]
    if int(mixed_target_bytes) != expected_mixed_target:
        raise H3Error(
            "paired mixed sample target must equal ple_target_bytes + expert_target_bytes "
            f"({mixed_target_bytes} != {expected_mixed_target})"
        )

    ple = ple_backing.resolve()
    expert = expert_backing.resolve()
    state: dict[int, dict[str, Any]] = {}
    handles: dict[tuple[int, str], Any] = {}
    for block in blocks:
        seed = int(block["seed"])
        window_id = f"block{int(block['block_index']):02d}-r{int(block['start_record'])}-{int(block['end_record'])}"
        state[seed] = {
            **block,
            "window_id": window_id,
            "selected_bytes": {"ple": 0, "expert": 0},
            "records": {"ple": 0, "expert": 0},
            "first_sequence": None,
            "last_sequence": None,
        }
        for kind in ("ple", "expert", "mixed"):
            iolog = (out_dir / f"{kind}-seed{seed}.iolog").resolve()
            stream = iolog.open("w", encoding="utf-8")
            stream.write("fio version 2 iolog\n")
            if kind in ("ple", "mixed"):
                stream.write(f"{ple} add\n{ple} open\n")
            if kind in ("expert", "mixed"):
                stream.write(f"{expert} add\n{expert} open\n")
            handles[(seed, kind)] = stream
            state[seed][f"{kind}_iolog"] = str(iolog)

    block_index = 0
    try:
        for record_index, row in enumerate(iter_jsonl(miss_path)):
            while block_index + 1 < len(blocks) and record_index >= int(blocks[block_index]["end_record"]):
                block_index += 1
            block = blocks[block_index]
            if not (int(block["start_record"]) <= record_index < int(block["end_record"])):
                continue
            seed = int(block["seed"])
            entry = state[seed]
            row_kind = str(row["kind"])
            if row_kind not in ("ple", "expert"):
                raise H3Error(f"unsupported miss kind {row_kind}")
            length = int(row["length_bytes"])
            if int(entry["selected_bytes"][row_kind]) >= targets[row_kind]:
                continue
            backing = ple if row_kind == "ple" else expert
            line = f"{backing} read {int(row['offset_bytes'])} {length}\n"
            handles[(seed, row_kind)].write(line)
            handles[(seed, "mixed")].write(line)
            entry["selected_bytes"][row_kind] += length
            entry["records"][row_kind] += 1
            sequence = int(row["sequence"])
            if entry["first_sequence"] is None:
                entry["first_sequence"] = sequence
            entry["last_sequence"] = sequence
    finally:
        # Do not emit terminal fio "close" records. With asynchronous
        # read_iolog replay, fio 3.42 can process the terminal close while
        # QD-1 reads remain in flight: total_ios still counts them, but
        # read.io_bytes omits their completions. Natural job teardown closes
        # the files after outstanding I/O is drained.
        for stream in handles.values():
            stream.close()

    samples: list[dict[str, Any]] = []
    seen_ranges: set[tuple[int, int]] = set()
    for block in blocks:
        seed = int(block["seed"])
        entry = state[seed]
        source_range = (int(block["start_record"]), int(block["end_record"]))
        if source_range in seen_ranges:
            raise H3Error("storage sampling produced a duplicate source window")
        seen_ranges.add(source_range)
        if int(entry["selected_bytes"]["ple"]) <= 0 or int(entry["selected_bytes"]["expert"]) <= 0:
            raise H3Error(
                f"paired source window {entry['window_id']} lacks one conditional-state class"
            )
        for kind in ("ple", "expert", "mixed"):
            if kind == "mixed":
                bytes_by_kind = {
                    "ple": int(entry["selected_bytes"]["ple"]),
                    "expert": int(entry["selected_bytes"]["expert"]),
                }
                selected_bytes = sum(bytes_by_kind.values())
                records = int(entry["records"]["ple"]) + int(entry["records"]["expert"])
                target = expected_mixed_target
            else:
                bytes_by_kind = {
                    "ple": int(entry["selected_bytes"]["ple"]) if kind == "ple" else 0,
                    "expert": int(entry["selected_bytes"]["expert"]) if kind == "expert" else 0,
                }
                selected_bytes = int(entry["selected_bytes"][kind])
                records = int(entry["records"][kind])
                target = targets[kind]
            iolog = Path(entry[f"{kind}_iolog"])
            samples.append({
                "sample_id": f"{kind}-seed{seed}",
                "kind": kind,
                "mode": "paired_nonoverlapping_complete_request_block",
                "seed": seed,
                "window_id": entry["window_id"],
                "records": records,
                "bytes": selected_bytes,
                "bytes_by_kind": bytes_by_kind,
                "window": {
                    "block_index": int(block["block_index"]),
                    "source_record_start_inclusive": int(block["start_record"]),
                    "source_record_end_exclusive": int(block["end_record"]),
                    "source_record_count": int(block["end_record"]) - int(block["start_record"]),
                    "request_start_index": int(block["request_start_index"]),
                    "request_end_index": int(block["request_end_index"]),
                    "request_ids": list(block["request_ids"]),
                    "population_limited": selected_bytes < target,
                    "target_bytes": int(target),
                    "first_sequence": entry["first_sequence"],
                    "last_sequence": entry["last_sequence"],
                },
                "iolog": str(iolog),
                "iolog_sha256": sha256_file(iolog),
            })

    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_storage_samples",
        "miss_stream": str(miss_path.resolve()),
        "miss_stream_sha256": sha256_file(miss_path),
        "storage_binding": str(binding_path.resolve()),
        "storage_binding_sha256": sha256_file(binding_path),
        "device_major_minor": binding["ple"]["mount"]["maj:min"],
        "population": population,
        "sampling_design": {
            "method": "nonoverlapping_complete_request_blocks_with_paired_class_filters",
            "requested_window_seeds": seed_list,
            "realized_window_count": len(blocks),
            "window_ranges_unique": True,
            "class_pairing": "ple/expert/mixed derive from the identical contiguous complete-request block; mixed is exact source-order union of admitted class operations",
            "mixed_target_equals_class_target_sum": True,
            "terminal_file_close_records_omitted": True,
        },
        "samples": sorted(samples, key=lambda row: row["sample_id"]),
        "contract_sha256": contract_sha256,
    }
    write_json(out_dir / "samples.json", result)
    return result

def fio_runtime_ns(payload: dict[str, Any]) -> int:
    jobs = payload.get("jobs") or []
    if not jobs:
        raise H3Error("fio JSON has no jobs")
    runtime_ms = max(int(job.get("job_runtime", 0)) for job in jobs)
    if runtime_ms <= 0:
        raise H3Error("fio job_runtime is missing or zero")
    return runtime_ms * 1_000_000


def fio_read_bytes(payload: dict[str, Any]) -> int:
    total = 0
    for job in payload.get("jobs") or []:
        read = job.get("read") or {}
        if read.get("io_bytes") is not None:
            total += int(read["io_bytes"])
        elif read.get("io_kbytes") is not None:
            total += int(read["io_kbytes"]) * 1024
        else:
            raise H3Error("fio JSON does not report read bytes")
    return total


def fio_achieved_qd_fraction(
    payload: dict[str, Any], qd: int, threshold_fraction: float = 0.5
) -> float:
    if int(qd) <= 1:
        return 1.0
    threshold = max(1, int(math.ceil(int(qd) * float(threshold_fraction))))
    levels = {"1": 1, "2": 2, "4": 4, "8": 8, "16": 16, "32": 32, ">=64": 64}
    fractions: list[float] = []
    for job in payload.get("jobs") or []:
        hist = job.get("iodepth_level") or {}
        if not hist:
            raise H3Error("fio JSON lacks iodepth_level for QD qualification")
        pct = sum(float(value) for key, value in hist.items() if levels.get(str(key), 0) >= threshold)
        fractions.append(min(1.0, max(0.0, pct / 100.0)))
    if not fractions:
        raise H3Error("fio JSON has no jobs for QD qualification")
    return min(fractions)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise H3Error("cannot take quantile of empty values")
    index = int(round(q * (len(sorted_values) - 1)))
    return float(sorted_values[max(0, min(len(sorted_values) - 1, index))])


def hierarchical_bootstrap_median_ci(
    groups: Mapping[str, list[float]], seed: int, draws: int = 10000
) -> tuple[float, float, float]:
    cleaned = {str(key): [float(value) for value in values] for key, values in groups.items() if values}
    if not cleaned:
        raise H3Error("cannot summarize empty hierarchical samples")
    window_centers = [median(values) for values in cleaned.values()]
    center = float(median(window_centers))
    if len(cleaned) == 1 and len(next(iter(cleaned.values()))) == 1:
        return center, center, center
    rng = random.Random(seed)
    keys = sorted(cleaned)
    boot: list[float] = []
    for _ in range(int(draws)):
        sampled_centers: list[float] = []
        for _window in keys:
            key = keys[rng.randrange(len(keys))]
            values = cleaned[key]
            # Resample the run replicates within the selected spatial window.
            replicate_draw = [values[rng.randrange(len(values))] for _ in values]
            sampled_centers.append(float(median(replicate_draw)))
        boot.append(float(median(sampled_centers)))
    boot.sort()
    return center, _quantile(boot, 0.025), _quantile(boot, 0.975)


def _verified_fio_result(
    result_path: Path,
    sample: dict[str, Any],
    samples_payload: dict[str, Any],
    qd: int,
    samples_sha256: str,
    repeat_index: int,
    minimum_achieved_qd_fraction: float,
    achieved_qd_threshold_fraction: float,
) -> dict[str, Any]:
    meta_path = result_path.with_suffix(".meta.json")
    if not result_path.is_file() or not meta_path.is_file():
        raise H3Error(
            f"missing fio result/provenance for {sample['sample_id']} qd={qd} rep={repeat_index}"
        )
    meta = load_json(meta_path)
    require_same_execution_repository(
        ("fio sample manifest", samples_payload),
        ("fio run provenance", meta),
    )
    if int(meta.get("returncode", -1)) != 0:
        raise H3Error(f"fio run failed for {sample['sample_id']} qd={qd} rep={repeat_index}")
    if int(meta.get("queue_depth", -1)) != qd:
        raise H3Error("fio queue-depth provenance mismatch")
    if int(meta.get("repeat_index", -1)) != int(repeat_index):
        raise H3Error("fio repeat-index provenance mismatch")
    if meta.get("samples_sha256") != samples_sha256:
        raise H3Error("fio sample-manifest SHA mismatch")
    if meta.get("iolog_sha256") != sample["iolog_sha256"]:
        raise H3Error("fio iolog SHA mismatch")
    tool = meta.get("fio_tool") or {}
    if len(str(tool.get("head", ""))) != 40 or tool.get("clean") is not True:
        raise H3Error("fio exact tool provenance is missing or unqualified")
    if not tool.get("version") or not tool.get("binary_sha256"):
        raise H3Error("fio version/binary provenance is incomplete")
    iolog = Path(sample["iolog"])
    if sha256_file(iolog) != sample["iolog_sha256"]:
        raise H3Error("fio iolog changed after sampling")
    payload = load_json(result_path)
    observed_bytes = fio_read_bytes(payload)
    if observed_bytes != int(sample["bytes"]):
        raise H3Error("fio read byte count differs from sample byte count")
    if int(meta.get("actual_read_bytes", -1)) != observed_bytes:
        raise H3Error("fio provenance actual_read_bytes mismatch")
    achieved = fio_achieved_qd_fraction(payload, qd, achieved_qd_threshold_fraction)
    if achieved + 1e-12 < float(minimum_achieved_qd_fraction):
        raise H3Error(
            f"fio configured QD{qd} was not achieved: fraction={achieved:.4f} "
            f"< {minimum_achieved_qd_fraction:.4f}"
        )
    return payload


def summarize_fio(
    samples_path: Path,
    runs_dir: Path,
    queue_depths: Iterable[int],
    out_path: Path,
    runs_per_sample: int = 1,
    minimum_achieved_qd_fraction: float = 0.0,
    achieved_qd_threshold_fraction: float = 0.5,
    bootstrap_draws: int = 10000,
    preferred_common_windows: int = 5,
    target_common_windows: int = 8,
) -> dict[str, Any]:
    payload = load_json(samples_path)
    samples = list(payload["samples"])
    samples_sha = sha256_file(samples_path)
    qds = [int(value) for value in queue_depths]

    by_window: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for sample in samples:
        window_id = str(sample.get("window_id", ""))
        if not window_id:
            raise H3Error("fio sample lacks paired window_id")
        kind = str(sample["kind"])
        if kind in by_window[window_id]:
            raise H3Error(f"duplicate {kind} fio sample in window {window_id}")
        by_window[window_id][kind] = sample
    if len(by_window) < 3:
        raise H3Error("fio calibration requires at least three distinct paired source windows")
    source_ranges: set[tuple[int, int]] = set()
    for window_id, kinds in sorted(by_window.items()):
        if set(kinds) != {"ple", "expert", "mixed"}:
            raise H3Error(f"paired fio window {window_id} does not contain ple/expert/mixed samples")
        ranges = {
            (
                int(sample["window"]["source_record_start_inclusive"]),
                int(sample["window"]["source_record_end_exclusive"]),
            )
            for sample in kinds.values()
        }
        if len(ranges) != 1:
            raise H3Error(f"fio window {window_id} class samples do not share one source interval")
        source_range = next(iter(ranges))
        if source_range in source_ranges:
            raise H3Error("fio sample manifest contains duplicate source intervals")
        source_ranges.add(source_range)
        mixed = kinds["mixed"]
        if int(mixed["bytes_by_kind"]["ple"]) != int(kinds["ple"]["bytes"]):
            raise H3Error(f"mixed PLE bytes are not paired to PLE sample in {window_id}")
        if int(mixed["bytes_by_kind"]["expert"]) != int(kinds["expert"]["bytes"]):
            raise H3Error(f"mixed expert bytes are not paired to expert sample in {window_id}")
        if int(mixed["bytes"]) != int(kinds["ple"]["bytes"]) + int(kinds["expert"]["bytes"]):
            raise H3Error(f"mixed bytes are not exact paired class union in {window_id}")

    class_groups: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    achieved_qd: dict[str, list[float]] = defaultdict(list)

    def result_path(sample_id: str, qd: int, rep: int) -> Path:
        return runs_dir / f"{sample_id}-qd{qd}-rep{rep:02d}.json"

    for sample in samples:
        if sample["kind"] == "mixed":
            continue
        window_id = str(sample["window_id"])
        for qd in qds:
            for rep in range(int(runs_per_sample)):
                result = _verified_fio_result(
                    result_path(sample["sample_id"], qd, rep),
                    sample,
                    payload,
                    qd,
                    samples_sha,
                    rep,
                    minimum_achieved_qd_fraction,
                    achieved_qd_threshold_fraction,
                )
                class_groups[(sample["kind"], qd)][window_id].append(
                    fio_runtime_ns(result) / int(sample["bytes"])
                )
                achieved_qd[f"{sample['sample_id']}:qd{qd}"].append(
                    fio_achieved_qd_fraction(result, qd, achieved_qd_threshold_fraction)
                )

    classes: dict[str, Any] = {}
    for (kind, qd), groups in sorted(class_groups.items()):
        center, low, high = hierarchical_bootstrap_median_ci(
            groups, 20260906 + qd + (0 if kind == "ple" else 100), bootstrap_draws
        )
        values = [value for group in groups.values() for value in group]
        classes[f"{kind}:qd{qd}"] = {
            "windows": len(groups),
            "runs_per_window": int(runs_per_sample),
            "replicates": len(values),
            "ns_per_byte_median": center,
            "ns_per_byte_ci95": [low, high],
            "replicate_values": values,
            "replicate_groups": {key: list(values) for key, values in sorted(groups.items())},
            "window_median_ns_per_byte": {
                key: float(median(values)) for key, values in sorted(groups.items())
            },
            "uncertainty_method": "hierarchical_bootstrap_nonoverlapping_source_windows_then_run_replicates",
        }

    mixed_groups: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for window_id, kinds in sorted(by_window.items()):
        sample = kinds["mixed"]
        for qd in qds:
            ple_window = class_groups[("ple", qd)][window_id]
            expert_window = class_groups[("expert", qd)][window_id]
            ple_coeff = float(median(ple_window))
            expert_coeff = float(median(expert_window))
            predicted = (
                int(sample["bytes_by_kind"]["ple"]) * ple_coeff
                + int(sample["bytes_by_kind"]["expert"]) * expert_coeff
            )
            if predicted <= 0:
                raise H3Error("mixed fio same-window independent projection is zero")
            for rep in range(int(runs_per_sample)):
                result = _verified_fio_result(
                    result_path(sample["sample_id"], qd, rep),
                    sample,
                    payload,
                    qd,
                    samples_sha,
                    rep,
                    minimum_achieved_qd_fraction,
                    achieved_qd_threshold_fraction,
                )
                mixed_groups[qd][window_id].append(fio_runtime_ns(result) / predicted)
                achieved_qd[f"{sample['sample_id']}:qd{qd}"].append(
                    fio_achieved_qd_fraction(result, qd, achieved_qd_threshold_fraction)
                )

    mixed: dict[str, Any] = {}
    for qd in qds:
        groups = mixed_groups[qd]
        center, low, high = hierarchical_bootstrap_median_ci(
            groups, 20261000 + qd, bootstrap_draws
        )
        values = [value for group in groups.values() for value in group]
        mixed[f"qd{qd}"] = {
            "windows": len(groups),
            "runs_per_window": int(runs_per_sample),
            "replicates": len(values),
            "observed_over_same_window_independent_projection_median": center,
            # Compatibility alias retained in the artifact, but its denominator
            # is now explicitly same-window rather than global class medians.
            "observed_over_independent_projection_median": center,
            "ci95": [low, high],
            "replicate_values": values,
            "replicate_groups": {key: list(values) for key, values in sorted(groups.items())},
            "interaction_denominator": "same-window median PLE/expert class service coefficients",
            "uncertainty_method": "hierarchical_bootstrap_nonoverlapping_source_windows_then_run_replicates",
        }

    tool_fingerprints: dict[str, dict[str, Any]] = {}
    for sample in samples:
        for qd in qds:
            for rep in range(int(runs_per_sample)):
                meta = load_json(result_path(sample["sample_id"], qd, rep).with_suffix(".meta.json"))
                tool = dict(meta.get("fio_tool") or {})
                key = str(tool.get("head", "")) + ":" + str(tool.get("binary_sha256", ""))
                tool_fingerprints[key] = tool
    if len(tool_fingerprints) != 1:
        raise H3Error("fio runs do not share one exact qualified binary/tool identity")

    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_fio_calibration",
        "class_coefficients": classes,
        "mixed_interaction": mixed,
        "fio_tool": next(iter(tool_fingerprints.values())),
        "runs_per_sample": int(runs_per_sample),
        "run_variability_separated_from_window_variability": int(runs_per_sample) > 1,
        "achieved_queue_depth_fraction": dict(sorted(achieved_qd.items())),
        "minimum_achieved_qd_fraction": float(minimum_achieved_qd_fraction),
        "achieved_qd_threshold_fraction": float(achieved_qd_threshold_fraction),
        "bootstrap_draws": int(bootstrap_draws),
        "paired_window_design": {
            "verified": True,
            "distinct_nonoverlapping_windows": len(by_window),
            "window_ids": sorted(by_window),
            "source_ranges": [list(value) for value in sorted(source_ranges)],
            "mixed_is_exact_union_of_paired_class_samples": True,
            "minimum_windows": 3,
            "preferred_windows": int(preferred_common_windows),
            "target_windows": int(target_common_windows),
            "statistical_strength": (
                "target" if len(by_window) >= int(target_common_windows)
                else "preferred" if len(by_window) >= int(preferred_common_windows)
                else "minimum_only"
            ),
        },
        "contract_sha256": payload.get("contract_sha256"),
        "provenance": {
            "samples_sha256": samples_sha,
            "miss_stream_sha256": payload["miss_stream_sha256"],
            "storage_binding_sha256": payload["storage_binding_sha256"],
            "device_major_minor": payload.get("device_major_minor"),
            "contract_sha256": payload.get("contract_sha256"),
        },
    }
    write_json(out_path, result)
    return result

def _flatten_groups(node: dict[str, Any]) -> list[float]:
    groups = node.get("replicate_groups") or {}
    values = [float(value) for group in groups.values() for value in group]
    if not values:
        values = [float(value) for value in node.get("replicate_values") or []]
    if not values:
        raise H3Error("fio calibration lacks empirical replicate values")
    return values


def _grouped_values(node: dict[str, Any]) -> dict[str, list[float]]:
    groups = node.get("replicate_groups") or {}
    out = {
        str(key): [float(value) for value in values]
        for key, values in groups.items()
        if values
    }
    if not out:
        raise H3Error("fio calibration lacks hierarchical replicate groups")
    return out


def project_storage_bytes_ns(
    ple_bytes: float, expert_bytes: float, fio: dict[str, Any], qd: int, draws: int | None = None
) -> dict[str, Any]:
    """Hierarchical joint bootstrap of the final projected storage service.

    One common spatial window/seed is sampled first, then one physical replicate
    for PLE, expert and mixed interaction from that same window.  This preserves
    the experiment hierarchy and the available pairing rather than flattening
    three marginal empirical distributions.
    """
    paired = fio.get("paired_window_design") or {}
    if paired.get("verified") is not True:
        raise H3Error("storage projection requires verified paired non-overlapping fio windows")
    ple = fio["class_coefficients"][f"ple:qd{qd}"]
    expert = fio["class_coefficients"][f"expert:qd{qd}"]
    interaction = fio["mixed_interaction"][f"qd{qd}"]
    ple_groups = _grouped_values(ple)
    expert_groups = _grouped_values(expert)
    mix_groups = _grouped_values(interaction)
    common = sorted(set(ple_groups) & set(expert_groups) & set(mix_groups))
    if len(common) < 3:
        raise H3Error(
            "fio calibration needs at least three common spatial windows for joint projection"
        )
    n_draws = int(draws or fio.get("bootstrap_draws", 10000))
    rng = random.Random(stable_u64(20260906, qd, int(ple_bytes), int(expert_bytes), "storage-projection-v4"))
    samples: list[float] = []
    for _ in range(n_draws):
        window = common[rng.randrange(len(common))]
        pvals = ple_groups[window]
        evals = expert_groups[window]
        mvals = mix_groups[window]
        p = pvals[rng.randrange(len(pvals))]
        e = evals[rng.randrange(len(evals))]
        m = mvals[rng.randrange(len(mvals))]
        samples.append((float(ple_bytes) * p + float(expert_bytes) * e) * m)
    samples.sort()
    return {
        "median_ns": _quantile(samples, 0.5),
        "low_ns": _quantile(samples, 0.025),
        "high_ns": _quantile(samples, 0.975),
        "draws": n_draws,
        "common_spatial_windows": common,
        "window_pairing_preserved": True,
        "uncertainty_method": "hierarchical_joint_bootstrap_common_window_then_within_window_replicates",
    }


def project_storage_ns(
    residency: dict[str, Any], fio: dict[str, Any], qd: int, draws: int | None = None
) -> dict[str, Any]:
    counters = residency["counters"]
    return project_storage_bytes_ns(
        float(counters.get("ple_storage_bytes", 0)),
        float(counters.get("expert_storage_bytes", 0)),
        fio, qd, draws,
    )
