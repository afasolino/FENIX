"""Final hostile-bandwidth refinement for the rootless H3 page-cache oracle."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

from analysis.h3.common import H3Error, load_json, write_json
from analysis.h3.decision import _candidate_interval_summary, _service_ns
from analysis.h3.oracle_gate import (
    campaign_gate_with_oracle,
    decide_with_oracle as _base_decide_with_oracle,
)


def _optimistic_point_bandwidth(point: dict) -> float:
    """Return the most favorable measured aggregate bandwidth at one LPDDR point.

    The page-cache dominance oracle changes the traffic mix by eliminating most
    lower-tier fills.  Reusing the original mixed-traffic calibration would
    therefore penalize the oracle for traffic it no longer generates.  For gap
    falsification only, use the maximum of the point-local measured read, write,
    and mixed aggregate bandwidths.  This is intentionally optimistic and is
    never realizable evidence for conventional sufficiency.
    """
    values = [
        float(point.get("aggregate_read_bandwidth_gb_s", 0)),
        float(point.get("aggregate_write_bandwidth_gb_s", 0)),
        float(point.get("aggregate_mixed_total_bandwidth_gb_s", 0)),
    ]
    if any(value <= 0 for value in values):
        raise H3Error("oracle LPDDR point lacks positive read/write/mixed bandwidth")
    return max(values)


def decide_with_oracle(
    residency_path: Path,
    lpddr_path: Path,
    fio_path: Path,
    contract_path: Path,
    queue_depth: int,
    out_path: Path,
    amendment_path: Path,
    actual_lpddr_path: Path | None = None,
    oracle_path: Path | None = None,
    pagecache_paths: Iterable[Path] | None = None,
) -> dict:
    """Run the amended decision using a maximally favorable LPDDR oracle bound."""
    with tempfile.TemporaryDirectory(prefix="fenix-h3-hostile-bw-") as tmp:
        intermediate = Path(tmp) / "decision.json"
        _base_decide_with_oracle(
            residency_path,
            lpddr_path,
            fio_path,
            contract_path,
            int(queue_depth),
            intermediate,
            amendment_path,
            actual_lpddr_path,
            oracle_path,
            pagecache_paths,
        )
        result = load_json(intermediate)

    result["oracle_lpddr_bandwidth_semantics"] = (
        "point_local_max_of_measured_read_write_mixed_aggregate_bandwidths; "
        "gap_falsification_only"
    )

    if oracle_path is not None:
        oracle = load_json(oracle_path)
        miss_bytes = float(oracle["lower_tier_read_bytes"])
        residency = load_json(residency_path)
        read_bytes = float(residency["counters"]["lpddr_read_useful_bytes"])

        for point in result["sensitivity_points"]:
            oracle_bw = _optimistic_point_bandwidth(point)
            all_lpddr = float(point["all_lpddr_oracle_ns"])
            fill = _service_ns(miss_bytes, oracle_bw)
            shared_bus = _service_ns(read_bytes + miss_bytes, oracle_bw)
            lower = max(all_lpddr, shared_bus)
            upper = all_lpddr + fill

            matches = [
                candidate
                for candidate in point["conventional_candidates"]
                if candidate.get("baseline")
                == "future_aware_pagecache_dominance_oracle"
            ]
            if len(matches) != 1:
                raise H3Error(
                    "oracle-aware decision must contain exactly one page-cache oracle candidate"
                )
            candidate = matches[0]
            candidate.update({
                "oracle_lpddr_bandwidth_gb_s": oracle_bw,
                "oracle_lpddr_bandwidth_semantics": (
                    "maximum point-local measured aggregate read/write/mixed bandwidth"
                ),
                "lpddr_fill_service_ns": fill,
                "shared_lpddr_bus_lower_bound_ns": shared_bus,
                "service_ns_bounds": [lower, upper],
            })

            interval = _candidate_interval_summary(
                point["conventional_candidates"], all_lpddr
            )
            point["gap_falsification_penalty_fraction_bounds"] = interval[
                "gap_falsification_penalty_fraction_bounds"
            ]
            point["realizable_penalty_fraction_bounds"] = interval[
                "realizable_penalty_fraction_bounds"
            ]
            point["penalty_fraction_bounds"] = [
                interval["gap_falsification_penalty_fraction_bounds"][0],
                interval["realizable_penalty_fraction_bounds"][1],
            ]

        global_low = min(
            float(point["gap_falsification_penalty_fraction_bounds"][0])
            for point in result["sensitivity_points"]
        )
        global_high = max(
            float(point["realizable_penalty_fraction_bounds"][1])
            for point in result["sensitivity_points"]
        )
        sufficient = float(
            result["thresholds"]["conventional_sufficient_max_penalty_fraction"]
        )
        gap = float(result["thresholds"]["memory_gap_min_penalty_fraction"])
        if global_high <= sufficient:
            verdict = "CONVENTIONAL_MEMORY_SUFFICIENT"
        elif global_low >= gap:
            verdict = "H3_MEMORY_GAP_SUPPORTED"
        else:
            verdict = "INCONCLUSIVE_ESCALATE_INTEGRATED_TIMING"
        result["global_penalty_fraction_bounds"] = [global_low, global_high]
        result["verdict"] = verdict

    write_json(out_path, result)
    return result


__all__ = [
    "_optimistic_point_bandwidth",
    "decide_with_oracle",
    "campaign_gate_with_oracle",
]
