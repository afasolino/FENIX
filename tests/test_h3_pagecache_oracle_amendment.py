from __future__ import annotations

import json
from pathlib import Path

from analysis.h3.common import set_execution_repository_identity, sha256_file
from analysis.h3.pagecache_oracle import build_pagecache_oracle, simulate_grouped_belady


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/h3/contract_v6.json"
AMENDMENT = ROOT / "configs/h3/pagecache_oracle_amendment_v1.json"
MEASUREMENT_ID = {
    "head": "e08bc8d32b85ef3fe929d350ca122159fcee4073",
    "tree": "a1af6ceff0e41e0cde20ee0ee2848e7c700d8a48",
    "branch": "study/h3-conventional-hierarchy-v6",
    "clean": True,
    "required_base_commit": "8b8694647840e5af73fd4e1fb1275d0b15c19056",
    "root": str(ROOT),
}
ANALYSIS_ID = {
    "head": "3" * 40,
    "tree": "4" * 40,
    "branch": "study/h3-oracle-test",
    "clean": True,
    "required_base_commit": "8b8694647840e5af73fd4e1fb1275d0b15c19056",
    "root": str(ROOT),
}


def _expanded_belady_misses(sequence: list[int], pages: int, capacity: int) -> int:
    expanded: list[tuple[int, int]] = []
    for key in sequence:
        expanded.extend((key, page) for page in range(pages))
    resident: set[tuple[int, int]] = set()
    misses = 0
    for index, page in enumerate(expanded):
        if page not in resident:
            misses += 1
            resident.add(page)
            if len(resident) > capacity:
                future = expanded[index + 1:]
                def next_use(candidate: tuple[int, int]) -> int:
                    try:
                        return future.index(candidate)
                    except ValueError:
                        return 10**9
                victim = max(resident, key=next_use)
                resident.remove(victim)
    return misses


def test_grouped_belady_matches_expanded_equal_page_min():
    sequence = [0, 1, 2, 0, 3, 1, 0, 2, 3, 0]
    pages = 3
    capacity = 7
    grouped = simulate_grouped_belady(
        sequence, pages_per_object=pages, capacity_pages=capacity
    )
    assert grouped["expert_full_page_misses"] == _expanded_belady_misses(
        sequence, pages, capacity
    )
    assert grouped["peak_resident_pages"] <= capacity


def test_oracle_grants_free_ple_tail_and_binds_frozen_window(tmp_path: Path):
    contract = json.loads(CONTRACT.read_text())
    geometry = contract["geometry"]
    ple = int(geometry["ple_row_bytes"])
    expert = int(geometry["expert_useful_bytes"])
    stride = int(geometry["expert_storage_stride_bytes"])
    iolog = tmp_path / "pagecache.iolog"
    iolog.write_text(
        "fio version 2 iolog\n"
        f"/tmp/ple read 0 {ple}\n"
        f"/tmp/expert read 0 {expert}\n"
        f"/tmp/expert read {stride} {expert}\n"
        f"/tmp/expert read 0 {expert}\n"
    )
    window = tmp_path / "window.json"
    payload = {
        "artifact_kind": "fenix_h3_pagecache_window",
        "schema_version": 6,
        "stratum": "session",
        "capacity_gib": 7.0,
        "full_trace_replayed": True,
        "logical_bytes": ple + 3 * expert,
        "iolog": str(iolog),
        "iolog_sha256": sha256_file(iolog),
        "manifest_sha256": "trace-manifest",
        "contract_sha256": sha256_file(CONTRACT),
        "execution_repository": MEASUREMENT_ID,
    }
    window.write_text(json.dumps(payload))
    out = tmp_path / "oracle.json"
    set_execution_repository_identity(ANALYSIS_ID)
    try:
        result = build_pagecache_oracle(window, CONTRACT, AMENDMENT, 7.0, out)
    finally:
        set_execution_repository_identity(None)
    assert result["ple_storage_traffic_free"] is True
    assert result["expert_partial_tail_free"] is True
    assert result["storage_latency_hidden"] is True
    assert result["conventional_sufficiency_allowed"] is False
    assert result["source_measurement_execution_repository"]["head"] == MEASUREMENT_ID["head"]
    assert result["expert_partial_tail_bytes_ignored_per_access"] == expert % int(geometry["storage_page_bytes"])
    assert json.loads(out.read_text())["execution_repository"] == ANALYSIS_ID
