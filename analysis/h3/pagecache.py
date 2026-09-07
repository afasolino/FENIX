"""Linux page-cache control helpers for H3."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from analysis.h3.common import GIB, H3Error, load_json, sha256_file, write_json
from analysis.h3.trace import Geometry, iter_case_epochs


def _flatten(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for node in nodes:
        output.append({key: value for key, value in node.items() if key != "children"})
        output.extend(_flatten(node.get("children") or []))
    return output


def _run(command: list[str]) -> str:
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise H3Error(f"command failed: {' '.join(command)}\n{proc.stdout}")
    return proc.stdout


def probe_storage(
    ple_backing: Path,
    expert_backing: Path,
    out_path: Path,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    modern_columns = "NAME,PATH,MODEL,SERIAL,TRAN,SIZE,TYPE,PKNAME,MAJ:MIN,MOUNTPOINTS"
    legacy_columns = "NAME,MODEL,SERIAL,TRAN,SIZE,TYPE,PKNAME,MAJ:MIN,MOUNTPOINT"
    try:
        raw_nodes = json.loads(_run(["lsblk", "-J", "-p", "-o", modern_columns])).get("blockdevices") or []
    except H3Error:
        raw_nodes = json.loads(_run(["lsblk", "-J", "-p", "-o", legacy_columns])).get("blockdevices") or []

    nodes = _flatten(raw_nodes)
    for node in nodes:
        if "path" not in node and node.get("name") is not None:
            node["path"] = node["name"]
        if "mountpoints" not in node:
            mountpoint = node.get("mountpoint")
            node["mountpoints"] = [mountpoint] if mountpoint else []

    def probe(path: Path) -> dict[str, Any]:
        real = path.resolve()
        if not real.is_file():
            raise H3Error(f"backing file is missing: {real}")
        mount = json.loads(
            _run(["findmnt", "-T", str(real), "-J", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS,MAJ:MIN"])
        )["filesystems"][0]
        matches = [node for node in nodes if str(node.get("maj:min")) == str(mount["maj:min"])]
        if len(matches) != 1:
            raise H3Error(f"cannot uniquely resolve block device {mount['maj:min']}")
        stat = real.stat()
        return {
            "path": str(real),
            "size_bytes": stat.st_size,
            "allocated_bytes": stat.st_blocks * 512,
            "mount": mount,
            "block_device": matches[0],
        }

    ple = probe(ple_backing)
    expert = probe(expert_backing)
    geometry_sha = None
    if contract_path is not None:
        contract = load_json(contract_path)
        geometry = contract["geometry"]
        expected_ple = int(geometry["ple_data_bytes"])
        expected_expert = (
            int(geometry["layers"])
            * int(geometry["experts_per_layer"])
            * int(geometry["expert_storage_stride_bytes"])
        )
        if int(ple["size_bytes"]) != expected_ple:
            raise H3Error(f"PLE backing geometry mismatch: {ple['size_bytes']} != {expected_ple}")
        if int(expert["size_bytes"]) != expected_expert:
            raise H3Error(f"expert backing geometry mismatch: {expert['size_bytes']} != {expected_expert}")
        if int(expert["allocated_bytes"]) < int(expected_expert * 0.95):
            raise H3Error("expert backing is sparse/heavily compressed")
        geometry_sha = sha256_file(contract_path)
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_storage_binding",
        "ple": ple,
        "expert": expert,
        "same_filesystem_source": ple["mount"]["source"] == expert["mount"]["source"],
        "same_device_major_minor": ple["mount"]["maj:min"] == expert["mount"]["maj:min"],
        "contract_sha256": geometry_sha,
        "claim_boundary": "fio evidence applies only to the resolved backing files and block-device identity; local storage is not relabelled UFS",
    }
    write_json(out_path, result)
    return result


def build_pagecache_window(
    manifest_path: Path,
    stratum: str,
    capacity_gib: float,
    pressure_factor: float,
    ple_backing: Path,
    expert_backing: Path,
    out_dir: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    contract_sha = (manifest.get("inputs") or {}).get("h3_contract_sha256")
    if not contract_sha:
        raise H3Error("trace manifest lacks exact H3 contract provenance")
    geometry = Geometry(**{key: int(value) for key, value in manifest["geometry"].items()})
    case = next((row for row in manifest["cases"] if row["stratum"] == stratum), None)
    if case is None:
        raise H3Error(f"stratum {stratum} is absent from trace manifest")
    target = int(float(capacity_gib) * GIB * float(pressure_factor))
    case_dir = Path(case["case_dir"])
    for relative, expected in case["source_sha256"].items():
        source = case_dir / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise H3Error(f"source trace drift before page-cache window: {source}")
    ple = ple_backing.resolve()
    expert = expert_backing.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    iolog = (out_dir / "pagecache-logical.iolog").resolve()
    unique_pages: set[int] = set()
    unique_experts: set[tuple[int, int]] = set()
    logical_reads = 0
    logical_bytes = 0

    with iolog.open("w", encoding="utf-8") as stream:
        stream.write("fio version 2 iolog\n")
        stream.write(f"{ple} add\n{ple} open\n{expert} add\n{expert} open\n")
        for epoch in iter_case_epochs(case_dir, geometry):
            for event in epoch.events:
                if event.kind == "ple":
                    for row in event.object_ids:
                        offset = int(row) * geometry.ple_row_bytes
                        first = offset // geometry.storage_page_bytes
                        last = (offset + geometry.ple_row_bytes - 1) // geometry.storage_page_bytes
                        unique_pages.update(range(first, last + 1))
                        stream.write(f"{ple} read {offset} {geometry.ple_row_bytes}\n")
                        logical_reads += 1
                        logical_bytes += geometry.ple_row_bytes
                else:
                    if event.layer is None:
                        raise H3Error("expert event lacks layer")
                    for selected in event.object_ids:
                        key = (int(event.layer), int(selected))
                        unique_experts.add(key)
                        index = key[0] * geometry.experts + key[1]
                        stream.write(
                            f"{expert} read {index * geometry.expert_stride_bytes} {geometry.expert_bytes}\n"
                        )
                        logical_reads += 1
                        logical_bytes += geometry.expert_bytes

    observed_unique = (
        len(unique_pages) * geometry.storage_page_bytes
        + len(unique_experts) * geometry.expert_stride_bytes
    )
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_pagecache_window",
        "stratum": stratum,
        "capacity_gib": float(capacity_gib),
        "pressure_factor": float(pressure_factor),
        "target_unique_bytes": target,
        "observed_unique_bytes": observed_unique,
        "pressure_sufficient": observed_unique >= target,
        "full_trace_replayed": True,
        "terminal_file_close_records_omitted": True,
        "logical_reads": logical_reads,
        "logical_bytes": logical_bytes,
        "unique_ple_pages": len(unique_pages),
        "unique_experts": len(unique_experts),
        "iolog": str(iolog),
        "iolog_sha256": sha256_file(iolog),
        "manifest_sha256": sha256_file(manifest_path),
        "contract_sha256": str(contract_sha),
        "capacity_semantics": "process_plus_pagecache_memcg_envelope_not_exact_pagecache_capacity",
    }
    write_json(out_dir / "window.json", result)
    return result
