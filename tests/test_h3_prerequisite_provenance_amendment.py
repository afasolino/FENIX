from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analysis.h3.causal_prerequisite_provenance import _validate_provenance_split
from analysis.h3.common import H3Error
from analysis.h3.placement_invariance_control import build_placement_invariance_control

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/h3/contract_v6.json"
PREREQ_AMENDMENT = ROOT / "configs/h3/prerequisite_provenance_amendment_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _case(root: Path, stratum: str, ids: tuple[str, str], endpoint: str) -> dict[str, str]:
    case = root / stratum
    case.mkdir(parents=True, exist_ok=True)
    prompt_paths = []
    prompt_rows = []
    for ordinal, text in enumerate(("alpha", "beta")):
        path = case / "prompts" / f"{ordinal:04d}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        prompt_paths.append(path)
        prompt_rows.append(
            {
                "ordinal": ordinal,
                "path": str(Path("prompts") / path.name),
                "rendered_prompt_sha256": _sha(path),
                "rendered_prompt_tokens": ordinal + 3,
            }
        )
    prompts = case / "prompts.json"
    _json(prompts, {"tokenize_url": endpoint, "prompts": prompt_rows})

    ple = case / "ple_normalized.jsonl"
    _jsonl(
        ple,
        [
            {
                "request_id": ids[0], "token_position": 0, "ple_head": 0,
                "physical_row_id": 11, "bytes": 160, "phase": "prefill",
                "address_known_ns": 1,
            },
            {
                "request_id": ids[1], "token_position": 0, "ple_head": 0,
                "physical_row_id": 22, "bytes": 160, "phase": "decode",
                "address_known_ns": 2,
            },
        ],
    )
    moe = case / "moe_normalized.jsonl"
    _jsonl(
        moe,
        [
            {
                "request_id": ids[0], "layer": "model.layers.0.mlp",
                "selected_expert_ids": list(range(10)), "phase": "prefill",
                "trace_scope": "router_all_layers", "timestamp_ns": 10,
            },
            {
                "request_id": ids[1], "layer": "model.layers.0.mlp",
                "selected_expert_ids": list(range(10, 20)), "phase": "decode",
                "trace_scope": "router_all_layers", "timestamp_ns": 20,
            },
        ],
    )
    evidence = case / "evidence.json"
    _json(
        evidence,
        {
            "trace_valid": True,
            "repository_clean": True,
            "repository_commit": "8b8694647840e5af73fd4e1fb1275d0b15c19056",
            "source_manifest_sha256": "source",
            "frozen_corpus_sha256": "corpus",
            "runtime_lane": {"lane_id": "same"},
            "launch": {"runtime_image_id": "sha256:" + "1" * 64},
            "case": {"stratum": stratum, "concurrency": 1},
            "artifacts_sha256": {
                "prompts.json": _sha(prompts),
                "ple_normalized.jsonl": _sha(ple),
                "moe_normalized.jsonl": _sha(moe),
            },
        },
    )
    return {
        "prompts": str(prompts),
        "ple_normalized": str(ple),
        "moe_normalized": str(moe),
        "evidence": str(evidence),
    }


def test_placement_control_ignores_run_local_uuid_and_endpoint_metadata(tmp_path: Path):
    strata = ("chat_en", "session", "long_context_8k")
    resident = tmp_path / "resident"
    mmap = tmp_path / "mmap"
    resident_log = tmp_path / "resident.log"
    mmap_log = tmp_path / "mmap.log"
    resident_log.write_text('{"FENIX_PLE_STORAGE_MODE":"resident"}\n')
    mmap_log.write_text(
        '{"FENIX_PLE_STORAGE_MODE":"mmap",'
        '"FENIX_PLE_BANK_MANIFEST":"/fenix-ple-bank/ple.manifest.json"}\n'
    )
    resident_cases = {
        s: _case(resident, s, (f"resident-{s}-a", f"resident-{s}-b"), "http://127.0.0.1:8000/tokenize")
        for s in strata
    }
    mmap_cases = {
        s: _case(mmap, s, (f"mmap-{s}-x", f"mmap-{s}-y"), "http://127.0.0.1:8001/tokenize")
        for s in strata
    }
    control = tmp_path / "input.json"
    _json(
        control,
        {
            "placements": [
                {
                    "name": "resident",
                    "ple_storage_mode": "resident",
                    "server_log": str(resident_log),
                    "cases": resident_cases,
                },
                {
                    "name": "mmap",
                    "ple_storage_mode": "mmap",
                    "server_log": str(mmap_log),
                    "cases": mmap_cases,
                },
            ]
        },
    )
    result = build_placement_invariance_control(control, CONTRACT, tmp_path / "out.json")
    assert result["derived_pass"] is True
    assert result["mismatches"] == []
    assert result["source_runtime_provenance_invariant"] is True
    assert {row["ple_storage_mode"] for row in result["placements"]} == {"resident", "mmap"}


def _identity(head: str, tree: str) -> dict:
    return {
        "head": head,
        "tree": tree,
        "required_base_commit": "8b8694647840e5af73fd4e1fb1275d0b15c19056",
        "clean": True,
    }


def _prereq(path: Path, identity: dict, extra: dict | None = None) -> None:
    payload = {"execution_repository": identity}
    payload.update(extra or {})
    _json(path, payload)


def test_provenance_amendment_allows_only_semantic_analysis_prerequisites(tmp_path: Path):
    frozen = _identity("e08bc8d32b85ef3fe929d350ca122159fcee4073", "frozen-tree")
    analysis = _identity("analysis-head", "analysis-tree")
    files = {
        "placement_invariance": tmp_path / "placement.json",
        "long_context_beyond_8k": tmp_path / "long.json",
        "deployment_capacity_budget": tmp_path / "capacity.json",
        "tool_qualification": tmp_path / "tools.json",
    }
    _prereq(
        files["placement_invariance"],
        analysis,
        {
            "derived_pass": True,
            "evaluator": "ordered_routing_and_ple_semantic_identity_across_physical_placements",
            "source_runtime_provenance_invariant": True,
            "placements": [
                {"ple_storage_mode": "resident"},
                {"ple_storage_mode": "mmap"},
            ],
        },
    )
    _prereq(files["long_context_beyond_8k"], frozen)
    _prereq(files["deployment_capacity_budget"], frozen)
    _prereq(files["tool_qualification"], frozen)
    manifest = tmp_path / "manifest.json"
    _json(
        manifest,
        {
            "evidence": {
                key: {"artifact": str(path), "sha256": _sha(path)}
                for key, path in files.items()
            }
        },
    )
    status = _validate_provenance_split(
        manifest,
        {
            "frozen_measurement_execution_repository": frozen,
            "analysis_execution_repository": analysis,
        },
        PREREQ_AMENDMENT,
        CONTRACT,
    )
    assert status["analysis_derived_prerequisites"] == ["placement_invariance"]

    _prereq(files["tool_qualification"], analysis)
    _json(
        manifest,
        {
            "evidence": {
                key: {"artifact": str(path), "sha256": _sha(path)}
                for key, path in files.items()
            }
        },
    )
    with pytest.raises(H3Error, match="frozen-only prerequisite tool_qualification"):
        _validate_provenance_split(
            manifest,
            {
                "frozen_measurement_execution_repository": frozen,
                "analysis_execution_repository": analysis,
            },
            PREREQ_AMENDMENT,
            CONTRACT,
        )
