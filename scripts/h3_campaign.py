#!/usr/bin/env python3
"""FENIX H3 conventional LPDDR5 + storage campaign CLI."""
from __future__ import annotations

import argparse
import ctypes
import mmap
import json
import os
import resource
import random
import shutil
import subprocess
import sys
import threading
import tomllib
import time
import uuid
import venv
from pathlib import Path
from typing import Any

from analysis.h3.common import (
    GIB, H3Error, load_json, set_execution_repository_identity, sha256_file, write_json,
)
from analysis.h3.decision import campaign_gate, decide_point
from analysis.h3.pagecache import build_pagecache_window, probe_storage
from analysis.h3.prerequisites import (
    build_capacity_budget, build_long_context_scaling, build_placement_invariance,
    build_storage_concurrency_evidence,
)
from analysis.h3.residency import replay_case
from analysis.h3.storage import build_samples, summarize_fio
from analysis.h3.trace import build_manifest
from analysis.h3.lpddr import (
    offered_frontend_gb_s, run_actual_h3_trace_point, run_address_trace_point,
    run_stream_point, theoretical_channel_gb_s, validate_resolved_profile,
)
from analysis.h3.tools import fio_identity, git_checkout_identity

DEFAULT_CONTRACT = Path("configs/h3/contract_v6.json")
DEFAULT_LOCK = Path("configs/h3/upstream_tools.lock.json")
H3_BASE_COMMIT = "8b8694647840e5af73fd4e1fb1275d0b15c19056"



def _execution_repository_identity(
    root: Path | None = None,
    required_base: str = H3_BASE_COMMIT,
) -> dict[str, Any]:
    """Require a clean committed H3 implementation descended from the H1/H2 base."""
    cwd = (root or Path.cwd()).resolve()

    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            ["git", *args], cwd=cwd, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if check and proc.returncode:
            raise H3Error(f"FENIX git command failed: git {' '.join(args)}\n{proc.stdout}")
        return proc

    top = Path(git("rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top != cwd:
        raise H3Error(f"run H3 campaign commands from the FENIX repository root: {top}")
    head = git("rev-parse", "HEAD").stdout.strip()
    tree = git("rev-parse", "HEAD^{tree}").stdout.strip()
    status = git("status", "--porcelain").stdout.strip()
    if status:
        raise H3Error("H3 campaign requires a clean committed FENIX implementation")
    ancestor = git("merge-base", "--is-ancestor", required_base, head, check=False)
    if ancestor.returncode != 0:
        raise H3Error(f"H3 implementation HEAD {head} does not descend from base {required_base}")
    branch_proc = git("symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    branch = branch_proc.stdout.strip() if branch_proc.returncode == 0 else None
    return {
        "root": str(top),
        "head": head,
        "tree": tree,
        "branch": branch,
        "clean": True,
        "required_base_commit": required_base,
    }

def _run(command: list[str | Path], cwd: Path | None = None) -> str:
    printable = [str(item) for item in command]
    print("+", " ".join(printable), flush=True)
    proc = subprocess.run(
        printable,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, end="")
    if proc.returncode:
        raise H3Error(f"command failed rc={proc.returncode}: {' '.join(printable)}")
    return proc.stdout.strip()


def _git(path: Path, *args: str) -> str:
    return _run(["git", *args], cwd=path)


def _checkout(root: Path, spec: dict[str, Any]) -> Path:
    path = root / spec["path"]
    fresh = not path.exists()
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A --no-checkout clone reports every tracked file as deleted and thus
        # fails the clean-tree gate. Clone a normal worktree, then detach-pin.
        _run(["git", "clone", spec["repository"], path], cwd=root)
    if not (path / ".git").exists():
        raise H3Error(f"mature-tool path is not a git checkout: {path}")
    origin = _git(path, "remote", "get-url", "origin").rstrip("/")
    if origin != str(spec["repository"]).rstrip("/"):
        raise H3Error(f"origin mismatch for {path}: {origin}")
    if _git(path, "status", "--porcelain"):
        raise H3Error(f"dirty mature-tool checkout: {path}")
    head = _git(path, "rev-parse", "HEAD")
    if head != spec["commit"]:
        _git(path, "fetch", "--force", "origin", spec["commit"])
        _git(path, "checkout", "--detach", spec["commit"])
    if _git(path, "rev-parse", "HEAD") != spec["commit"]:
        raise H3Error(f"failed to pin mature tool {path}")
    if _git(path, "status", "--porcelain"):
        raise H3Error(f"mature-tool checkout became dirty after pinning: {path}")
    return path


def _project_runtime_dependencies(project_root: Path) -> list[str]:
    """Return the pinned project's declared Python runtime dependencies."""
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        raise H3Error(f"pinned Python project lacks pyproject.toml: {project_root}")
    payload = tomllib.loads(pyproject.read_text())
    raw = list((payload.get("project") or {}).get("dependencies") or [])
    if not all(isinstance(value, str) and value.strip() for value in raw):
        raise H3Error(f"invalid project.dependencies in {pyproject}")
    return [str(value).strip() for value in raw]


def bootstrap_tools(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd()
    lock = load_json(args.lock)
    tools = lock["tools"]
    ramulator = _checkout(root, tools["ramulator2"])
    fio = _checkout(root, tools["fio"])

    ram_tools = root / ".runtime/h3-tools"
    ram_tools.mkdir(parents=True, exist_ok=True)
    env_lock = ram_tools / "ramulator-python-resolved.lock.txt"
    if not env_lock.exists():
        resolver = ram_tools / "ramulator-resolver-venv"
        if resolver.exists():
            shutil.rmtree(resolver)
        venv.EnvBuilder(with_pip=True).create(resolver)
        resolver_python = resolver / "bin/python"
        _run([resolver_python, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], cwd=ramulator)
        _run([resolver_python, "-m", "pip", "install", "-r", "requirements-dev.txt"], cwd=ramulator)
        runtime_dependencies = _project_runtime_dependencies(ramulator)
        if runtime_dependencies:
            _run([resolver_python, "-m", "pip", "install", *runtime_dependencies], cwd=ramulator)
        # --all captures pip/setuptools/wheel as well as runtime/dev dependencies,
        # so the later no-build-isolation editable install cannot silently resolve
        # a different Python build environment.
        freeze = _run([resolver_python, "-m", "pip", "freeze", "--all"], cwd=ramulator)
        env_lock.write_text(freeze + "\n")
        shutil.rmtree(resolver)
    ram_venv = ram_tools / "ramulator-venv"
    if ram_venv.exists():
        shutil.rmtree(ram_venv)
    venv.EnvBuilder(with_pip=True).create(ram_venv)
    ram_python = ram_venv / "bin/python"
    _run([ram_python, "-m", "pip", "install", "-r", env_lock], cwd=ramulator)
    _run(["cmake", "-S", ".", "-B", "build"], cwd=ramulator)
    _run(["cmake", "--build", "build", "--parallel", str(args.jobs)], cwd=ramulator)
    _run(
        [ram_python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", "."],
        cwd=ramulator,
    )
    _run(
        [
            ram_python,
            "-m",
            "pytest",
            "-q",
            "tests/device_timings/test_lpddr5.py",
            "tests/latency_throughput/test_fast.py",
            "-k", "LPDDR5",
        ],
        cwd=ramulator,
    )

    _run(["./configure"], cwd=fio)
    _run(["make", f"-j{args.jobs}"], cwd=fio)
    version = _run(["./fio", "--version"], cwd=fio)
    enghelp = _run(["./fio", "--enghelp"], cwd=fio)
    if "libaio" not in {line.strip() for line in enghelp.splitlines()}:
        raise H3Error("pinned fio build lacks required libaio engine")

    built = {
        "ramulator2": str(ramulator),
        "fio": str(fio),
        "fio_version": version,
        "ramulator_python_lock": str(env_lock.resolve()),
        "ramulator_python_lock_sha256": sha256_file(env_lock),
    }
    if args.with_energy:
        drampower = _checkout(root, tools["drampower"])
        _run(
            [
                "cmake", "-S", ".", "-B", "build",
                "-DDRAMPOWER_BUILD_CLI=Y", "-DDRAMPOWER_BUILD_TESTS=Y",
            ],
            cwd=drampower,
        )
        _run(["cmake", "--build", "build", "--parallel", str(args.jobs)], cwd=drampower)
        _run(["ctest", "--test-dir", "build", "--output-on-failure"], cwd=drampower)
        built["drampower"] = str(drampower)
    return {"complete": True, "built": built}


def qualify_tools(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd()
    lock = load_json(args.lock)
    contract = load_json(args.contract)
    failures: list[str] = []
    observed: dict[str, Any] = {}

    for name in ("ramulator2", "fio"):
        spec = lock["tools"][name]
        path = root / spec["path"]
        if not path.is_dir():
            failures.append(f"{name}:missing")
            continue
        try:
            observed[name] = git_checkout_identity(path, spec)
        except H3Error as exc:
            failures.append(f"{name}:provenance:{exc}")

    ram_python = root / ".runtime/h3-tools/ramulator-venv/bin/python"
    lpddr = contract["lpddr"]
    if not ram_python.exists():
        failures.append("ramulator_python:missing")
    else:
        for ncs in [int(value) for value in lpddr["ncs_values"]]:
            code = (
                'import json,ramulator; '
                f'd=ramulator.dram.LPDDR5(org_preset={lpddr["org_preset"]!r},'
                f'timing_preset={lpddr["timing_preset"]!r},rank={int(lpddr["ranks_per_channel"])},nCS={ncs}); '
                'o,t=d.resolve(); print(json.dumps({"org":o,"timing":t},sort_keys=True))'
            )
            proc = subprocess.run(
                [str(ram_python), "-c", code],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if proc.returncode:
                failures.append(f"ramulator_lpddr5_profile_ncs{ncs}:failed")
                observed[f"ramulator_profile_ncs{ncs}_output"] = proc.stdout[-4000:]
                continue
            try:
                resolved = json.loads(proc.stdout.strip().splitlines()[-1])
                validate_resolved_profile(resolved["org"], resolved["timing"], lpddr, ncs)
                observed[f"ramulator_profile_ncs{ncs}"] = resolved
            except (json.JSONDecodeError, H3Error) as exc:
                failures.append(f"ramulator_lpddr5_profile_ncs{ncs}:{exc}")

    fio_path = root / lock["tools"]["fio"]["path"] / "fio"
    try:
        observed["fio_binary"] = fio_identity(fio_path, args.lock)
        enghelp = _run([fio_path, "--enghelp"], cwd=root)
        fio_engines = sorted({
            line.strip() for line in enghelp.splitlines() if line.strip()
        })
        observed["fio_engines"] = fio_engines
        if "libaio" not in fio_engines:
            failures.append("fio_engine:libaio_missing")
    except H3Error as exc:
        failures.append(f"fio_binary:{exc}")

    # Capture the exact machine build environment used for Ramulator. The
    # upstream development requirements are not hash-locked, so publication
    # evidence must preserve the resolved environment and built extension bytes.
    if ram_python.exists():
        try:
            freeze = _run([ram_python, "-m", "pip", "freeze"], cwd=root)
            freeze_path = args.out.with_name("ramulator-python-freeze.txt")
            freeze_path.parent.mkdir(parents=True, exist_ok=True)
            freeze_path.write_text(freeze + "\n")
            resolved_lock = root / ".runtime/h3-tools/ramulator-python-resolved.lock.txt"
            archived_lock = args.out.with_name("ramulator-python-resolved.lock.txt")
            if not resolved_lock.is_file():
                failures.append("ramulator_python_resolved_lock:missing")
            else:
                archived_lock.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(resolved_lock, archived_lock)
            observed["ramulator_python_environment"] = {
                "python_version": _run([ram_python, "--version"], cwd=root),
                "pip_version": _run([ram_python, "-m", "pip", "--version"], cwd=root),
                "freeze_path": str(freeze_path.resolve()),
                "freeze_sha256": sha256_file(freeze_path),
                "resolved_lock_path": str(archived_lock.resolve()) if archived_lock.is_file() else None,
                "resolved_lock_relative_path": archived_lock.name if archived_lock.is_file() else None,
                "resolved_lock_sha256": sha256_file(archived_lock) if archived_lock.is_file() else None,
                "lock_replay_semantics": (
                    "final Ramulator venv is recreated from this resolved pip-freeze --all lock; "
                    "the pinned pyproject runtime dependencies are included and editable install uses --no-build-isolation --no-deps"
                ),
            }
            observed["build_toolchain"] = {
                "cmake": _run(["cmake", "--version"], cwd=root).splitlines()[0],
                "cxx": _run([os.environ.get("CXX", "c++"), "--version"], cwd=root).splitlines()[0],
            }
            ramulator_path = root / lock["tools"]["ramulator2"]["path"]
            extensions = sorted(
                path for path in ramulator_path.rglob("*.so")
                if "build" in path.parts or "python" in path.parts
            )
            if not extensions:
                failures.append("ramulator_extension:missing")
            else:
                observed["ramulator_built_extensions"] = [
                    {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for path in extensions
                ]
        except H3Error as exc:
            failures.append(f"build_environment:{exc}")

    if getattr(args, "with_energy", False):
        spec = lock["tools"]["drampower"]
        path = root / spec["path"]
        if not path.is_dir():
            failures.append("drampower:missing")
        else:
            try:
                observed["drampower"] = git_checkout_identity(path, spec)
                candidates = list((path / "build").rglob("drampower_cli"))
                if len(candidates) != 1:
                    failures.append(f"drampower_cli:expected_one_observed_{len(candidates)}")
                else:
                    observed["drampower_cli"] = str(candidates[0].resolve())
            except H3Error as exc:
                failures.append(f"drampower:{exc}")

    offered = offered_frontend_gb_s(lpddr)
    channel_peak = theoretical_channel_gb_s(lpddr)
    observed["lpddr_frontend_offered_gb_s"] = offered
    observed["lpddr_single_channel_theoretical_gb_s"] = channel_peak
    if offered <= channel_peak:
        failures.append(f"ramulator_frontend_underdriven:{offered}<={channel_peak}")

    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_tool_qualification",
        "complete": not failures,
        "failures": failures,
        "observed": observed,
        "tool_lock_sha256": sha256_file(args.lock),
        "contract_sha256": sha256_file(args.contract),
    }
    write_json(args.out, result)
    if failures:
        raise H3Error("tool qualification failed: " + "; ".join(failures))
    return result

def ramulator_calibration(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(args.contract)
    lpddr = contract["lpddr"]
    lock = load_json(args.lock)
    ramulator_root = Path(args.ramulator_root).resolve()
    identity = git_checkout_identity(ramulator_root, lock["tools"]["ramulator2"])

    peak = theoretical_channel_gb_s(lpddr)
    expected = float(lpddr["single_channel_theoretical_bandwidth_gb_s"])
    if abs(peak - expected) / expected > 1e-12:
        raise H3Error(f"LPDDR theoretical channel peak drift: {peak} != {expected}")
    offered = offered_frontend_gb_s(lpddr)
    if offered <= peak:
        raise H3Error(
            f"Ramulator frontend does not saturate x16 channel: offered={offered}, peak={peak}"
        )

    channels = int(lpddr["logical_channels"])
    regimes = [str(value) for value in lpddr["calibration_regimes"]]
    points: list[dict[str, Any]] = []
    for ncs in [int(value) for value in lpddr["ncs_values"]]:
        for regime in regimes:
            if regime == "streaming":
                read = run_stream_point(lpddr, ncs=ncs, read_ratio=100)
                write = run_stream_point(lpddr, ncs=ncs, read_ratio=0)
                mixed = run_address_trace_point(lpddr, ncs=ncs, regime=regime, operation_mode="mixed")
                source = "pinned_ramulator2_streaming_plus_LoadStoreTrace_mixed"
            else:
                read = run_address_trace_point(lpddr, ncs=ncs, regime=regime, operation_mode="read")
                write = run_address_trace_point(lpddr, ncs=ncs, regime=regime, operation_mode="write")
                mixed = run_address_trace_point(lpddr, ncs=ncs, regime=regime, operation_mode="mixed")
                source = "pinned_ramulator2_LoadStoreTrace_flat_address_replay"
            mixed_total = float(mixed["bandwidth_gb_s"])
            if mixed_total <= 0:
                raise H3Error(f"Ramulator mixed bandwidth is non-positive for {regime} nCS={ncs}")
            points.append(
                {
                    "sensitivity_id": f"ramulator_{regime}_nCS{ncs}",
                    "source": source,
                    "regime": regime,
                    "nCS": ncs,
                    "refresh_enabled": bool(lpddr["refresh_enabled"]),
                    "single_channel_read_bandwidth_gb_s": float(read["bandwidth_gb_s"]),
                    "single_channel_write_bandwidth_gb_s": float(write["bandwidth_gb_s"]),
                    "single_channel_mixed_total_bandwidth_gb_s": mixed_total,
                    "aggregate_read_bandwidth_gb_s": float(read["bandwidth_gb_s"]) * channels,
                    "aggregate_write_bandwidth_gb_s": float(write["bandwidth_gb_s"]) * channels,
                    "aggregate_mixed_total_bandwidth_gb_s": mixed_total * channels,
                    "read_calibration": read,
                    "write_calibration": write,
                    "mixed_calibration": mixed,
                }
            )

    points.append(
        {
            "sensitivity_id": "platform_theoretical_peak",
            "source": "LPDDR5_6400_x16_peak_times_16_independent_logical_channels",
            "regime": "theoretical_peak",
            "nCS": None,
            "refresh_enabled": None,
            "single_channel_read_bandwidth_gb_s": peak,
            "single_channel_write_bandwidth_gb_s": peak,
            "single_channel_mixed_total_bandwidth_gb_s": peak,
            "aggregate_read_bandwidth_gb_s": peak * channels,
            "aggregate_write_bandwidth_gb_s": peak * channels,
            "aggregate_mixed_total_bandwidth_gb_s": peak * channels,
        }
    )
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_lpddr_calibration",
        "optimistic_baseline": True,
        "ramulator_tool": identity,
        "ramulator_standard": lpddr["ramulator_standard"],
        "org_preset": lpddr["org_preset"],
        "timing_preset": lpddr["timing_preset"],
        "ranks_per_channel": int(lpddr["ranks_per_channel"]),
        "frontend_clock_ratio": int(lpddr["frontend_clock_ratio"]),
        "stream_cls": int(lpddr["stream_cls"]),
        "stream_requests": int(lpddr["ramulator_stream_requests"]),
        "frontend_offered_bandwidth_gb_s": offered,
        "single_channel_theoretical_bandwidth_gb_s": peak,
        "aggregate_theoretical_bandwidth_gb_s": peak * channels,
        "calibration_regimes": regimes,
        "sensitivity_points": points,
        "claim_boundary": (
            "one measured x16 channel is scaled over 16 explicitly independent logical channels as an "
            "intentionally optimistic conventional-memory envelope; access-pattern regimes are preserved "
            "as sensitivities and are not collapsed to streaming bandwidth"
        ),
        "contract_sha256": sha256_file(args.contract),
        "tool_lock_sha256": sha256_file(args.lock),
    }
    write_json(args.out, result)
    return result


def _fio_binary(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    path = Path(args.fio).resolve()
    lock_path = Path(getattr(args, "lock", DEFAULT_LOCK))
    identity = fio_identity(path, lock_path)
    return path, identity



def ramulator_actual_calibration(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(args.contract)
    residency = load_json(args.residency)
    if residency.get("artifact_kind") != "fenix_h3_residency_replay":
        raise H3Error("actual-H3 Ramulator calibration requires a residency replay")
    contract_sha = sha256_file(args.contract)
    observed_contract = residency.get("contract_sha256") or (residency.get("provenance") or {}).get("contract_sha256")
    if observed_contract != contract_sha:
        raise H3Error("residency/contract mismatch for actual-H3 Ramulator calibration")
    logical = residency.get("logical_access_trace") or {}
    logical_path = Path(str(logical.get("path", "")))
    if not logical_path.is_file() or sha256_file(logical_path) != logical.get("sha256"):
        raise H3Error("residency logical-access trace is missing or changed")
    lock = load_json(args.lock)
    identity = git_checkout_identity(Path(args.ramulator_root).resolve(), lock["tools"]["ramulator2"])
    lpddr = contract["lpddr"]
    channels = int(lpddr["logical_channels"])
    points: list[dict[str, Any]] = []
    mappings = [
        str(value)
        for value in lpddr.get(
            "actual_trace_address_mappings", ["dense_first_touch", "hashed_bank_spread"]
        )
    ]
    if not mappings or len(mappings) != len(set(mappings)):
        raise H3Error("actual-H3 LPDDR mapping sensitivity set is empty or duplicated")
    for mapping in mappings:
        for ncs in [int(value) for value in lpddr["ncs_values"]]:
            read = run_actual_h3_trace_point(
                lpddr, contract, residency, ncs, "read", mapping
            )
            write = run_actual_h3_trace_point(
                lpddr, contract, residency, ncs, "write", mapping
            )
            mixed = run_actual_h3_trace_point(
                lpddr, contract, residency, ncs, "mixed", mapping
            )
            points.append({
                "sensitivity_id": f"actual_h3_policy_trace_{mapping}_nCS{ncs}",
                "source": "pinned_ramulator2_LoadStoreTrace_from_actual_H3_object_sequence",
                "regime": "actual_h3_policy_trace",
                "address_mapping_strategy": mapping,
                "nCS": ncs,
                "aggregate_read_bandwidth_gb_s": float(read["bandwidth_gb_s"]) * channels,
                "aggregate_write_bandwidth_gb_s": float(write["bandwidth_gb_s"]) * channels,
                "aggregate_mixed_total_bandwidth_gb_s": float(mixed["bandwidth_gb_s"]) * channels,
                "read_calibration": read,
                "write_calibration": write,
                "mixed_calibration": mixed,
            })
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_lpddr_actual_trace_calibration",
        "scope": "policy_stratum_capacity_specific_sampled_actual_H3_logical_sequence",
        "stratum": residency.get("stratum"),
        "policy": residency.get("policy"),
        "capacity_gib": residency.get("capacity_gib"),
        "phase_filter": residency.get("phase_filter"),
        "ramulator_tool": identity,
        "address_mapping_strategies": mappings,
        "sensitivity_points": points,
        "contract_sha256": contract_sha,
        "provenance": {
            "residency_sha256": sha256_file(args.residency),
            "logical_access_trace_sha256": logical.get("sha256"),
            "contract_sha256": contract_sha,
        },
        "claim_boundary": "bounded actual-object/order high-MLP throughput calibration under multiple collision-free sample-local address layouts; not dependency-limited latency and not a claim of native physical-address preservation",
    }
    write_json(args.out, result)
    return result


def run_fio_one(
    samples_path: Path,
    sample_id: str,
    qd: int,
    out: Path,
    fio: Path,
    fio_tool: dict[str, Any],
    repeat_index: int = 0,
) -> int:
    payload = load_json(samples_path)
    matches = [row for row in payload["samples"] if row["sample_id"] == sample_id]
    if len(matches) != 1:
        raise H3Error(f"sample ID {sample_id!r} occurs {len(matches)} times")
    sample = matches[0]
    iolog = Path(sample["iolog"])
    if not iolog.is_file() or sha256_file(iolog) != sample["iolog_sha256"]:
        raise H3Error(f"iolog drift for {sample_id}")
    text = iolog.read_text()
    if " write " in text or " trim " in text:
        raise H3Error("H3 storage calibration must be read-only")
    command = [
        str(fio),
        "--name=fenix-h3",
        f"--read_iolog={iolog.resolve()}",
        "--replay_no_stall=1",
        "--ioengine=libaio",
        "--direct=1",
        f"--iodepth={int(qd)}",
        "--readonly",
        "--output-format=json",
    ]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(proc.stdout)
    actual_read_bytes: int | None = None
    achieved_qd_fraction: float | None = None
    parse_error: str | None = None
    if proc.stdout.strip():
        try:
            obj = json.loads(proc.stdout)
            from analysis.h3.storage import fio_achieved_qd_fraction, fio_read_bytes
            actual_read_bytes = fio_read_bytes(obj)
            achieved_qd_fraction = fio_achieved_qd_fraction(obj, int(qd), 0.5)
        except Exception as exc:
            parse_error = str(exc)
    write_json(
        out.with_suffix(".meta.json"),
        {
            "schema_version": 6,
            "artifact_kind": "fenix_h3_fio_run_provenance",
            "returncode": proc.returncode,
            "stderr": proc.stderr,
            "stdout_parse_error": parse_error,
            "actual_read_bytes": actual_read_bytes,
            "expected_read_bytes": int(sample["bytes"]),
            "queue_depth": int(qd),
            "achieved_qd_fraction_at_half_configured_depth": achieved_qd_fraction,
            "repeat_index": int(repeat_index),
            "sample_id": sample_id,
            "samples_sha256": sha256_file(samples_path),
            "iolog_sha256": sample["iolog_sha256"],
            "storage_binding_sha256": payload.get("storage_binding_sha256"),
            "device_major_minor": payload.get("device_major_minor"),
            "contract_sha256": payload.get("contract_sha256"),
            "fio_tool": fio_tool,
            "command": command,
        },
    )
    if proc.returncode == 0 and actual_read_bytes != int(sample["bytes"]):
        return 4
    return proc.returncode

def run_fio(args: argparse.Namespace) -> dict[str, Any]:
    fio, identity = _fio_binary(args)
    rc = run_fio_one(
        args.samples, args.sample_id, args.qd, args.out, fio, identity,
        int(getattr(args, "repeat_index", 0)),
    )
    if rc:
        raise H3Error(f"fio failed for {args.sample_id} qd={args.qd}")
    return {
        "complete": True, "sample_id": args.sample_id, "qd": args.qd,
        "repeat_index": int(getattr(args, "repeat_index", 0)),
    }


def run_fio_matrix(args: argparse.Namespace) -> dict[str, Any]:
    fio, identity = _fio_binary(args)
    samples = load_json(args.samples)["samples"]
    storage_cfg = load_json(args.contract)["storage"]
    qds = [int(value) for value in storage_cfg["queue_depths"]]
    repeats = int(storage_cfg.get("runs_per_sample", 1))
    jobs = [
        (sample, qd, rep)
        for sample in samples for qd in qds for rep in range(repeats)
    ]
    if bool(storage_cfg.get("randomize_run_order", False)):
        rng = random.Random(int(storage_cfg.get("run_order_seed", 20260906)))
        rng.shuffle(jobs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    order: list[str] = []
    for sample, qd, rep in jobs:
        out = args.out_dir / f"{sample['sample_id']}-qd{qd}-rep{rep:02d}.json"
        if run_fio_one(args.samples, sample["sample_id"], qd, out, fio, identity, rep):
            raise H3Error(
                f"fio matrix failed at {sample['sample_id']} qd={qd} rep={rep}"
            )
        order.append(f"{sample['sample_id']}:qd{qd}:rep{rep}")
        completed += 1
    manifest = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_fio_matrix_execution",
        "complete": True,
        "runs": completed,
        "runs_per_sample": repeats,
        "randomized": bool(storage_cfg.get("randomize_run_order", False)),
        "run_order_seed": storage_cfg.get("run_order_seed"),
        "execution_order": order,
        "samples_sha256": sha256_file(args.samples),
        "contract_sha256": sha256_file(args.contract),
    }
    write_json(args.out_dir / "matrix-execution.json", manifest)
    return manifest


def prepare_backing(args: argparse.Namespace) -> dict[str, Any]:
    geometry = load_json(args.contract)["geometry"]
    expected_ple = int(geometry["ple_addressable_rows"]) * int(geometry["ple_row_bytes"])
    if expected_ple != int(geometry["ple_data_bytes"]):
        raise H3Error("H3 PLE byte geometry drift")
    expected_ple_sha = str(geometry["ple_data_sha256"])
    expected_expert = (
        int(geometry["layers"])
        * int(geometry["experts_per_layer"])
        * int(geometry["expert_storage_stride_bytes"])
    )
    failures: list[str] = []
    ple_manifest_path = args.ple_manifest or (args.ple.parent / "ple.manifest.json")
    ple_manifest: dict[str, Any] | None = None
    ple_sha256: str | None = None
    if not args.ple.is_file() or args.ple.stat().st_size != expected_ple:
        failures.append(f"PLE backing must be exactly {expected_ple} bytes")
    if not ple_manifest_path.is_file():
        failures.append(f"PLE manifest is missing: {ple_manifest_path}")
    else:
        ple_manifest = load_json(ple_manifest_path)
        kind = ple_manifest.get("artifact_kind")
        if kind is not None and kind != "fenix_ple_storage_bank":
            failures.append("PLE manifest artifact kind is invalid")
        layout = ple_manifest.get("layout") or {}
        if not isinstance(layout, dict):
            failures.append("PLE manifest layout must be an object when present")
            layout = {}

        def manifest_value(name: str) -> Any:
            return ple_manifest.get(name, layout.get(name))

        if int(manifest_value("data_bytes") or -1) != expected_ple:
            failures.append("PLE manifest data_bytes differs from H3 geometry")
        if int(manifest_value("total_rows") or -1) != int(geometry["ple_addressable_rows"]):
            failures.append("PLE manifest row count differs from H3 geometry")
        if int(manifest_value("row_bytes") or -1) != int(geometry["ple_row_bytes"]):
            failures.append("PLE manifest row width differs from H3 geometry")
        dtype = manifest_value("dtype")
        if dtype is not None and str(dtype) != str(geometry.get("ple_dtype")):
            failures.append("PLE manifest dtype differs from checkpoint-exact H3 contract")
        shard_count = manifest_value("shard_count")
        if shard_count is not None and int(shard_count) != int(geometry.get("ple_shard_count", shard_count)):
            failures.append("PLE manifest shard count differs from checkpoint-exact H3 contract")
        manifest_sha = (
            ple_manifest.get("data_sha256")
            or ple_manifest.get("sha256")
            or ple_manifest.get("SHA256")
            or layout.get("data_sha256")
            or layout.get("sha256")
        )
        if not isinstance(manifest_sha, str):
            failures.append("PLE manifest data SHA256 is missing")
        elif manifest_sha != expected_ple_sha:
            failures.append("PLE manifest SHA256 differs from checkpoint-exact H3 contract")

    # The local bank itself is always hashed against the checkpoint-exact H3
    # contract; manifest agreement alone is insufficient evidence.
    if args.ple.is_file() and args.ple.stat().st_size == expected_ple:
        ple_sha256 = sha256_file(args.ple)
        if ple_sha256 != expected_ple_sha:
            failures.append("PLE backing SHA256 differs from checkpoint-exact H3 contract")
    if args.create_expert and not args.expert.exists():
        fio, _ = _fio_binary(args)
        args.expert.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(fio),
            "--name=fenix-h3-expert-backing",
            f"--filename={args.expert.resolve()}",
            "--rw=write",
            "--bs=1M",
            f"--size={expected_expert}",
            "--ioengine=sync",
            "--direct=0",
            "--fallocate=native",
            "--refill_buffers=1",
            "--scramble_buffers=1",
            "--buffer_compress_percentage=0",
            "--end_fsync=1",
        ]
        if subprocess.run(command).returncode:
            failures.append("expert-backing fio creation failed")
    if not args.expert.is_file():
        failures.append("expert backing missing")
    else:
        if args.expert.stat().st_size != expected_expert:
            failures.append(f"expert backing must be exactly {expected_expert} bytes")
        if args.expert.stat().st_blocks * 512 < int(expected_expert * 0.95):
            failures.append("expert backing appears sparse or heavily compressed")
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_backing_validation",
        "expected_ple_bytes": expected_ple,
        "expected_ple_sha256": expected_ple_sha,
        "expected_expert_bytes": expected_expert,
        "expert_expected_gib": expected_expert / GIB,
        "ple_manifest": str(ple_manifest_path.resolve()),
        "ple_manifest_sha256": (sha256_file(ple_manifest_path) if ple_manifest_path.is_file() else None),
        "ple_data_sha256": ple_sha256,
        "expert_content_semantics": "geometry-equivalent incompressible storage payload; not model weights",
        "complete": not failures,
        "failures": failures,
    }
    write_json(args.out, result)
    if failures:
        raise H3Error("; ".join(failures))
    return result


def capacity_budget(args: argparse.Namespace) -> dict[str, Any]:
    return build_capacity_budget(args.input, args.contract, args.out)



def bind_prerequisites(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(args.contract)
    contract_sha = sha256_file(args.contract)
    supplied = {
        "placement_invariance": args.placement_invariance,
        "long_context_beyond_8k": args.long_context,
        "deployment_capacity_budget": args.capacity_budget,
        "tool_qualification": args.tool_qualification,
    }
    if getattr(args, "storage_concurrency", None) is not None:
        supplied["storage_concurrency_feasibility"] = args.storage_concurrency
    evidence: dict[str, Any] = {}
    required_specs = contract["prerequisites"]["required_evidence"]
    conditional_specs = contract["prerequisites"].get("conditional_evidence", {})
    for key, path in supplied.items():
        if not path.is_file():
            raise H3Error(f"prerequisite evidence missing: {path}")
        payload = load_json(path)
        spec = required_specs.get(key) or conditional_specs.get(key)
        if not isinstance(spec, dict):
            raise H3Error(f"unknown prerequisite evidence key {key}")
        expected = spec["artifact_kind"]
        if payload.get("artifact_kind") != expected:
            raise H3Error(f"{key}: expected artifact kind {expected}")
        observed_contract = payload.get("contract_sha256") or (payload.get("provenance") or {}).get("contract_sha256")
        if observed_contract != contract_sha:
            raise H3Error(f"{key}: contract SHA mismatch")
        evidence[key] = {"artifact": str(path.resolve()), "sha256": sha256_file(path)}
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_prerequisite_manifest",
        "evidence": evidence,
        "contract_sha256": contract_sha,
    }
    write_json(args.out, result)
    return result


def _cgroup_dir() -> Path:
    if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
        raise H3Error("cgroup v2 required for page-cache control")
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0":
            return Path("/sys/fs/cgroup") / fields[2].lstrip("/")
    raise H3Error("cannot resolve cgroup v2 path")


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_io_stat(path: Path) -> dict[str, dict[str, int]]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise H3Error(f"cannot read cgroup io.stat: {exc}") from exc
    output: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        values: dict[str, int] = {}
        for field in fields[1:]:
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            try:
                values[key] = int(value)
            except ValueError:
                continue
        output[fields[0]] = values
    return output


def _io_delta(before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]) -> dict[str, int]:
    keys = {"rbytes", "wbytes", "rios", "wios", "dbytes", "dios"}
    totals = {key: 0 for key in keys}
    for device in set(before) | set(after):
        for key in keys:
            totals[key] += int(after.get(device, {}).get(key, 0)) - int(before.get(device, {}).get(key, 0))
    return totals




def _read_key_values(path: Path) -> dict[str, int]:
    try:
        text = path.read_text()
    except OSError:
        return {}
    out: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                out[fields[0]] = int(fields[1])
            except ValueError:
                continue
    return out


def _kv_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in sorted(set(before) | set(after))}


def _read_text_best_effort(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def _io_delta_by_device(before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    keys = {"rbytes", "wbytes", "rios", "wios", "dbytes", "dios"}
    return {
        device: {key: int(after.get(device, {}).get(key, 0)) - int(before.get(device, {}).get(key, 0)) for key in sorted(keys)}
        for device in sorted(set(before) | set(after))
    }


def _sample_iolog_page_offsets(iolog: Path, max_pages_per_file: int) -> dict[Path, list[int]]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    candidates: dict[Path, list[int]] = {}
    seen: dict[Path, set[int]] = {}
    for line in iolog.read_text().splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[1] != "read":
            continue
        path = Path(parts[0]).resolve()
        offset = int(parts[2])
        length = int(parts[3])
        if length <= 0:
            continue
        first = offset // page_size * page_size
        last = (offset + length - 1) // page_size * page_size
        middle = ((first + last) // (2 * page_size)) * page_size
        bucket = candidates.setdefault(path, [])
        known = seen.setdefault(path, set())
        for value in (first, middle, last):
            if value not in known:
                known.add(value)
                bucket.append(value)
    sampled: dict[Path, list[int]] = {}
    for path, values in candidates.items():
        if len(values) <= max_pages_per_file:
            sampled[path] = values
            continue
        # Deterministic even coverage over the logical trace-derived candidate set.
        step = (len(values) - 1) / float(max_pages_per_file - 1)
        sampled[path] = [values[round(index * step)] for index in range(max_pages_per_file)]
    return sampled


def _mincore_fraction(path: Path, page_offsets: list[int]) -> tuple[int, int, float]:
    if not page_offsets:
        return 0, 0, 0.0
    libc = ctypes.CDLL(None, use_errno=True)
    mincore = libc.mincore
    mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    mincore.restype = ctypes.c_int
    page_size = os.sysconf("SC_PAGE_SIZE")
    resident = 0
    checked = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        file_size = os.fstat(fd).st_size
        for offset in page_offsets:
            if offset < 0 or offset + page_size > file_size:
                continue
            mm = mmap.mmap(fd, page_size, access=mmap.ACCESS_COPY, offset=offset)
            try:
                view = (ctypes.c_char * page_size).from_buffer(mm)
                vec = ctypes.c_ubyte(0)
                rc = mincore(ctypes.addressof(view), page_size, ctypes.byref(vec))
                del view
                if rc != 0:
                    errno = ctypes.get_errno()
                    raise H3Error(f"mincore failed for {path} at {offset}: errno={errno}")
                checked += 1
                resident += 1 if (int(vec.value) & 1) else 0
            finally:
                mm.close()
    finally:
        os.close(fd)
    return checked, resident, (resident / checked if checked else 0.0)


def _sample_residency(iolog: Path, max_pages_per_file: int) -> dict[str, Any]:
    offsets = _sample_iolog_page_offsets(iolog, max_pages_per_file)
    files: dict[str, Any] = {}
    total_checked = 0
    total_resident = 0
    for path, values in offsets.items():
        checked, resident, fraction = _mincore_fraction(path, values)
        files[str(path)] = {
            "sampled_pages": checked,
            "resident_pages": resident,
            "resident_fraction": fraction,
        }
        total_checked += checked
        total_resident += resident
    if total_checked == 0:
        raise H3Error("mincore cold-cache qualification sampled zero pages")
    return {
        "files": files,
        "sampled_pages": total_checked,
        "resident_pages": total_resident,
        "resident_fraction": total_resident / total_checked,
    }


def _drop_file_cache(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise H3Error("POSIX_FADV_DONTNEED is unavailable; cannot qualify cold page-cache control")
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def pagecache_inner(args: argparse.Namespace) -> dict[str, Any]:
    window = load_json(args.window)
    if not window.get("pressure_sufficient"):
        raise H3Error("page-cache window does not satisfy unique-footprint pressure gate")
    iolog = Path(window["iolog"])
    if not iolog.is_file() or sha256_file(iolog) != str(window.get("iolog_sha256")):
        raise H3Error("page-cache logical iolog drift")
    binding = load_json(args.storage_binding)
    if binding.get("artifact_kind") != "fenix_h3_storage_binding":
        raise H3Error("unexpected storage-binding artifact for page-cache control")
    if not binding.get("same_device_major_minor"):
        raise H3Error("page-cache control requires PLE/expert backings on one resolved block device")
    cg = _cgroup_dir()
    observed_max = _read_int(cg / "memory.max")
    observed_swap_max = _read_int(cg / "memory.swap.max")
    if observed_max is None or observed_max > args.expected_memory_max_bytes:
        raise H3Error(f"MemoryMax not enforced: observed={observed_max}")
    if observed_swap_max != 0:
        raise H3Error(f"MemorySwapMax=0 not enforced: observed={observed_swap_max}")
    fio, fio_tool = _fio_binary(args)
    ple_backing = Path(binding["ple"]["path"])
    expert_backing = Path(binding["expert"]["path"])
    _drop_file_cache(ple_backing)
    _drop_file_cache(expert_backing)
    pagecache_cfg = load_json(args.contract)["pagecache"]
    mincore_before = _sample_residency(
        iolog, int(pagecache_cfg.get("mincore_sample_pages_per_file", 256))
    )
    max_initial_resident = float(
        pagecache_cfg.get("max_initial_resident_sample_fraction", 0.05)
    )
    if float(mincore_before["resident_fraction"]) > max_initial_resident:
        raise H3Error(
            "page-cache cold-control mincore gate failed: "
            f"resident_fraction={mincore_before['resident_fraction']:.4f} > {max_initial_resident:.4f}"
        )
    samples: list[int] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            samples.append(_read_int(cg / "memory.current") or 0)
            stop.wait(0.05)

    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    io_before = _read_io_stat(cg / "io.stat")
    memstat_before = _read_key_values(cg / "memory.stat")
    events_before = _read_key_values(cg / "memory.events")
    psi_before = _read_text_best_effort(cg / "memory.pressure")
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    engine = "sync" if args.mode == "buffered" else "mmap"
    command = [
        str(fio),
        "--name=fenix-h3-pagecache",
        f"--read_iolog={iolog.resolve()}",
        "--replay_no_stall=1",
        f"--ioengine={engine}",
        "--direct=0",
        "--iodepth=1",
        "--readonly",
        "--output-format=json",
    ]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stop.set()
    thread.join(timeout=2)
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    io_after = _read_io_stat(cg / "io.stat")
    memstat_after = _read_key_values(cg / "memory.stat")
    events_after = _read_key_values(cg / "memory.events")
    psi_after = _read_text_best_effort(cg / "memory.pressure")
    io_delta = _io_delta(io_before, io_after)
    io_devices = _io_delta_by_device(io_before, io_after)
    event_delta = _kv_delta(events_before, events_after)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fio_json_path = args.out.with_suffix(".fio.json")
    fio_json_path.write_text(proc.stdout)
    actual_read_bytes: int | None = None
    actual_runtime_ns: int | None = None
    if proc.returncode == 0:
        try:
            from analysis.h3.storage import fio_read_bytes, fio_runtime_ns
            fio_payload = json.loads(proc.stdout)
            actual_read_bytes = fio_read_bytes(fio_payload)
            actual_runtime_ns = fio_runtime_ns(fio_payload)
        except Exception as exc:
            raise H3Error(f"page-cache fio JSON invalid: {exc}") from exc
    max_current = max(samples, default=0)
    fraction = float(load_json(args.contract)["pagecache"]["minimum_observed_memcg_fraction"])
    pressure = max_current >= fraction * args.expected_memory_max_bytes
    target_device = str(binding["ple"]["mount"]["maj:min"])
    target_io = io_devices.get(target_device, {})
    target_rbytes = int(target_io.get("rbytes", 0))
    target_wbytes = int(target_io.get("wbytes", 0))
    target_rios = int(target_io.get("rios", 0))
    target_wios = int(target_io.get("wios", 0))
    # Qualification is tied to the resolved backing device, not aggregate cgroup
    # I/O, so unrelated child/process traffic cannot manufacture a cold-cache pass.
    cold_qualified = bool(proc.returncode == 0 and pressure and target_rbytes > 0)
    no_oom = int(event_delta.get("oom", 0)) == 0 and int(event_delta.get("oom_kill", 0)) == 0
    mincore_after = _sample_residency(
        iolog, int(pagecache_cfg.get("mincore_sample_pages_per_file", 256))
    )
    result = {
        "schema_version": 6,
        "artifact_kind": "fenix_h3_pagecache_measurement",
        "mode": args.mode,
        "stratum": window["stratum"],
        "capacity_gib": float(window["capacity_gib"]),
        "returncode": proc.returncode,
        "stderr": proc.stderr,
        "fio_tool": fio_tool,
        "fio_actual_read_bytes": actual_read_bytes,
        "fio_runtime_ns": actual_runtime_ns,
        "mincore_before": mincore_before,
        "mincore_after": mincore_after,
        "max_initial_resident_sample_fraction": max_initial_resident,
        "cgroup": str(cg),
        "observed_memory_max_bytes": observed_max,
        "observed_memory_swap_max_bytes": observed_swap_max,
        "expected_memory_max_bytes": args.expected_memory_max_bytes,
        "max_memory_current_bytes": max_current,
        "minimum_observed_fraction": fraction,
        "pressure_observed": pressure,
        "cold_cache_advisory_applied": True,
        "cold_cache_qualified": cold_qualified,
        "cold_cache_method": "POSIX_FADV_DONTNEED plus trace-derived mincore residency sampling before replay; qualification additionally requires lower-tier reads",
        "storage_read_bytes_cgroup": target_rbytes,
        "storage_write_bytes_cgroup": target_wbytes,
        "storage_read_ios_cgroup": target_rios,
        "storage_write_ios_cgroup": target_wios,
        "aggregate_cgroup_io_delta": io_delta,
        "io_delta_by_device": io_devices,
        "resolved_backing_device_major_minor": target_device,
        "resolved_backing_device_io_delta": target_io,
        "logical_read_bytes": int(window["logical_bytes"]),
        "storage_read_fraction_of_logical_bytes": (
            target_rbytes / int(window["logical_bytes"]) if int(window["logical_bytes"]) else None
        ),
        "child_minor_faults": usage_after.ru_minflt - usage_before.ru_minflt,
        "child_major_faults": usage_after.ru_majflt - usage_before.ru_majflt,
        "memory_stat_before": memstat_before,
        "memory_stat_after": memstat_after,
        "memory_events_before": events_before,
        "memory_events_after": events_after,
        "memory_events_delta": event_delta,
        "memory_psi_before": psi_before,
        "memory_psi_after": psi_after,
        "no_oom": no_oom,
        "window_sha256": sha256_file(args.window),
        "trace_manifest_sha256": window.get("manifest_sha256"),
        "full_trace_replayed": window.get("full_trace_replayed") is True,
        "iolog_sha256": sha256_file(iolog),
        "storage_binding_sha256": sha256_file(args.storage_binding),
        "contract_sha256": sha256_file(args.contract),
        "capacity_semantics": "process_plus_pagecache_memcg_envelope_not_exact_pagecache_capacity",
    }
    write_json(args.out, result)
    if proc.returncode:
        raise H3Error(f"page-cache fio failed rc={proc.returncode}")
    if actual_read_bytes != int(window["logical_bytes"]):
        raise H3Error("page-cache fio logical read-byte count mismatch")
    if not pressure:
        raise H3Error("memcg pressure gate was not reached")
    if not no_oom:
        raise H3Error("page-cache run encountered memcg OOM events")
    if not cold_qualified:
        raise H3Error("page-cache cold-control qualification requires observed cgroup lower-tier reads")
    return result

def pagecache_scope(args: argparse.Namespace) -> dict[str, Any]:
    if shutil.which("systemd-run") is None:
        raise H3Error("systemd-run is required for automatic page-cache MemoryMax scope")
    if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
        raise H3Error("cgroup v2 is unavailable")
    window = load_json(args.window)
    binding = load_json(args.storage_binding)
    if binding.get("artifact_kind") != "fenix_h3_storage_binding":
        raise H3Error("unexpected storage-binding artifact")
    if float(window["capacity_gib"]) != float(args.memory_gib):
        raise H3Error("page-cache window capacity differs from requested MemoryMax")
    memory_bytes = int(float(args.memory_gib) * GIB)
    python = Path.cwd() / ".venv/bin/python"
    command = [
        "systemd-run", "--user", "--scope", "--quiet", "--collect",
        f"--unit=fenix-h3-{uuid.uuid4().hex[:10]}.scope",
        "-p", f"MemoryMax={memory_bytes}",
        "-p", "MemorySwapMax=0",
        str(python), "-m", "scripts.h3_campaign", "pagecache-inner",
        "--window", str(args.window.resolve()),
        "--mode", args.mode,
        "--expected-memory-max-bytes", str(memory_bytes),
        "--out", str(args.out.resolve()),
        "--storage-binding", str(args.storage_binding.resolve()),
        "--fio", str(Path(args.fio).resolve()),
        "--lock", str(args.lock.resolve()),
        "--contract", str(args.contract.resolve()),
    ]
    proc = subprocess.run(command)
    if proc.returncode:
        raise H3Error(f"page-cache cgroup scope failed rc={proc.returncode}")
    return load_json(args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bootstrap-tools")
    p.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    p.add_argument("--jobs", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    p.add_argument("--with-energy", action="store_true")
    p.set_defaults(func=bootstrap_tools)

    p = sub.add_parser("qualify-tools")
    p.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--with-energy", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=qualify_tools)

    p = sub.add_parser("manifest")
    p.add_argument("--trace-root", type=Path, required=True)
    p.add_argument("--robustness-contract", type=Path, default=Path("configs/h1_h2_workload_robustness_v1.json"))
    p.add_argument("--campaign", type=Path, default=Path("configs/campaign.json"))
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--execution-verification", type=Path, default=Path("configs/h1_h2_trace_execution_v1.json"))
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: build_manifest(a.trace_root, a.robustness_contract, a.campaign, a.contract, a.out, a.execution_verification))

    p = sub.add_parser("replay")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--stratum", required=True)
    p.add_argument("--capacity-gib", type=float, required=True)
    p.add_argument(
        "--policy",
        choices=[
            "useful_object_lru", "vm_granularity_lru",
            "useful_object_lfu", "vm_granularity_lfu",
        ],
        required=True,
    )
    p.add_argument("--phase", choices=["prefill", "decode"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=lambda a: replay_case(a.manifest, a.stratum, a.capacity_gib, a.policy, a.out_dir, a.phase))

    p = sub.add_parser("prepare-backing")
    p.add_argument("--ple", type=Path, required=True)
    p.add_argument("--ple-manifest", type=Path)
    p.add_argument("--expert", type=Path, required=True)
    p.add_argument("--create-expert", action="store_true")
    p.add_argument("--fio", type=Path, default=Path("external/tools/fio/fio"))
    p.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=prepare_backing)

    p = sub.add_parser("probe-storage")
    p.add_argument("--ple", type=Path, required=True)
    p.add_argument("--expert", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: probe_storage(a.ple, a.expert, a.out, a.contract))

    p = sub.add_parser("samples")
    p.add_argument("--misses", type=Path, required=True)
    p.add_argument("--ple", type=Path, required=True)
    p.add_argument("--expert", type=Path, required=True)
    p.add_argument("--storage-binding", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    def _samples(a: argparse.Namespace) -> dict[str, Any]:
        s = load_json(a.contract)["storage"]
        return build_samples(
            a.misses, a.ple, a.expert, a.storage_binding, a.out_dir,
            s["seeds"], s["ple_sample_bytes"], s["expert_sample_bytes"], s["mixed_sample_bytes"],
            contract_sha256=sha256_file(a.contract),
        )
    p.set_defaults(func=_samples)

    p = sub.add_parser("fio-one")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--qd", type=int, required=True)
    p.add_argument("--repeat-index", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fio", type=Path, default=Path("external/tools/fio/fio"))
    p.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    p.set_defaults(func=run_fio)

    p = sub.add_parser("fio-matrix")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--fio", type=Path, default=Path("external/tools/fio/fio"))
    p.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    p.set_defaults(func=run_fio_matrix)

    p = sub.add_parser("fio-summary")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--out", type=Path, required=True)
    def _summary(a: argparse.Namespace) -> dict[str, Any]:
        storage_cfg = load_json(a.contract)["storage"]
        qds = [int(value) for value in storage_cfg["queue_depths"]]
        return summarize_fio(
            a.samples, a.runs_dir, qds, a.out,
            runs_per_sample=int(storage_cfg.get("runs_per_sample", 1)),
            minimum_achieved_qd_fraction=float(storage_cfg.get("minimum_achieved_qd_fraction", 0.0)),
            achieved_qd_threshold_fraction=float(storage_cfg.get("achieved_qd_threshold_fraction", 0.5)),
            bootstrap_draws=int(storage_cfg.get("bootstrap_draws", 10000)),
            preferred_common_windows=int(storage_cfg.get("preferred_common_windows_for_promotion", 5)),
            target_common_windows=int(storage_cfg.get("target_common_windows_for_promotion", 8)),
        )
    p.set_defaults(func=_summary)

    p = sub.add_parser("ramulator-actual")
    p.add_argument("--residency", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    p.add_argument("--ramulator-root", type=Path, default=Path("external/tools/ramulator2"))
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=ramulator_actual_calibration)

    p = sub.add_parser("ramulator")
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    p.add_argument("--ramulator-root", type=Path, default=Path("external/tools/ramulator2"))
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=ramulator_calibration)

    p = sub.add_parser("decide")
    p.add_argument("--residency", type=Path, required=True)
    p.add_argument("--lpddr", type=Path, required=True)
    p.add_argument("--fio", type=Path, required=True)
    p.add_argument("--pagecache", type=Path, action="append", default=[])
    p.add_argument("--actual-lpddr", type=Path)
    p.add_argument("--qd", type=int, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: decide_point(
        a.residency, a.lpddr, a.fio, a.contract, a.qd, a.out, a.pagecache, a.actual_lpddr
    ))

    p = sub.add_parser("campaign-gate")
    p.add_argument("--decision", type=Path, action="append", required=True)
    p.add_argument("--prerequisites", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: campaign_gate(a.decision, a.prerequisites, a.contract, a.out))

    p = sub.add_parser("capacity-budget")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=capacity_budget)

    p = sub.add_parser("placement-invariance-evidence")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: build_placement_invariance(a.input, a.contract, a.out))

    p = sub.add_parser("long-context-evidence")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: build_long_context_scaling(a.input, a.contract, a.out))

    p = sub.add_parser("storage-concurrency-evidence")
    p.add_argument("--measurement", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=lambda a: build_storage_concurrency_evidence(a.measurement, a.contract, a.out))

    p = sub.add_parser("bind-prerequisites")
    p.add_argument("--placement-invariance", type=Path, required=True)
    p.add_argument("--long-context", type=Path, required=True)
    p.add_argument("--capacity-budget", type=Path, required=True)
    p.add_argument("--tool-qualification", type=Path, required=True)
    p.add_argument("--storage-concurrency", type=Path)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=bind_prerequisites)

    p = sub.add_parser("pagecache-window")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--stratum", required=True)
    p.add_argument("--capacity-gib", type=float, required=True)
    p.add_argument("--ple", type=Path, required=True)
    p.add_argument("--expert", type=Path, required=True)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=lambda a: build_pagecache_window(a.manifest, a.stratum, a.capacity_gib, load_json(a.contract)["pagecache"]["pressure_factor"], a.ple, a.expert, a.out_dir))

    p = sub.add_parser("pagecache-scope")
    p.add_argument("--window", type=Path, required=True)
    p.add_argument("--mode", choices=["buffered", "mmap"], required=True)
    p.add_argument("--memory-gib", type=float, required=True)
    p.add_argument("--storage-binding", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fio", type=Path, default=Path("external/tools/fio/fio"))
    p.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.set_defaults(func=pagecache_scope)

    p = sub.add_parser("pagecache-inner")
    p.add_argument("--window", type=Path, required=True)
    p.add_argument("--mode", choices=["buffered", "mmap"], required=True)
    p.add_argument("--expected-memory-max-bytes", type=int, required=True)
    p.add_argument("--storage-binding", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fio", type=Path, default=Path("external/tools/fio/fio"))
    p.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.set_defaults(func=pagecache_inner)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        execution_identity = _execution_repository_identity()
        set_execution_repository_identity(execution_identity)
        result = args.func(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except H3Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
