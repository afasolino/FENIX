from pathlib import Path
import subprocess

import pytest

from qualification.runtime_lane import inspect_runtime_source


def _config() -> dict:
    return {
        "runtime": {"revision": "abc123"},
        "source_checks": [
            {
                "id": "worker",
                "path": "worker.py",
                "markers": ["spawn_ple_offload", "tensor_parallel_size"],
            }
        ],
    }


def test_source_qualification_rejects_missing_marker(tmp_path: Path, monkeypatch):
    (tmp_path / "worker.py").write_text("spawn_ple_offload")

    class Result:
        ok = True
        stdout = "abc123"

    monkeypatch.setattr(
        "qualification.runtime_lane.run_command",
        lambda *args, **kwargs: Result(),
    )

    result = inspect_runtime_source(tmp_path, _config())

    assert result["passed"] is False
    assert result["failures"] == ["worker"]


def test_source_qualification_accepts_required_structure(tmp_path: Path, monkeypatch):
    (tmp_path / "worker.py").write_text(
        "spawn_ple_offload tensor_parallel_size"
    )

    class Result:
        ok = True
        stdout = "abc123"

    monkeypatch.setattr(
        "qualification.runtime_lane.run_command",
        lambda *args, **kwargs: Result(),
    )

    result = inspect_runtime_source(tmp_path, _config())

    assert result["passed"] is True
    assert result["failures"] == []



def _init_source_repository(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=path,
        check=True,
    )
    (path / "worker.py").write_text("worker")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()


def test_new_no_checkout_clone_is_initialized_before_cleanliness_check(
    tmp_path: Path,
):
    from qualification.runtime_lane import ensure_runtime_checkout

    source = tmp_path / "source"
    revision = _init_source_repository(source)

    root = tmp_path / "fenix"
    root.mkdir()
    config = {
        "runtime": {
            "repository": str(source),
            "revision": revision,
            "checkout": "external/runtime/qwen38",
        }
    }

    checkout = ensure_runtime_checkout(root, config)

    assert (checkout / "worker.py").read_text() == "worker"
    assert subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=checkout,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout == ""


def test_known_incomplete_clone_requires_explicit_repair(tmp_path: Path):
    from qualification.runtime_lane import ensure_runtime_checkout

    source = tmp_path / "source"
    revision = _init_source_repository(source)

    root = tmp_path / "fenix"
    checkout = root / "external/runtime/qwen38"
    checkout.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "--no-checkout", str(source), str(checkout)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    config = {
        "runtime": {
            "repository": str(source),
            "revision": revision,
            "checkout": "external/runtime/qwen38",
        }
    }

    with pytest.raises(RuntimeError, match="local modifications"):
        ensure_runtime_checkout(root, config)

    repaired = ensure_runtime_checkout(
        root,
        config,
        repair_incomplete_clone=True,
    )

    assert (repaired / "worker.py").exists()
    assert subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repaired,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout == ""


def test_repair_refuses_real_worktree_content(tmp_path: Path):
    from qualification.runtime_lane import repair_incomplete_no_checkout_clone

    source = tmp_path / "source"
    _init_source_repository(source)

    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "--no-checkout", str(source), str(checkout)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (checkout / "user-note.txt").write_text("preserve me")

    with pytest.raises(RuntimeError, match="not the recognized"):
        repair_incomplete_no_checkout_clone(checkout)

    assert (checkout / "user-note.txt").read_text() == "preserve me"
