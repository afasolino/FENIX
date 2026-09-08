from pathlib import Path
import subprocess

import pytest

from qualification.podman_runtime import podman_command, podman_paths
from scripts.fenix_podman import repository_root_from_cwd


def test_podman_store_is_project_local_but_runtime_paths_are_short(tmp_path: Path):
    paths = podman_paths(tmp_path)
    assert paths.storage == (tmp_path / ".runtime/podman/storage").resolve()
    assert paths.run.name == "run"
    assert paths.tmp.name == "tmp"
    assert paths.run.parent == paths.tmp.parent
    assert paths.run.parent.name.startswith("fenix-pd-")
    assert str(paths.run).startswith("/tmp/")
    assert len(str(paths.run)) < 50


def test_podman_runtime_paths_are_deterministic_and_worktree_specific(tmp_path: Path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()

    a = podman_paths(first)
    b = podman_paths(first)
    c = podman_paths(second)

    assert a.run == b.run
    assert a.tmp == b.tmp
    assert a.run != c.run
    assert a.tmp != c.tmp


def test_podman_command_uses_fuse_overlay_and_short_runtime_paths(tmp_path: Path):
    command = podman_command(tmp_path, "images")
    paths = podman_paths(tmp_path)
    assert command[0] == "podman"
    assert str((tmp_path / ".runtime/podman/storage").resolve()) in command
    assert str(paths.run) in command
    assert str(paths.tmp) in command
    assert "overlay.mount_program=/usr/bin/fuse-overlayfs" in command
    assert command[-1] == "images"


def test_repository_root_accepts_git_worktree(tmp_path: Path):
    repo = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "fenix@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "FENIX Test"], check=True)
    (repo / "tracked.txt").write_text("x\n")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(worktree), "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert (worktree / ".git").is_file()
    assert repository_root_from_cwd(worktree) == worktree.resolve()


def test_repository_root_rejects_subdirectory(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    child = repo / "child"
    child.mkdir()
    with pytest.raises(RuntimeError, match="repository root"):
        repository_root_from_cwd(child)
