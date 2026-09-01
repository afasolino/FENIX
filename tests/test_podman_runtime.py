from pathlib import Path

from qualification.podman_runtime import podman_command, podman_paths


def test_podman_store_is_project_local(tmp_path: Path):
    paths = podman_paths(tmp_path)
    assert paths.storage == (tmp_path / ".runtime/podman/storage").resolve()
    assert paths.run == (tmp_path / ".runtime/podman/run").resolve()
    assert paths.tmp == (tmp_path / ".runtime/podman/tmp").resolve()


def test_podman_command_uses_fuse_overlay_and_project_paths(tmp_path: Path):
    command = podman_command(tmp_path, "images")
    assert command[0] == "podman"
    assert str((tmp_path / ".runtime/podman/storage").resolve()) in command
    assert "overlay.mount_program=/usr/bin/fuse-overlayfs" in command
    assert command[-1] == "images"
