from pathlib import Path

import pytest

from instrumentation.prepare_runtime import ensure_docker_context_include


def test_generated_hardener_is_allow_listed_in_docker_context(
    tmp_path: Path,
):
    (tmp_path / ".dockerignore").write_text(
        "**\n"
        "!runtime/install_overlay.py\n"
        "!runtime/vllm-overlay/**\n"
    )
    artifact = tmp_path / "runtime/fenix_harden_runtime_image.py"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("print('hardener')\n")

    first = ensure_docker_context_include(
        tmp_path,
        Path("runtime/fenix_harden_runtime_image.py"),
    )
    second = ensure_docker_context_include(
        tmp_path,
        Path("runtime/fenix_harden_runtime_image.py"),
    )

    lines = (tmp_path / ".dockerignore").read_text().splitlines()
    entry = "!runtime/fenix_harden_runtime_image.py"
    assert lines.count(entry) == 1
    assert first["allow_entry"] == entry
    assert second == first


def test_missing_generated_artifact_fails_closed(tmp_path: Path):
    (tmp_path / ".dockerignore").write_text("**\n")

    with pytest.raises(RuntimeError, match="artifact is missing"):
        ensure_docker_context_include(
            tmp_path,
            Path("runtime/fenix_harden_runtime_image.py"),
        )
