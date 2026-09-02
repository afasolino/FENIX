from pathlib import Path

import pytest

from scripts import build_runtime


def test_default_build_plan_preserves_qualified_image(tmp_path: Path):
    root = tmp_path.resolve()
    python = root / ".venv/bin/python"

    plan = build_runtime.build_plan(
        root=root,
        python=python,
        image=build_runtime.DEFAULT_IMAGE,
        output=root / build_runtime.DEFAULT_OUTPUT,
    )

    assert plan.image == "fenix-qwen38:candidate"
    assert "fenix-qwen38:locked" not in plan.build_command
    assert plan.output == root / ".runtime/instrumented/qwen38-candidate"
    assert plan.prepare_command[-2:] == (
        "--output",
        ".runtime/instrumented/qwen38-candidate",
    )
    assert plan.build_command[-1] == ".runtime/instrumented/qwen38-candidate"


def test_qualified_image_is_protected_by_default(tmp_path: Path):
    with pytest.raises(
        build_runtime.RuntimeBuildError,
        match="refusing to overwrite qualified image",
    ):
        build_runtime.build_plan(
            root=tmp_path,
            python=tmp_path / ".venv/bin/python",
            image="fenix-qwen38:locked",
            output=tmp_path / ".runtime/instrumented/qwen38-candidate",
        )


def test_qualified_image_requires_explicit_escape_hatch(tmp_path: Path):
    plan = build_runtime.build_plan(
        root=tmp_path,
        python=tmp_path / ".venv/bin/python",
        image="fenix-qwen38:locked",
        output=tmp_path / ".runtime/instrumented/qwen38-candidate",
        allow_qualified_image_overwrite=True,
    )

    assert plan.image == "fenix-qwen38:locked"
    assert "fenix-qwen38:locked" in plan.build_command


def test_output_must_remain_project_local(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"

    with pytest.raises(
        build_runtime.RuntimeBuildError,
        match="must remain inside",
    ):
        build_runtime.build_plan(
            root=root,
            python=root / ".venv/bin/python",
            image=build_runtime.DEFAULT_IMAGE,
            output=outside,
        )


@pytest.mark.parametrize("image", ("", "  ", "bad tag"))
def test_invalid_image_tags_are_rejected(tmp_path: Path, image: str):
    with pytest.raises(build_runtime.RuntimeBuildError):
        build_runtime.build_plan(
            root=tmp_path,
            python=tmp_path / ".venv/bin/python",
            image=image,
            output=tmp_path / ".runtime/instrumented/qwen38-candidate",
        )


def test_shell_wrapper_delegates_policy_to_python():
    text = Path("scripts/build_runtime.sh").read_text()

    assert "scripts.build_runtime" in text
    assert "fenix-qwen38:locked" not in text
    assert "instrumentation/prepare_runtime.py" not in text
    assert "scripts.fenix_podman build" not in text
