from pathlib import Path

import pytest

from instrumentation.harden_runtime_image import (
    ROPE_VALIDATION_IGNORE_KEYS,
    patch_env_registry,
    patch_qwen3vl_video_docstring,
    patch_rope_validation_config,
)


def test_env_registration_is_idempotent(tmp_path: Path):
    path = tmp_path / "envs.py"
    path.write_text("environment_variables = {}\n")

    assert patch_env_registry(path) is True
    assert patch_env_registry(path) is False

    text = path.read_text()
    assert text.count(
        "# FENIX custom WNA16 environment registration"
    ) == 1


@pytest.mark.parametrize(
    "class_name",
    (
        "Qwen3_8FlashNextTextConfig",
        "Qwen4ExpTextConfig",
    ),
)
def test_rope_validation_patch_is_class_scoped_and_idempotent(
    tmp_path: Path,
    class_name: str,
):
    path = tmp_path / "config.py"
    path.write_text(
        "class UnrelatedConfig(object):\n"
        "    pass\n\n"
        f"class {class_name}(object):\n"
        "    pass\n"
    )

    assert patch_rope_validation_config(path, class_name) is True
    assert patch_rope_validation_config(path, class_name) is False

    namespace = {}
    exec(path.read_text(), namespace)

    target = namespace[class_name]
    assert (
        frozenset(target.ignore_keys_at_rope_validation)
        == ROPE_VALIDATION_IGNORE_KEYS
    )
    assert not hasattr(
        namespace["UnrelatedConfig"],
        "ignore_keys_at_rope_validation",
    )


def test_rope_validation_patch_handles_multiline_class_header(
    tmp_path: Path,
):
    path = tmp_path / "config.py"
    path.write_text(
        "class Qwen4ExpTextConfig(\n"
        "    object,\n"
        "):\n"
        "    model_type = 'qwen4_exp_text'\n"
    )

    assert patch_rope_validation_config(
        path,
        "Qwen4ExpTextConfig",
    ) is True

    compile(path.read_text(), str(path), "exec")


def test_rope_validation_patch_fails_closed_on_conflicting_value(
    tmp_path: Path,
):
    path = tmp_path / "config.py"
    path.write_text(
        "class Qwen4ExpTextConfig(object):\n"
        "    ignore_keys_at_rope_validation = {'different_key'}\n"
    )

    with pytest.raises(RuntimeError, match="unexpected value"):
        patch_rope_validation_config(
            path,
            "Qwen4ExpTextConfig",
        )


def test_rope_validation_patch_requires_exact_target(tmp_path: Path):
    path = tmp_path / "config.py"
    path.write_text("class OtherConfig(object):\n    pass\n")

    with pytest.raises(RuntimeError, match="found 0"):
        patch_rope_validation_config(
            path,
            "Qwen4ExpTextConfig",
        )


def test_qwen3vl_video_doc_patch(tmp_path: Path):
    path = tmp_path / "video.py"
    path.write_text(
        'class X:\n'
        '    r"""\n'
        '    merge_size (`int`, *optional*, defaults to 2):\n'
        '        The merge size of the vision encoder to llm encoder.\n'
        '    """\n'
    )

    assert patch_qwen3vl_video_docstring(path) is True
    assert patch_qwen3vl_video_docstring(path) is False

    text = path.read_text()
    assert "min_frames" in text
    assert "max_frames" in text
