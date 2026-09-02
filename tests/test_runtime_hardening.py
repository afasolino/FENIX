from pathlib import Path

import pytest

from instrumentation.harden_runtime_image import (
    ROPE_VALIDATION_IGNORE_KEYS,
    patch_env_registry,
    patch_qwen3vl_video_docstring,
    patch_rope_validation_config,
    resolve_rope_config_targets,
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


def _write_qwen38_config(path: Path, *, include_qwen4: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "class Qwen3_8FlashNextTextConfig(object):\n"
        "    pass\n"
    )
    if include_qwen4:
        text += (
            "\n"
            "class Qwen4ExpTextConfig(Qwen3_8FlashNextTextConfig):\n"
            "    pass\n"
        )
    path.write_text(text)


def test_rope_target_resolution_supports_pinned_colocated_layout(
    tmp_path: Path,
):
    qwen38 = (
        tmp_path
        / "models"
        / "qwen3_8_flash_next"
        / "config.py"
    )
    _write_qwen38_config(qwen38, include_qwen4=True)

    assert resolve_rope_config_targets(tmp_path) == (
        (qwen38, "Qwen3_8FlashNextTextConfig"),
        (qwen38, "Qwen4ExpTextConfig"),
    )


def test_rope_target_resolution_supports_newer_split_layout(
    tmp_path: Path,
):
    qwen38 = (
        tmp_path
        / "models"
        / "qwen3_8_flash_next"
        / "config.py"
    )
    _write_qwen38_config(qwen38, include_qwen4=False)

    qwen4 = tmp_path / "models" / "qwen4_exp" / "config.py"
    qwen4.parent.mkdir(parents=True, exist_ok=True)
    qwen4.write_text(
        "class Qwen4ExpTextConfig(object):\n"
        "    pass\n"
    )

    assert resolve_rope_config_targets(tmp_path) == (
        (qwen38, "Qwen3_8FlashNextTextConfig"),
        (qwen4, "Qwen4ExpTextConfig"),
    )


def test_rope_target_resolution_fails_without_qwen4_class(
    tmp_path: Path,
):
    qwen38 = (
        tmp_path
        / "models"
        / "qwen3_8_flash_next"
        / "config.py"
    )
    _write_qwen38_config(qwen38, include_qwen4=False)

    with pytest.raises(RuntimeError, match="supported runtime layouts"):
        resolve_rope_config_targets(tmp_path)


def test_colocated_targets_patch_both_exact_classes(tmp_path: Path):
    qwen38 = (
        tmp_path
        / "models"
        / "qwen3_8_flash_next"
        / "config.py"
    )
    _write_qwen38_config(qwen38, include_qwen4=True)

    for path, class_name in resolve_rope_config_targets(tmp_path):
        assert patch_rope_validation_config(path, class_name) is True

    namespace = {}
    exec(qwen38.read_text(), namespace)
    expected = ROPE_VALIDATION_IGNORE_KEYS

    assert frozenset(
        namespace[
            "Qwen3_8FlashNextTextConfig"
        ].ignore_keys_at_rope_validation
    ) == expected
    assert frozenset(
        namespace[
            "Qwen4ExpTextConfig"
        ].ignore_keys_at_rope_validation
    ) == expected
