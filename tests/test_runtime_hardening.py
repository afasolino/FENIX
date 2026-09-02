from pathlib import Path

from instrumentation.harden_runtime_image import (
    patch_env_registry,
    patch_qwen38_rope_config,
    patch_qwen3vl_video_docstring,
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


def test_qwen38_rope_validation_patch(tmp_path: Path):
    path = tmp_path / "config.py"
    path.write_text(
        "class Qwen3_8FlashNextTextConfig(object):\n"
        "    pass\n"
    )

    assert patch_qwen38_rope_config(path) is True

    namespace = {}
    exec(path.read_text(), namespace)

    cls = namespace["Qwen3_8FlashNextTextConfig"]
    assert cls.ignore_keys_at_rope_validation == {
        "mrope_section",
        "mrope_interleaved",
    }


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
