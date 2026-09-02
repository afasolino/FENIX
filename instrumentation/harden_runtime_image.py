#!/usr/bin/env python3
"""Surgical compatibility fixes for the FENIX Qwen3.8 candidate image."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import transformers
import vllm


CUSTOM_VLLM_ENV_VARS = (
    "VLLM_WNA16_DYNAMIC_LRU",
    "VLLM_WNA16_MIXED_VMM_HOT_CACHE",
    "VLLM_WNA16_STATIC_HOT_CACHE_SIZE",
    "VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS",
    "VLLM_WNA16_STATIC_HOT_CACHE_FILE",
)


def patch_env_registry(path: Path) -> bool:
    marker = "# FENIX custom WNA16 environment registration"
    text = path.read_text()

    if marker in text:
        return False

    entries = "\n".join(
        (
            f'    "{name}": lambda: '
            f'__import__("os").environ.get("{name}"),'
        )
        for name in CUSTOM_VLLM_ENV_VARS
    )

    block = (
        "\n\n"
        f"{marker}\n"
        "environment_variables.update({\n"
        f"{entries}\n"
        "})\n"
    )

    updated = text.rstrip() + block
    compile(updated, str(path), "exec")
    path.write_text(updated)
    return True


def patch_qwen38_rope_config(path: Path) -> bool:
    desired = (
        '    ignore_keys_at_rope_validation = '
        '{"mrope_section", "mrope_interleaved"}\n'
    )

    text = path.read_text()
    if desired in text:
        return False

    lines = text.splitlines(keepends=True)

    start = None
    for index, line in enumerate(lines):
        if line.startswith("class Qwen3_8FlashNextTextConfig"):
            start = index
            break

    if start is None:
        raise RuntimeError(
            "Qwen3_8FlashNextTextConfig declaration not found"
        )

    end = start
    while end < len(lines):
        if lines[end].rstrip().endswith(":"):
            break
        end += 1

    if end >= len(lines):
        raise RuntimeError(
            "Qwen3_8FlashNextTextConfig declaration is unterminated"
        )

    lines.insert(end + 1, desired)

    updated = "".join(lines)
    compile(updated, str(path), "exec")
    path.write_text(updated)
    return True


def patch_qwen3vl_video_docstring(path: Path) -> bool:
    text = path.read_text()

    old = '''    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """
'''

    new = '''    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    min_frames (`int`, *optional*, defaults to 4):
        Minimum number of video frames used during frame sampling.
    max_frames (`int`, *optional*, defaults to 768):
        Maximum number of video frames used during frame sampling.
    """
'''

    if new in text:
        return False

    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Qwen3-VL video docstring anchor count {count}, expected 1"
        )

    updated = text.replace(old, new, 1)
    compile(updated, str(path), "exec")
    path.write_text(updated)
    return True


def verify() -> None:
    code = r'''
from vllm import envs

names = (
    "VLLM_WNA16_DYNAMIC_LRU",
    "VLLM_WNA16_MIXED_VMM_HOT_CACHE",
    "VLLM_WNA16_STATIC_HOT_CACHE_SIZE",
    "VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS",
    "VLLM_WNA16_STATIC_HOT_CACHE_FILE",
)

for name in names:
    assert name in envs.environment_variables, name

from vllm.models.qwen3_8_flash_next.config import (
    Qwen3_8FlashNextTextConfig,
)

assert Qwen3_8FlashNextTextConfig.ignore_keys_at_rope_validation == {
    "mrope_section",
    "mrope_interleaved",
}

from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)

doc = Qwen3VLVideoProcessor.__doc__ or ""
assert "min_frames" in doc
assert "max_frames" in doc

print("FENIX_RUNTIME_HARDENING_OK")
'''

    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    output = completed.stdout + completed.stderr

    if completed.returncode != 0:
        raise RuntimeError(output)

    forbidden = (
        "Unrecognized keys in `rope_parameters`",
        "`min_frames` is part of Qwen3VLVideoProcessorInitKwargs",
        "`max_frames` is part of Qwen3VLVideoProcessorInitKwargs",
    )

    for message in forbidden:
        if message in output:
            raise RuntimeError(
                f"warning remained after hardening: {message}\n{output}"
            )

    print(output.strip())


def main() -> None:
    vllm_root = Path(vllm.__file__).resolve().parent
    transformers_root = Path(transformers.__file__).resolve().parent

    envs_path = vllm_root / "envs.py"
    qwen38_path = (
        vllm_root
        / "models"
        / "qwen3_8_flash_next"
        / "config.py"
    )
    video_path = (
        transformers_root
        / "models"
        / "qwen3_vl"
        / "video_processing_qwen3_vl.py"
    )

    for path in (envs_path, qwen38_path, video_path):
        if not path.is_file():
            raise RuntimeError(f"expected runtime file missing: {path}")

    print("env registry patched:", patch_env_registry(envs_path))
    print(
        "Qwen3.8 RoPE config patched:",
        patch_qwen38_rope_config(qwen38_path),
    )
    print(
        "Qwen3-VL video docs patched:",
        patch_qwen3vl_video_docstring(video_path),
    )

    verify()


if __name__ == "__main__":
    main()
