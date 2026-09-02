#!/usr/bin/env python3
"""Surgical compatibility fixes for the FENIX Qwen3.8 candidate image."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


CUSTOM_VLLM_ENV_VARS = (
    "VLLM_WNA16_DYNAMIC_LRU",
    "VLLM_WNA16_MIXED_VMM_HOT_CACHE",
    "VLLM_WNA16_STATIC_HOT_CACHE_SIZE",
    "VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS",
    "VLLM_WNA16_STATIC_HOT_CACHE_FILE",
)
ROPE_VALIDATION_IGNORE_KEYS = frozenset(
    {"mrope_section", "mrope_interleaved"}
)
ROPE_VALIDATION_ATTRIBUTE = "ignore_keys_at_rope_validation"


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


def _class_assignment_value(
    class_node: ast.ClassDef,
    attribute_name: str,
) -> object | None:
    for statement in class_node.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None

        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value

        if value is None:
            continue

        for target in targets:
            if isinstance(target, ast.Name) and target.id == attribute_name:
                try:
                    return ast.literal_eval(value)
                except (ValueError, TypeError) as exc:
                    raise RuntimeError(
                        f"{attribute_name} on {class_node.name} is not a "
                        "literal value"
                    ) from exc

    return None



def _top_level_class_names(path: Path) -> frozenset[str]:
    """Return top-level class names from one Python module."""

    if not path.is_file():
        return frozenset()

    tree = ast.parse(path.read_text(), filename=str(path))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    )


def resolve_rope_config_targets(
    vllm_root: Path,
) -> tuple[tuple[Path, str], ...]:
    """Resolve Qwen3.8/Qwen4Exp config classes for the installed vLLM layout.

    The pinned Qwen3.8 preview co-locates ``Qwen4ExpTextConfig`` in
    ``models/qwen3_8_flash_next/config.py``. Newer vLLM versions move that
    class to ``models/qwen4_exp/config.py``. FENIX supports both exact layouts
    and fails closed if neither contains the expected Qwen4Exp class.
    """

    qwen38_path = (
        vllm_root
        / "models"
        / "qwen3_8_flash_next"
        / "config.py"
    )
    if not qwen38_path.is_file():
        raise RuntimeError(
            f"expected Qwen3.8 runtime config missing: {qwen38_path}"
        )

    qwen38_classes = _top_level_class_names(qwen38_path)
    if "Qwen3_8FlashNextTextConfig" not in qwen38_classes:
        raise RuntimeError(
            "Qwen3_8FlashNextTextConfig declaration not found in "
            f"{qwen38_path}"
        )

    targets: list[tuple[Path, str]] = [
        (qwen38_path, "Qwen3_8FlashNextTextConfig")
    ]

    if "Qwen4ExpTextConfig" in qwen38_classes:
        targets.append((qwen38_path, "Qwen4ExpTextConfig"))
        return tuple(targets)

    split_path = (
        vllm_root
        / "models"
        / "qwen4_exp"
        / "config.py"
    )
    split_classes = _top_level_class_names(split_path)
    if "Qwen4ExpTextConfig" not in split_classes:
        raise RuntimeError(
            "Qwen4ExpTextConfig declaration not found in supported runtime "
            f"layouts: {qwen38_path} or {split_path}"
        )

    targets.append((split_path, "Qwen4ExpTextConfig"))
    return tuple(targets)


def patch_rope_validation_config(path: Path, class_name: str) -> bool:
    """Add the FENIX MRoPE validation ignore set to one exact config class."""

    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{path}: expected exactly one {class_name} class, "
            f"found {len(matches)}"
        )

    class_node = matches[0]
    existing = _class_assignment_value(
        class_node,
        ROPE_VALIDATION_ATTRIBUTE,
    )
    if existing is not None:
        try:
            existing_keys = frozenset(existing)
        except TypeError as exc:
            raise RuntimeError(
                f"{class_name}.{ROPE_VALIDATION_ATTRIBUTE} is not iterable"
            ) from exc

        if existing_keys == ROPE_VALIDATION_IGNORE_KEYS:
            return False

        raise RuntimeError(
            f"{class_name}.{ROPE_VALIDATION_ATTRIBUTE} already exists with "
            f"unexpected value {existing!r}"
        )

    if not class_node.body:
        raise RuntimeError(f"{class_name} has no class body")

    insertion_line = class_node.body[0].lineno - 1
    lines = text.splitlines(keepends=True)
    desired = (
        "    ignore_keys_at_rope_validation = "
        '{"mrope_section", "mrope_interleaved"}\n'
    )
    lines.insert(insertion_line, desired)

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

expected = {"mrope_section", "mrope_interleaved"}

from vllm.models.qwen3_8_flash_next.config import (
    Qwen3_8FlashNextTextConfig,
)

try:
    from vllm.models.qwen3_8_flash_next.config import Qwen4ExpTextConfig
except ImportError:
    from vllm.models.qwen4_exp.config import Qwen4ExpTextConfig

assert Qwen3_8FlashNextTextConfig.ignore_keys_at_rope_validation == expected
assert Qwen4ExpTextConfig.ignore_keys_at_rope_validation == expected

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
    import transformers
    import vllm

    vllm_root = Path(vllm.__file__).resolve().parent
    transformers_root = Path(transformers.__file__).resolve().parent

    envs_path = vllm_root / "envs.py"
    video_path = (
        transformers_root
        / "models"
        / "qwen3_vl"
        / "video_processing_qwen3_vl.py"
    )

    for path in (envs_path, video_path):
        if not path.is_file():
            raise RuntimeError(f"expected runtime file missing: {path}")

    rope_targets = resolve_rope_config_targets(vllm_root)

    print("env registry patched:", patch_env_registry(envs_path))
    for path, class_name in rope_targets:
        print(
            f"{class_name} RoPE config patched:",
            patch_rope_validation_config(path, class_name),
        )
    print(
        "Qwen3-VL video docs patched:",
        patch_qwen3vl_video_docstring(video_path),
    )

    verify()


if __name__ == "__main__":
    main()
