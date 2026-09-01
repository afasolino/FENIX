from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINTS = (
    ("fetch_runtime", "scripts/fetch_runtime.py"),
    ("qualify_runtime", "scripts/qualify_runtime.py"),
    ("fetch_model", "scripts/fetch_model.py"),
)


@pytest.mark.parametrize(("module_name", "script_path"), ENTRYPOINTS)
def test_runtime_gate_cli_supports_module_invocation(
    module_name: str,
    script_path: str,
):
    completed = subprocess.run(
        [sys.executable, "-m", f"scripts.{module_name}", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


@pytest.mark.parametrize(("module_name", "script_path"), ENTRYPOINTS)
def test_runtime_gate_cli_supports_direct_script_invocation(
    module_name: str,
    script_path: str,
):
    completed = subprocess.run(
        [sys.executable, script_path, "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
