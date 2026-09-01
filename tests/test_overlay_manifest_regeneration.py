import hashlib
import json
from pathlib import Path

from instrumentation.prepare_runtime import regenerate_overlay_manifest


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_regenerated_overlay_manifest_matches_installer_semantics(tmp_path: Path):
    overlay = tmp_path / "runtime/vllm-overlay"
    (overlay / "v1").mkdir(parents=True)
    (overlay / "a.py").write_text("a")
    (overlay / "v1/b.py").write_text("b")
    (overlay / "SHA256SUMS.json").write_text('{"stale": "manifest"}')

    metadata = regenerate_overlay_manifest(tmp_path)

    manifest_path = overlay / "SHA256SUMS.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "a.py": _sha256(overlay / "a.py"),
        "v1/b.py": _sha256(overlay / "v1/b.py"),
    }

    assert manifest == expected
    assert metadata["file_count"] == 2
    assert metadata["sha256"] == _sha256(manifest_path)


def test_injected_tracer_imports_resolve_inside_vllm_package():
    text = (ROOT / "instrumentation/prepare_runtime_inplace.py").read_text()

    assert text.count("from vllm.fenix_trace_runtime import") == 3
    assert "from fenix_trace_runtime import" not in text
