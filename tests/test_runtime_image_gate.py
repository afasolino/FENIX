import json
from pathlib import Path

from qualification.runtime_lane import inspect_image_smoke_report


def _config() -> dict:
    return {
        "lane_id": "lane",
        "runtime": {"revision": "runtime-rev", "container_image": "base"},
        "target": {"gpu_name_contains": "RTX A6000", "minimum_compute_capability": 8.0},
        "execution": {
            "runtime_image": "fenix-qwen38:locked",
            "image_smoke_report": "results/raw/runtime_qualification/image_smoke.json",
        },
    }


def test_image_smoke_gate_accepts_matching_passed_report(tmp_path: Path):
    path = tmp_path / "results/raw/runtime_qualification/image_smoke.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "lane_id": "lane",
        "runtime_source_revision": "runtime-rev",
        "base_container_image": "base",
        "image": "fenix-qwen38:locked",
        "passed": True,
        "observed": {
            "cuda_available": True,
            "device": "NVIDIA RTX A6000",
            "compute_capability": [8, 6],
            "sum": 523776.0,
        },
    }))
    result = inspect_image_smoke_report(tmp_path, _config())
    assert result["passed"] is True


def test_image_smoke_gate_fails_closed_when_report_is_missing(tmp_path: Path):
    result = inspect_image_smoke_report(tmp_path, _config())
    assert result["passed"] is False
    assert result["failures"] == ["image_smoke_report_missing"]
