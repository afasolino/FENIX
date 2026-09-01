from __future__ import annotations

import json
from pathlib import Path

from analysis.evaluate_motivation import evaluate


def _row(placement: str, repetition: int) -> dict[str, object]:
    return {
        "measured": True,
        "host_budget_gib": 96,
        "placement": placement,
        "tpot_ms": 10.0 if placement == "ple_in_host_dram" else 8.0,
        "expert_storage_bytes_per_token": (
            1000 if placement == "ple_in_host_dram" else 700
        ),
        "model_revision": "model-rev",
        "runtime_revision": "runtime-rev",
        "hardware_id": "hardware",
        "workload_id": "workload",
        "repetition": repetition,
    }


def test_missing_endpoint_metadata_is_inconclusive(tmp_path: Path):
    rows = [
        _row(placement, repetition)
        for placement in ("ple_in_host_dram", "ple_externalized")
        for repetition in range(3)
    ]
    rows[0].pop("runtime_revision")

    measured = tmp_path / "measured.json"
    measured.write_text(json.dumps({"evidence_kind": "local_measured", "rows": rows}))

    config = tmp_path / "campaign.json"
    config.write_text(
        json.dumps(
            {
                "experiments": {
                    "capacity_tradeoff": {"minimum_measured_repetitions": 3}
                },
                "motivation_gate": {
                    "max_externalized_to_baseline_tpot_ratio": 0.9,
                    "min_expert_storage_bytes_reduction": 0.2,
                    "bootstrap_samples": 500,
                    "bootstrap_alpha": 0.05,
                },
            }
        )
    )

    result = evaluate(measured, config)

    assert result["verdict"] == "INCONCLUSIVE"
    assert result["proceed_to_hardware_architecture"] is False
    assert "runtime_revision:missing=1" in result["reasons"][0]
