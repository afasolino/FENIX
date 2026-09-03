import json
from pathlib import Path

from analysis.evaluate_motivation import bootstrap_difference_interval, evaluate


def test_bootstrap_interval_is_negative_for_clear_improvement():
    baseline = [10.0, 10.1, 9.9, 10.0]
    externalized = [8.0, 8.1, 7.9, 8.0]
    lower, upper = bootstrap_difference_interval(
        baseline,
        externalized,
        samples=2000,
        alpha=0.05,
    )
    assert lower < 0
    assert upper < 0


def _config(path: Path):
    path.write_text(json.dumps({
        "experiments": {
            "capacity_tradeoff": {
                "host_memory_budgets_gib": [64, 112],
                "minimum_measured_repetitions": 3,
                "budget_roles": {
                    "64": {"role": "strong", "motivation_eligible": True},
                    "112": {"role": "control", "motivation_eligible": False},
                },
            }
        },
        "motivation_gate": {
            "max_externalized_to_baseline_tpot_ratio": 0.9,
            "min_expert_storage_bytes_reduction": 0.2,
            "bootstrap_samples": 500,
            "bootstrap_alpha": 0.05,
        },
    }))


def _row(budget: int, placement: str, repetition: int):
    return {
        "measured": True,
        "host_budget_gib": budget,
        "placement": placement,
        "tpot_ms": 10.0 if placement == "ple_in_host_dram" else 8.0,
        "expert_storage_bytes_per_token": 1000 if placement == "ple_in_host_dram" else 700,
        "model_revision": "model",
        "runtime_revision": "runtime",
        "hardware_id": "gpu",
        "workload_id": "workload",
        "repetition": repetition,
    }


def test_saturation_control_cannot_establish_supported(tmp_path: Path):
    measured = tmp_path / "measured.json"
    rows = [
        _row(112, placement, repetition)
        for placement in ("ple_in_host_dram", "ple_externalized")
        for repetition in range(3)
    ]
    measured.write_text(json.dumps({"evidence_kind": "local_measured", "rows": rows}))
    config = tmp_path / "campaign.json"
    _config(config)

    result = evaluate(measured, config)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["budgets"][0]["passed_materiality"] is True
    assert result["budgets"][0]["passed"] is False


def test_treatment_budget_can_establish_supported(tmp_path: Path):
    measured = tmp_path / "measured.json"
    rows = [
        _row(64, placement, repetition)
        for placement in ("ple_in_host_dram", "ple_externalized")
        for repetition in range(3)
    ]
    measured.write_text(json.dumps({"evidence_kind": "local_measured", "rows": rows}))
    config = tmp_path / "campaign.json"
    _config(config)

    result = evaluate(measured, config)
    assert result["verdict"] == "SUPPORTED"
    assert result["proceed_to_hardware_architecture"] is True
