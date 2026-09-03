from analysis.capacity_tradeoff import (
    distribute_expert_capacity,
    project_budget,
    simulate_layered_lru,
    simulate_lru,
)


def test_lru_capacity_changes_miss_count():
    sequence = [(0, 0), (0, 1), (0, 0), (0, 2), (0, 0)]
    selections, hits, misses = simulate_lru(sequence, capacity=2)

    assert selections == 5
    assert hits == 2
    assert misses == 3


def test_capacity_is_capped_at_finite_model_population():
    capacities = distribute_expert_capacity(
        100,
        num_hidden_layers=2,
        num_experts=3,
    )
    assert capacities == (3, 3)


def test_capacity_remainder_is_distributed_deterministically():
    capacities = distribute_expert_capacity(
        5,
        num_hidden_layers=3,
        num_experts=4,
    )
    assert capacities == (2, 2, 1)


def test_layered_lru_does_not_share_slots_across_layers():
    sequence = [(0, 0), (1, 0), (0, 1), (0, 0), (1, 0)]
    selections, hits, misses = simulate_layered_lru(
        sequence,
        capacities=(1, 1),
        num_experts=2,
    )
    assert selections == 5
    assert hits == 1
    assert misses == 4


def test_externalized_ple_never_reduces_effective_expert_capacity():
    sequence = [(0, 0), (0, 1), (0, 0)]
    common = dict(
        sequence=sequence,
        host_budget_gib=1,
        expert_bytes=128 * 1024**2,
        ple_host_bytes=512 * 1024**2,
        num_hidden_layers=1,
        num_experts=8,
    )
    baseline = project_budget(placement="ple_in_host_dram", **common)
    externalized = project_budget(placement="ple_externalized", **common)

    assert externalized.expert_capacity >= baseline.expert_capacity
    assert (
        externalized.expert_storage_bytes_per_selection
        <= baseline.expert_storage_bytes_per_selection
    )


def test_projection_reports_nominal_and_effective_capacity_separately():
    row = project_budget(
        [(0, 0)],
        2,
        128 * 1024**2,
        0,
        "ple_externalized",
        num_hidden_layers=1,
        num_experts=4,
    )
    assert row.nominal_expert_capacity == 16
    assert row.expert_capacity == 4
    assert row.expert_population == 4
    assert row.saturated is True


def test_runtime_layer_prefix_is_accepted():
    from analysis.expert_locality import parse_layer_id

    assert parse_layer_id(7) == 7
    assert parse_layer_id("7") == 7
    assert (
        parse_layer_id(
            "language_model.model.layers.23.mlp.experts"
        )
        == 23
    )
