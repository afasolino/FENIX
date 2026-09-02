from analysis.capacity_tradeoff import project_budget, simulate_lru


def test_lru_capacity_changes_miss_count():
    sequence = [(0, 0), (0, 1), (0, 0), (0, 2), (0, 0)]
    selections, hits, misses = simulate_lru(sequence, capacity=2)

    assert selections == 5
    assert hits == 2
    assert misses == 3


def test_externalized_ple_never_reduces_expert_capacity():
    sequence = [(0, 0), (0, 1), (0, 0)]
    baseline = project_budget(
        sequence,
        host_budget_gib=1,
        expert_bytes=128 * 1024**2,
        ple_host_bytes=512 * 1024**2,
        placement="ple_in_host_dram",
    )
    externalized = project_budget(
        sequence,
        host_budget_gib=1,
        expert_bytes=128 * 1024**2,
        ple_host_bytes=512 * 1024**2,
        placement="ple_externalized",
    )

    assert externalized.expert_capacity >= baseline.expert_capacity
    assert (
        externalized.expert_storage_bytes_per_selection
        <= baseline.expert_storage_bytes_per_selection
    )


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
