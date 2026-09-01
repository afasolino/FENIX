from analysis.ple_locality import exact_reuse_distances


def test_exact_reuse_distance_counts_unique_intervening_keys():
    distances, cold = exact_reuse_distances(["a", "b", "c", "a", "b", "a"])

    assert cold == 3
    assert distances == [2, 2, 1]
