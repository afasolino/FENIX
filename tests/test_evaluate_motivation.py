from analysis.evaluate_motivation import bootstrap_difference_interval


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
