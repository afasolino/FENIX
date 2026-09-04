import math

from analysis.h1_workload_robustness import jaccard, js_divergence


def test_jaccard_and_js_identity():
    assert jaccard({1, 2}, {1, 2}) == 1.0
    assert js_divergence({1: 10, 2: 5}, {1: 10, 2: 5}) == 0.0


def test_jaccard_disjoint_and_js_is_symmetric_bounded():
    assert jaccard({1}, {2}) == 0.0
    left = js_divergence({1: 10}, {2: 10})
    right = js_divergence({2: 10}, {1: 10})
    assert left is not None and right is not None
    assert math.isclose(left, right)
    assert 0.0 <= left <= 1.0
    assert math.isclose(left, 1.0)
