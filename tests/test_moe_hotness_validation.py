from analysis import moe_hotness_validation as m


def test_uniform_occupancy_one_token():
    expected, std = m._uniform_occupancy(512, 10, 1)
    assert abs(expected - 10.0) < 1e-9
    assert std < 1e-6


def test_uniform_occupancy_256_tokens_is_nearly_full():
    expected, _ = m._uniform_occupancy(512, 10, 256)
    assert 508 < expected < 512


def test_uniform_concentration():
    c = m._concentration({i: 1 for i in range(512)}, [64, 128])
    assert abs(c["64"] - 0.125) < 1e-12
    assert abs(c["128"] - 0.25) < 1e-12


def test_token_atomic_lru_reuse():
    r = m.RequestTrace("r", "code", 0, 2, {}, {0: [tuple(range(10)), tuple(range(10))]}, {})
    hits, misses = m._simulate_lru([r], 16, num_layers=1)
    assert (hits, misses) == (10, 10)


def test_static_compulsory_first_miss():
    r = m.RequestTrace("r", "code", 0, 2, {}, {0: [tuple(range(10)), tuple(range(10))]}, {})
    hits, misses = m._simulate_static([r], [set(range(10))], num_layers=1)
    assert (hits, misses) == (10, 10)
