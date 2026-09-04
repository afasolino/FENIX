from pathlib import Path
from analysis import moe_router_invariance as inv


def test_compare_exact(monkeypatch):
    payload = {
        "repository_commit": "abc",
        "runtime_image_id": "img",
        "prompt_set_sha256": "prompts",
        "stratum": "code",
        "client_shape": {0: (10, 5)},
        "fingerprints": {0: "same"},
    }
    monkeypatch.setattr(inv, "canonical_case", lambda path: dict(payload))
    monkeypatch.setattr(inv, "_hot_experts", lambda path: {"a.log": 16, "b.log": 32}[path.name])
    result = inv.compare([
        ("hot16", Path("a"), Path("a.log")),
        ("hot32", Path("b"), Path("b.log")),
    ])
    assert result["exact_invariance"] is True
    assert result["failures"] == []


def test_invariance_collector_imports():
    from scripts import moe_invariance_collect
    assert callable(moe_invariance_collect.main)
