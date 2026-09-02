import json
from pathlib import Path

import pytest

from scripts import workload_contract


def _campaign(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiments": {
                    "runtime_qualification": {
                        "workload_profile": "runtime_qualification_v1",
                        "warmup_requests": 3,
                        "measured_requests": 10,
                        "input_tokens": [32],
                        "output_tokens": 8,
                        "concurrency": [1],
                        "temperature": 0.0,
                        "repetitions": 3,
                    }
                }
            }
        )
    )


def test_campaign_contract_is_single_endpoint_and_versioned(tmp_path: Path):
    path = tmp_path / "campaign.json"
    _campaign(path)

    contract = workload_contract.load_experiment_contract(path)

    assert contract.workload_profile == "runtime_qualification_v1"
    assert contract.input_tokens == 32
    assert contract.output_tokens == 8
    assert contract.concurrency == 1
    assert contract.warmup_requests == 3
    assert contract.measured_requests == 10
    assert contract.repetitions == 3


def test_multi_endpoint_contract_is_rejected(tmp_path: Path):
    path = tmp_path / "campaign.json"
    _campaign(path)
    payload = json.loads(path.read_text())
    payload["experiments"]["runtime_qualification"]["input_tokens"] = [32, 64]
    path.write_text(json.dumps(payload))

    with pytest.raises(
        workload_contract.WorkloadContractError,
        match="exactly one",
    ):
        workload_contract.load_experiment_contract(path)


@pytest.mark.parametrize("index", (0, 4))
def test_repetition_index_must_match_campaign(tmp_path: Path, index: int):
    path = tmp_path / "campaign.json"
    _campaign(path)
    contract = workload_contract.load_experiment_contract(path)

    with pytest.raises(
        workload_contract.WorkloadContractError,
        match="1..3",
    ):
        workload_contract.validate_repetition_index(contract, index)


def test_chat_url_maps_to_runtime_tokenize_endpoint():
    assert workload_contract.derive_tokenize_url(
        "http://127.0.0.1:8000/v1/chat/completions"
    ) == "http://127.0.0.1:8000/tokenize"


def test_unknown_chat_path_requires_explicit_tokenize_url():
    with pytest.raises(
        workload_contract.WorkloadContractError,
        match="cannot derive",
    ):
        workload_contract.derive_tokenize_url(
            "http://127.0.0.1:8000/custom"
        )


def test_exact_prompt_builder_reaches_target_with_live_style_counter():
    profile = workload_contract.WorkloadProfile(
        name="test",
        seed="seed",
        bulk_fragment=" bulk",
        fine_fragments=(" x",),
    )

    def token_count(prompt: str) -> int:
        # Simulates fixed chat-template overhead plus one token per word.
        return 10 + len(prompt.split())

    prompt = workload_contract.build_exact_token_prompt(
        target_tokens=25,
        token_count=token_count,
        profile=profile,
    )

    assert token_count(prompt) == 25


def test_exact_prompt_builder_never_rounds_down():
    profile = workload_contract.WorkloadProfile(
        name="test",
        seed="seed",
        bulk_fragment=" bulk",
        fine_fragments=(" xx",),
    )

    def token_count(prompt: str) -> int:
        # Only even token counts are reachable.
        return 20 + 2 * len(prompt.split())

    with pytest.raises(
        workload_contract.WorkloadContractError,
        match="stalled",
    ):
        workload_contract.build_exact_token_prompt(
            target_tokens=23,
            token_count=token_count,
            profile=profile,
        )


def test_tokenize_prompt_parses_server_contract(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"count": 59, "max_model_len": 8192}

    observed = {}

    def post(url, json, timeout):
        observed.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(workload_contract.requests, "post", post)

    result = workload_contract.tokenize_prompt(
        "http://server/tokenize",
        "model",
        "prompt",
    )

    assert result.count == 59
    assert result.max_model_len == 8192
    assert observed["json"]["messages"][0]["content"] == "prompt"


def test_record_token_mismatches_are_detailed(tmp_path: Path):
    path = tmp_path / "campaign.json"
    _campaign(path)
    contract = workload_contract.load_experiment_contract(path)

    mismatches = workload_contract.record_token_mismatches(
        [
            {
                "ordinal": 0,
                "prompt_tokens": 31,
                "completion_tokens": 8,
            },
            {
                "ordinal": 1,
                "prompt_tokens": 32,
                "completion_tokens": 7,
            },
        ],
        contract=contract,
        phase="measured",
    )

    assert mismatches == [
        {
            "phase": "measured",
            "ordinal": 0,
            "field": "prompt_tokens",
            "expected": 32,
            "observed": 31,
        },
        {
            "phase": "measured",
            "ordinal": 1,
            "field": "completion_tokens",
            "expected": 8,
            "observed": 7,
        },
    ]
