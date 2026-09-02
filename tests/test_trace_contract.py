import json
from pathlib import Path

import pytest

from scripts import trace_contract, workload_contract


def write_corpus(path: Path, seeds=3):
    path.write_text(json.dumps({
        "schema_version": 1,
        "profile": "trace_characterization_v1",
        "request_seeds": [f"Natural seed prompt {i}." for i in range(seeds)],
        "continuation_sentences": [
            "A deterministic sentence describes memory behavior under a bounded capacity.",
            "Another natural sentence discusses sparse routing and explicit transfer evidence.",
        ],
    }))


def campaign_payload(corpus_path):
    return {
        "experiments": {
            "trace_characterization": {
                "seed": 20260901,
                "requests_per_input_length": 3,
                "input_tokens": [128, 1024, 4096],
                "output_tokens": 256,
                "temperature": 0.0,
                "exact_request_correlation_concurrency": [1],
                "aggregate_service_concurrency": [2, 4],
                "repetitions": 1,
                "workload_profile": "trace_characterization_v1",
                "workload_corpus": str(corpus_path),
            }
        }
    }


def test_contract_plans_exact_predeclared_matrix(tmp_path: Path):
    corpus = tmp_path / "corpus.json"
    write_corpus(corpus)
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign_payload(corpus)))
    contract = trace_contract.load_trace_contract(path)
    cases = trace_contract.planned_cases(contract)
    assert len(cases) == 9
    assert {case.concurrency for case in cases} == {1, 2, 4}
    assert all(
        case.correlation_mode == "exact_request_correlation"
        for case in cases
        if case.concurrency == 1
    )
    assert all(
        case.correlation_mode == "aggregate_service"
        for case in cases
        if case.concurrency > 1
    )


def test_contract_rejects_false_exact_correlation(tmp_path: Path):
    corpus = tmp_path / "corpus.json"
    write_corpus(corpus)
    payload = campaign_payload(corpus)
    payload["experiments"]["trace_characterization"][
        "exact_request_correlation_concurrency"
    ] = [2]
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(trace_contract.TraceContractError, match="concurrency=1"):
        trace_contract.load_trace_contract(path)


def test_prepare_trace_prompts_are_exact_distinct_and_deterministic(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.json"
    write_corpus(corpus)
    contract = trace_contract.TraceContract(
        experiment="trace_characterization",
        workload_profile="trace_characterization_v1",
        workload_corpus=corpus,
        seed=7,
        requests_per_input_length=3,
        input_tokens=(128,),
        output_tokens=16,
        temperature=0.0,
        exact_concurrency=(1,),
        aggregate_concurrency=(2,),
        repetitions=1,
    )

    def fake_tokenize(url, model, prompt, timeout_s=60.0):
        return workload_contract.TokenizationResult(
            count=len(prompt.split()), max_model_len=8192
        )

    monkeypatch.setattr(workload_contract, "tokenize_prompt", fake_tokenize)
    first = trace_contract.prepare_trace_prompts(
        contract=contract,
        input_tokens=128,
        chat_url="http://host/v1/chat/completions",
        model="model",
    )
    second = trace_contract.prepare_trace_prompts(
        contract=contract,
        input_tokens=128,
        chat_url="http://host/v1/chat/completions",
        model="model",
    )
    assert first.prompt_tokens == 128
    assert len(first.prompts) == 3
    assert len(set(first.prompt_hashes)) == 3
    assert first.prompt_hashes == second.prompt_hashes
    assert first.prompt_set_sha256 == second.prompt_set_sha256


def test_case_selector_fails_closed_when_not_predeclared():
    contract = trace_contract.TraceContract(
        experiment="trace_characterization",
        workload_profile="trace_characterization_v1",
        workload_corpus=Path("unused.json"),
        seed=1,
        requests_per_input_length=1,
        input_tokens=(128,),
        output_tokens=1,
        temperature=0.0,
        exact_concurrency=(1,),
        aggregate_concurrency=(2,),
        repetitions=1,
    )
    with pytest.raises(trace_contract.TraceContractError, match="matched no"):
        trace_contract.select_cases(
            trace_contract.planned_cases(contract), concurrency=4
        )


def test_long_prompt_construction_has_bounded_tokenizer_calls(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.json"
    write_corpus(corpus, seeds=1)
    contract = trace_contract.TraceContract(
        experiment="trace_characterization",
        workload_profile="trace_characterization_v1",
        workload_corpus=corpus,
        seed=7,
        requests_per_input_length=1,
        input_tokens=(4096,),
        output_tokens=256,
        temperature=0.0,
        exact_concurrency=(1,),
        aggregate_concurrency=(2, 4),
        repetitions=1,
    )
    calls = 0

    def fake_tokenize(url, model, prompt, timeout_s=60.0):
        nonlocal calls
        calls += 1
        return workload_contract.TokenizationResult(
            count=len(prompt.split()), max_model_len=8192
        )

    monkeypatch.setattr(workload_contract, "tokenize_prompt", fake_tokenize)
    prepared = trace_contract.prepare_trace_prompts(
        contract=contract,
        input_tokens=4096,
        chat_url="http://host/v1/chat/completions",
        model="model",
    )
    assert prepared.prompt_tokens == 4096
    assert calls < 64
