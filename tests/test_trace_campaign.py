import json
from pathlib import Path

import pytest

from scripts import trace_campaign, trace_capture, trace_case, trace_contract


def make_contract(requests=2):
    return trace_contract.TraceContract(
        experiment="trace_characterization",
        workload_profile="trace_characterization_v1",
        workload_corpus=Path("unused.json"),
        seed=1,
        requests_per_input_length=requests,
        input_tokens=(128,),
        output_tokens=4,
        temperature=0.0,
        exact_concurrency=(1,),
        aggregate_concurrency=(2,),
        repetitions=1,
    )


def test_prompt_benchmark_preserves_one_result_per_prompt(monkeypatch):
    def fake_run_one(url, model, prompt, max_tokens, temperature, request_id):
        return {
            "request_id": request_id,
            "start_ns": 1,
            "first_token_ns": 2,
            "end_ns": 3,
            "prompt_tokens": 128,
            "completion_tokens": 4,
            "ttft_ms": 1.0,
            "e2e_ms": 2.0,
            "tpot_ms": 0.5,
            "decode_tokens_s": 2.0,
            "prompt_value": prompt,
        }
    monkeypatch.setattr(trace_case.bench_openai, "run_one", fake_run_one)
    rows, wall = trace_case.run_prompt_benchmark(
        ["a", "b", "c"], url="u", model="m", max_tokens=4,
        temperature=0.0, concurrency=2,
    )
    assert [row["ordinal"] for row in rows] == [0, 1, 2]
    assert {row["prompt_value"] for row in rows} == {"a", "b", "c"}
    assert all(row["concurrency"] == 2 for row in rows)
    assert wall >= 0


def test_client_validation_fails_on_token_drift():
    reasons = trace_case.validate_client_records(
        [{
            "ordinal": 0, "prompt_tokens": 127, "completion_tokens": 4,
            "concurrency": 1, "start_ns": 1, "end_ns": 2,
        }],
        expected_requests=1, input_tokens=128, output_tokens=4, concurrency=1,
    )
    assert reasons == ["prompt_tokens_mismatch:0"]


def test_plan_has_all_cases():
    contract = trace_contract.TraceContract(
        experiment="trace_characterization",
        workload_profile="trace_characterization_v1",
        workload_corpus=Path("unused.json"),
        seed=1,
        requests_per_input_length=20,
        input_tokens=(128, 1024, 4096),
        output_tokens=256,
        temperature=0.0,
        exact_concurrency=(1,),
        aggregate_concurrency=(2, 4),
        repetitions=1,
    )
    plan = trace_campaign.plan(contract)
    assert plan["case_count"] == 9


def test_verify_complete_checks_hashes_and_cross_case_provenance(tmp_path: Path):
    contract = make_contract(requests=1)
    out = tmp_path / "campaign"
    out.mkdir()
    for case in trace_contract.planned_cases(contract):
        case_dir = out / case.case_id
        case_dir.mkdir()
        artifact = case_dir / "client.jsonl"
        artifact.write_text("{}\n")
        evidence = {
            "trace_valid": True,
            "repository_commit": "abc",
            "campaign_sha256": "campaign",
            "runtime_lane": {"runtime_revision": "runtime", "model_revision": "model"},
            "launch": {"runtime_image": "fenix-qwen38:candidate", "runtime_image_id": "sha256:image"},
            "case": {
                "input_tokens": case.input_tokens,
                "concurrency": case.concurrency,
                "repetition_index": case.repetition_index,
                "correlation_mode": case.correlation_mode,
                "prompt_set_sha256": "same",
            },
            "artifacts_sha256": {"client.jsonl": trace_capture.sha256_file(artifact)},
        }
        (case_dir / "evidence.json").write_text(json.dumps(evidence))
    result = trace_campaign.verify_complete(out, contract)
    assert result["complete"] is True
    assert result["observed_cases"] == 2


def test_verify_complete_detects_artifact_mutation(tmp_path: Path):
    contract = make_contract(requests=1)
    out = tmp_path / "campaign"; out.mkdir()
    for case in trace_contract.planned_cases(contract):
        case_dir=out/case.case_id; case_dir.mkdir(); artifact=case_dir/'client.jsonl'; artifact.write_text('{}\n')
        payload={
            'trace_valid':True,'repository_commit':'abc','campaign_sha256':'campaign',
            'runtime_lane':{'runtime_revision':'runtime','model_revision':'model'},
            'launch':{'runtime_image':'fenix-qwen38:candidate','runtime_image_id':'sha256:image'},
            'case':{'input_tokens':case.input_tokens,'concurrency':case.concurrency,'repetition_index':case.repetition_index,'correlation_mode':case.correlation_mode,'prompt_set_sha256':'same'},
            'artifacts_sha256':{'client.jsonl':trace_capture.sha256_file(artifact)},
        }
        (case_dir/'evidence.json').write_text(json.dumps(payload))
    first = out / trace_contract.planned_cases(contract)[0].case_id / 'client.jsonl'
    first.write_text('{"mutated":true}\n')
    result=trace_campaign.verify_complete(out,contract)
    assert result['complete'] is False
    assert any('artifact_hash_mismatch' in item for item in result['failures'])


def test_ple_event_validation_enforces_versioned_rows_per_token(tmp_path: Path):
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({
        "model": {"ngram_size": 3, "heads_per_ngram": 8}
    }))
    good = [{
        "kind": "address_batch",
        "input_ids": [1],
        "physical_row_ids": [list(range(16))],
        "row_bytes": 160,
    }]
    bad = [{
        "kind": "address_batch",
        "input_ids": [1],
        "physical_row_ids": [[1, 2]],
        "row_bytes": 160,
    }]
    assert trace_case.validate_ple_events(good, campaign) == []
    assert trace_case.validate_ple_events(bad, campaign) == [
        "ple_rows_per_token_mismatch:0"
    ]
