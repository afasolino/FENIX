import json
from pathlib import Path

from scripts import trace_campaign, trace_capture, trace_case, trace_contract


def write_campaign(path: Path):
    path.write_text(json.dumps({
        "model": {"ngram_size": 3, "heads_per_ngram": 8, "ngram_vocab_size_base": 20_000_000},
        "trace_analysis": {"ple_cache_capacities_gib": [0.125]},
    }))


def test_exact_case_publishes_correlated_artifacts(monkeypatch, tmp_path: Path):
    contract = trace_contract.TraceContract(
        experiment="trace_characterization",
        workload_profile="trace_characterization_v1",
        workload_corpus=tmp_path / "unused.json",
        seed=1,
        requests_per_input_length=2,
        input_tokens=(128,),
        output_tokens=4,
        temperature=0.0,
        exact_concurrency=(1,),
        aggregate_concurrency=(2,),
        repetitions=1,
    )
    case = trace_contract.planned_cases(contract)[0]
    ple_source = tmp_path / "ple_runtime.jsonl"
    moe_source = tmp_path / "moe_runtime.jsonl"
    server_log = tmp_path / "server.log"
    server_log.write_text('start\n')
    campaign = tmp_path / "campaign.json"; write_campaign(campaign)
    lane = tmp_path / "runtime_lane.json"
    lane.write_text(json.dumps({
        "lane_id":"lane",
        "runtime":{"repository":"runtime","revision":"r","container_image":"base"},
        "model":{"repository":"model","revision":"m"},
    }))

    monkeypatch.setattr(
        trace_case.trace_capture,
        "repository_state",
        lambda root: trace_capture.RepositoryState("commit", True, ()),
    )
    monkeypatch.setattr(
        trace_case.trace_capture,
        "require_trace_server",
        lambda path: trace_capture.TraceLaunchMetadata(True, ("1",), ("fenix-qwen38:candidate",)),
    )
    monkeypatch.setattr(
        trace_case.trace_capture,
        "resolve_image_id",
        lambda root, image: "sha256:image",
    )
    prepared = trace_contract.PreparedTracePrompts(
        prompts=("p0", "p1"), prompt_hashes=("h0", "h1"),
        prompt_set_sha256="set", prompt_tokens=128, max_model_len=8192,
        tokenize_url="http://tokenize",
    )
    monkeypatch.setattr(
        trace_case.trace_contract,
        "prepare_trace_prompts",
        lambda **kwargs: prepared,
    )

    def fake_benchmark(prompts, **kwargs):
        clients=[]
        ple_rows=[]
        moe_rows=[]
        for ordinal, _ in enumerate(prompts):
            start=1_000+ordinal*1_000; first=start+500; end=start+900
            rid=f"r{ordinal}"
            clients.append({
                "request_id":rid,"ordinal":ordinal,"concurrency":1,
                "start_ns":start,"first_token_ns":first,"end_ns":end,
                "prompt_tokens":128,"completion_tokens":4,
                "ttft_ms":0.5,"e2e_ms":0.9,"tpot_ms":0.1,"decode_tokens_s":10.0,
            })
            ple_rows.append({
                "kind":"address_batch","step_id":ordinal,"address_known_ns":start+100,
                "input_ids":[1],"query_start_loc":[0,1],"ngram_context":[[]],
                "physical_row_ids":[[100+ordinal+i for i in range(16)]],"row_bytes":160,
            })
            moe_rows.append({
                "timestamp_ns":start+100,"step_id":ordinal,"layer":"model.layers.0.mlp",
                "selected_expert_ids":[1],"cache_hit":[False],
                "transfer_expert_ids":[1],"transfer_bytes":4096,
                "resident_expert_ids":[1],
            })
        ple_source.write_text(''.join(json.dumps(x)+'\n' for x in ple_rows))
        moe_source.write_text(''.join(json.dumps(x)+'\n' for x in moe_rows))
        with server_log.open('a') as stream: stream.write('case complete\n')
        return clients, 1.0

    monkeypatch.setattr(trace_case, "run_prompt_benchmark", fake_benchmark)
    out = tmp_path / "out"
    destination = trace_case.run_case(
        root=tmp_path, contract=contract, case=case, campaign_path=campaign,
        runtime_lane_path=lane, server_log=server_log, out_root=out,
        ple_source=ple_source, moe_source=moe_source,
        url="u", model="m", tokenize_url=None, settle_ms=0,
    )
    evidence=json.loads((destination/'evidence.json').read_text())
    assert evidence['trace_valid'] is True
    assert evidence['case']['correlation_mode']=='exact_request_correlation'
    assert (destination/'ple_normalized.jsonl').is_file()
    assert (destination/'moe_normalized.jsonl').is_file()
    assert (destination/'joint_characterization.json').is_file()
