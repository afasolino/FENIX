from scripts.robust_trace_campaign import validate_client_records_variable


def test_variable_trace_validation_accepts_natural_early_stop():
    prompts = [
        {"rendered_prompt_tokens": 301, "max_output_tokens": 128},
        {"rendered_prompt_tokens": 97, "max_output_tokens": 128},
    ]
    clients = [
        {
            "ordinal": 0,
            "prompt_tokens": 301,
            "completion_tokens": 17,
            "concurrency": 1,
            "start_ns": 1,
            "end_ns": 2,
        },
        {
            "ordinal": 1,
            "prompt_tokens": 97,
            "completion_tokens": 128,
            "concurrency": 1,
            "start_ns": 3,
            "end_ns": 4,
        },
    ]
    assert validate_client_records_variable(clients, prompts, concurrency=1) == []


def test_variable_trace_validation_rejects_token_drift_or_zero_output():
    prompts = [{"rendered_prompt_tokens": 301, "max_output_tokens": 128}]
    clients = [
        {
            "ordinal": 0,
            "prompt_tokens": 300,
            "completion_tokens": 0,
            "concurrency": 1,
            "start_ns": 1,
            "end_ns": 2,
        }
    ]
    reasons = validate_client_records_variable(clients, prompts, concurrency=1)
    assert "prompt_tokens_mismatch:0" in reasons
    assert "completion_tokens_outside_natural_range:0" in reasons
