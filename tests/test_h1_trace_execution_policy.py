from pathlib import Path

import pytest

from scripts import check_h1_trace_server_policy, launch_h1_h2_robust_server


def test_policy_launcher_forces_prefix_cache_off_after_model():
    env, command = launch_h1_h2_robust_server.build_policy_command(
        model_directory=Path("/model"),
        runtime_directory=Path("/runtime"),
        trace_directory=Path("/traces"),
        gpu_ids=["0"],
        port=8000,
        cpu_offload_gib=40.0,
        hot_experts=16,
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=1073741824,
        runtime_image="fenix-qwen38:h1-h2-robust-v1",
        expandable_segments=False,
    )
    assert env["FENIX_TRACE"] == "1"
    assert env["FENIX_PREFIX_CACHING"] == "0"
    assert "--enable-prefix-caching" not in command

    image_index = command.index("fenix-qwen38:h1-h2-robust-v1")
    assert command.index("FENIX_PREFIX_CACHING=0") < image_index

    serve_index = command.index("serve")
    assert command[serve_index + 1] == "/model"
    assert command[serve_index + 2] == "--no-enable-prefix-caching"


def _server_log(prefix_flag: str, *, malformed_order: bool = False) -> str:
    if malformed_order:
        serve_tail = f"serve {prefix_flag} /model --max-model-len 8192"
    else:
        serve_tail = f"serve /model {prefix_flag} --max-model-len 8192"
    return "\n".join(
        [
            '{"FENIX_TRACE": "1", "FENIX_PREFIX_CACHING": "0"}',
            (
                "python -m scripts.fenix_podman run --rm "
                "-e FENIX_TRACE=1 -e FENIX_PREFIX_CACHING=0 "
                "fenix-qwen38:h1-h2-robust-v1 "
                + serve_tail
            ),
            "Application startup complete.",
            "",
        ]
    )


def test_policy_checker_accepts_explicit_disable_after_model(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(_server_log("--no-enable-prefix-caching"))
    policy = tmp_path / "policy.json"
    policy.write_text(Path("configs/h1_h2_trace_execution_v1.json").read_text())
    result = check_h1_trace_server_policy.verify(log, policy)
    assert result["prefix_caching"] == "disabled"
    assert result["vllm_model_positional"] == "/model"
    assert result["server_max_model_len"] == 8192


def test_policy_checker_rejects_runtime_default(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(_server_log(""))
    policy = tmp_path / "policy.json"
    policy.write_text(Path("configs/h1_h2_trace_execution_v1.json").read_text())
    with pytest.raises(
        check_h1_trace_server_policy.TracePolicyError,
        match="--no-enable-prefix-caching",
    ):
        check_h1_trace_server_policy.verify(log, policy)


def test_policy_checker_rejects_flag_before_model(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        _server_log("--no-enable-prefix-caching", malformed_order=True)
    )
    policy = tmp_path / "policy.json"
    policy.write_text(Path("configs/h1_h2_trace_execution_v1.json").read_text())
    with pytest.raises(
        check_h1_trace_server_policy.TracePolicyError,
        match="model must be the first positional argument",
    ):
        check_h1_trace_server_policy.verify(log, policy)
