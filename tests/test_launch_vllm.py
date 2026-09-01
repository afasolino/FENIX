from pathlib import Path

from scripts.launch_vllm import build_command


def test_tp1_forces_multiprocess_executor():
    _, command = build_command(
        model_directory=Path("/model"),
        runtime_directory=Path("/runtime"),
        trace_directory=Path("/traces"),
        gpu_ids=["0"],
        port=8000,
        cpu_offload_gib=64,
        hot_experts=64,
        max_model_len=8192,
        max_num_seqs=1,
        trace_enabled=False,
    )

    index = command.index("--distributed-executor-backend")
    assert command[index + 1] == "mp"
