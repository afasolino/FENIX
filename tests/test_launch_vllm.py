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
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=1073741824,
        trace_enabled=False,
    )

    index = command.index("--distributed-executor-backend")
    assert command[index + 1] == "mp"



def test_launcher_uses_project_local_podman_cdi():
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
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=1073741824,
        trace_enabled=False,
    )
    assert "docker" not in command
    assert "scripts.fenix_podman" in command
    i = command.index("--device")
    assert command[i + 1] == "nvidia.com/gpu=all"



def test_launcher_overrides_image_entrypoint_and_uses_humming():
    _, command = build_command(
        model_directory=Path("/model"),
        runtime_directory=Path("/runtime"),
        trace_directory=Path("/traces"),
        gpu_ids=["0"],
        port=8000,
        cpu_offload_gib=40,
        hot_experts=0,
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=1073741824,
        trace_enabled=False,
    )
    i = command.index("--entrypoint")
    assert command[i + 1] == "vllm"
    image = command.index("fenix-qwen38:locked")
    assert command[image + 1] == "serve"
    assert command[image + 2] == "/model"
    backend = command.index("--moe-backend")
    assert command[backend + 1] == "humming"


def test_first_boot_environment_is_bounded():
    environment, _ = build_command(
        model_directory=Path("/model"),
        runtime_directory=Path("/runtime"),
        trace_directory=Path("/traces"),
        gpu_ids=["0"],
        port=8000,
        cpu_offload_gib=40,
        hot_experts=0,
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=1073741824,
        trace_enabled=False,
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert environment["VLLM_PLE_OFFLOAD_READY_TIMEOUT"] == "1200"
    assert environment["VLLM_WNA16_STATIC_HOT_CACHE_SIZE"] == "0"
