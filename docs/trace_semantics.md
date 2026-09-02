# Trace semantics

Qwen3.8-Flash-Next uses n-gram size 3 and 8 heads per n-gram order: 16 PLE row
addresses per token.

`physical_row_id` is the global concatenated-table row after per-head offset.

`address_known_ns` is host CLOCK_MONOTONIC immediately after runtime row-ID
computation. `consumption_ns` must be populated from the correlated GPU
`fenix.ple.consume.*` NVTX/Nsight event; raw logs do not fabricate it.

The first exact request-ID trace campaign uses one active request at a time,
joining server events to the unique client interval on the same host monotonic
clock. Concurrent service traces are retained separately until scheduler
request IDs are exported directly.

Address verification in `analysis/ple_address.py` imports neither vLLM nor
SGLang.

MoE runtime records use the trace writer's host `timestamp_ns`, derived from
`time.monotonic_ns()`. For concurrency-one cases this timestamp is joined to the
unique client interval from `time.perf_counter_ns()` on the same Linux host.
The event is classified as prefill when it precedes the client's first
recognized generated delta and as decode at or after that point.

Trace campaign cases isolate the append-only `ple_runtime.jsonl` and
`moe_runtime.jsonl` streams by byte range. A byte window is valid only when it
is non-empty and ends on a complete JSONL record. Concurrent cases are
prevented by a repository-local campaign lock.
