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
